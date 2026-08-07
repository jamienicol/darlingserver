/**
 * This file is part of Darling.
 *
 * Copyright (C) 2026 Darling developers
 *
 * Darling is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Darling is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Darling.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * Standalone mode: run a single Mach-O program with no container.
 *
 * The normal Darling startup path (`darling shell ...`) needs to be setuid root so that it can
 * create a mount namespace, mount an overlayfs over the prefix, mount a fresh procfs in a new PID
 * namespace, and boot launchd inside it. None of that is needed to run a self-contained command
 * line program:
 *
 *  - Path translation into the Darling root is done entirely in userspace by
 *    `vchroot_expand()` in libsystem_kernel, which also special-cases `/Volumes/SystemRoot` to
 *    escape back out to the real Linux root. So no mounts are required to make `/usr/lib/...`
 *    resolve.
 *  - A program that doesn't use XPC never sends a Mach message to launchd. Its `mach_msg` traffic
 *    goes to its own task port and to the host port, both of which duct-tape serves internally.
 *  - The PID namespace only exists to give the container Darling-local PIDs. The server already
 *    distinguishes global IDs from namespace-local IDs and works fine when they're equal.
 *
 * So standalone mode keeps the parts that matter (the duct-taped XNU, the RPC handlers) and drops
 * the container: it binds a private socket, forks the program under mldr, and runs the normal
 * server event loop until the program exits. Nothing is setuid, nothing is unshared, nothing is
 * mounted, and no state outside the runtime directory is touched.
 *
 * Programs that need launchd (anything using XPC, CoreFoundation notifications, or a bootstrap
 * lookup) still need `darling shell`.
 */

#define _GNU_SOURCE 1

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <sys/signalfd.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <darling-config.h>

#include <darlingserver/config.hpp>
#include <darlingserver/monitor.hpp>
#include <darlingserver/server.hpp>
#include <darlingserver/utility.hpp>

namespace {

std::string runtimeDirectory;

void removeRuntimeDirectory(void) {
	if (runtimeDirectory.empty()) {
		return;
	}
	std::error_code error;
	std::filesystem::remove_all(runtimeDirectory, error);
}

static const char runtimeDirectoryPrefix[] = "darling-standalone.";

/**
 * Clean up after runs that were SIGKILLed, which get no chance to clean up after themselves.
 *
 * A directory is stale if no live process holds its PID. Anything we can't parse or remove is left
 * alone; this is opportunistic tidying, not a correctness requirement.
 */
void reapStaleRuntimeDirectories(const std::string& base) {
	std::error_code error;

	for (const auto& entry: std::filesystem::directory_iterator(base, error)) {
		auto name = entry.path().filename().string();

		if (name.compare(0, sizeof(runtimeDirectoryPrefix) - 1, runtimeDirectoryPrefix) != 0) {
			continue;
		}

		pid_t owner = -1;

		try {
			owner = std::stoi(name.substr(sizeof(runtimeDirectoryPrefix) - 1));
		} catch (...) {
			continue;
		}

		if (owner > 0 && kill(owner, 0) != 0 && errno == ESRCH) {
			std::filesystem::remove_all(entry.path(), error);
		}
	}
}

/**
 * Pick a private directory to hold the RPC socket and the server log.
 *
 * This deliberately does not live in the Darling root: standalone mode treats the root as
 * read-only, and concurrent standalone runs must not share a socket.
 */
std::string createRuntimeDirectory(void) {
	const char* base = getenv("XDG_RUNTIME_DIR");

	if (!base || base[0] != '/') {
		base = "/tmp";
	}

	reapStaleRuntimeDirectories(base);

	std::string path = std::string(base) + "/" + runtimeDirectoryPrefix + std::to_string(getpid());

	std::error_code error;
	std::filesystem::remove_all(path, error);

	if (mkdir(path.c_str(), 0700) != 0) {
		std::cerr << "darlingserver: failed to create runtime directory " << path << ": " << strerror(errno) << std::endl;
		exit(1);
	}

	return path;
}

/**
 * Resolve the program named on the command line to a Linux path, which is what mldr opens.
 *
 * The cases, in order:
 *  - spelled out inside the root already (`<root>/bin/echo`) -> use as-is
 *  - resolves inside the root (`/bin/echo`) -> `<root>/bin/echo`. The Darling root wins over the
 *    host, matching `darling shell`, so `/bin/echo` means macOS echo rather than coreutils echo.
 *  - exists on the host (`./build/prog`) -> the absolute host path
 *
 * mldr turns whichever of these we pick back into the vchroot-relative path the program sees as
 * its argv[0] (`vchroot_unexpand_interpreter`), including via `/Volumes/SystemRoot` for paths
 * outside the root, so there is nothing left for us to translate.
 */
bool resolveExecutablePath(const std::string& root, const std::string& path, std::string& out) {
	std::error_code error;

	if (path.length() > root.length() && path[root.length()] == '/' && path.compare(0, root.length(), root) == 0) {
		out = path;
		return true;
	}

	if (path[0] == '/' && std::filesystem::exists(root + path, error)) {
		out = root + path;
		return true;
	}

	auto absolute = std::filesystem::absolute(path, error);

	if (!error) {
		auto absoluteString = absolute.lexically_normal().string();

		if (std::filesystem::exists(absoluteString, error)) {
			out = absoluteString;
			return true;
		}
	}

	return false;
}

/**
 * Hand off to mldr, which loads the Mach-O and jumps to dyld.
 *
 * Container mode goes through the `vchroot` helper here, which chdir's into the prefix before
 * exec'ing the real program. Standalone mode doesn't: the server hands the process its vchroot at
 * check-in instead (see Server::setDefaultVchrootDirectory), so the caller's working directory
 * survives and relative paths still mean what they meant in the invoking shell.
 */
[[noreturn]]
void execViaMldr(const std::string& root, const std::string& socketPath, const std::string& program, char** programArgs, pid_t supervisorPID) {
	// Never outlive the supervisor: without it there is nobody to answer this process's Mach traps,
	// so it would hang forever on the first RPC. PR_SET_PDEATHSIG follows the forking *thread*,
	// which is the one that goes on to run the event loop, and it survives the exec below.
	if (prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0) != 0) {
		std::cerr << "darlingserver: failed to set parent-death signal: " << strerror(errno) << std::endl;
		_exit(127);
	}

	// ...and close the race where the supervisor died between the fork and the prctl above
	if (getppid() != supervisorPID) {
		_exit(128 + SIGKILL);
	}

	// The supervisor blocks the signals it forwards so that it can consume them through a signalfd.
	// That mask is inherited across fork and exec, so it has to be undone here or the program would
	// silently ignore SIGINT and friends.
	sigset_t emptySet;
	sigemptyset(&emptySet);
	pthread_sigmask(SIG_SETMASK, &emptySet, NULL);

	// Invoke mldr the way a shebang line would: argv[1] is the Mach-O to load, and mldr shifts argv
	// down by one so the program sees it as its own argv[0]. Going through this path (rather than
	// the `mldr!<path>` form) is what makes mldr unexpand argv[0] back into a vchroot path.
	std::vector<char*> args;
	args.push_back(const_cast<char*>("mldr"));
	args.push_back(const_cast<char*>(program.c_str()));

	for (char** arg = programArgs; *arg != NULL; ++arg) {
		args.push_back(*arg);
	}

	args.push_back(NULL);

	setenv("__mldr_DYLD_ROOT_PATH", root.c_str(), 1);
	setenv("__mldr_sockpath", socketPath.c_str(), 1);

	execv(std::string(DarlingServer::Config::defaultMldrPath).c_str(), args.data());

	std::cerr << "darlingserver: failed to exec mldr: " << strerror(errno) << std::endl;
	_exit(127);
}

void usage(void) {
	std::cerr <<
		"usage: darlingserver --standalone [--root <path>] <program> [args...]\n"
		"\n"
		"Run a single Mach-O program with no container: no root privileges, no namespaces,\n"
		"no mounts, and no launchd. Suitable for self-contained command line programs;\n"
		"anything that needs XPC or a bootstrap lookup requires `darling shell` instead.\n"
		"\n"
		"  --root <path>  Darling root to vchroot into (default: $DARLING_ROOT, else\n"
		"                 " LIBEXEC_PATH ")\n"
		"\n"
		"<program> may be a path on the Linux filesystem, a path inside the root, or a\n"
		"path relative to the root (e.g. /bin/echo).\n";
}

} // anonymous namespace

int standaloneMain(int argc, char** argv) {
	// argv[0] is the darlingserver path, argv[1] is `--standalone`
	int argi = 2;
	std::string root;

	if (const char* fromEnv = getenv("DARLING_ROOT")) {
		root = fromEnv;
	} else {
		root = LIBEXEC_PATH;
	}

	while (argi < argc && argv[argi][0] == '-' && strcmp(argv[argi], "--") != 0) {
		if (strcmp(argv[argi], "--root") == 0) {
			if (argi + 1 >= argc) {
				std::cerr << "darlingserver: --root requires an argument" << std::endl;
				return 1;
			}
			root = argv[argi + 1];
			argi += 2;
		} else if (strcmp(argv[argi], "--help") == 0 || strcmp(argv[argi], "-h") == 0) {
			usage();
			return 0;
		} else {
			std::cerr << "darlingserver: unrecognized standalone option: " << argv[argi] << std::endl;
			return 1;
		}
	}

	if (argi < argc && strcmp(argv[argi], "--") == 0) {
		++argi;
	}

	if (argi >= argc) {
		usage();
		return 1;
	}

	// normalize the root: vchroot path comparisons assume no trailing slash
	while (root.length() > 1 && root.back() == '/') {
		root.pop_back();
	}

	if (!std::filesystem::is_directory(root + "/usr/libexec/darling")) {
		std::cerr << "darlingserver: " << root << " does not look like a Darling root "
		          << "(no usr/libexec/darling); pass --root or set DARLING_ROOT" << std::endl;
		return 1;
	}

	std::string program;

	if (!resolveExecutablePath(root, argv[argi], program)) {
		std::cerr << "darlingserver: " << argv[argi] << ": no such file, either in " << root << " or on the host" << std::endl;
		return 1;
	}

	runtimeDirectory = createRuntimeDirectory();

	std::string socketPath = runtimeDirectory + "/dserver.sock";
	std::string logPath = runtimeDirectory + "/dserver.log";

	// Take delivery of the signals a user would expect to interrupt the program through a signalfd
	// rather than a handler, so that forwarding them happens as ordinary code on the event loop
	// instead of async-signal-safe code racing it.
	//
	// This has to happen before anything spawns a thread: a process-directed signal is delivered to
	// any thread that hasn't blocked it, so a workqueue thread created before the mask was set would
	// take the SIGTERM with its default disposition and kill the supervisor out from under the child.
	// A new thread inherits the mask of its creator, so blocking here covers all of them. The child
	// inherits it too, across both the fork and the exec, and execViaMldr() undoes it there.
	sigset_t forwardedSignals;
	sigemptyset(&forwardedSignals);
	sigaddset(&forwardedSignals, SIGTERM);
	sigaddset(&forwardedSignals, SIGINT);
	sigaddset(&forwardedSignals, SIGQUIT);
	sigaddset(&forwardedSignals, SIGHUP);

	if (pthread_sigmask(SIG_BLOCK, &forwardedSignals, NULL) != 0) {
		std::cerr << "darlingserver: failed to block signals: " << strerror(errno) << std::endl;
		removeRuntimeDirectory();
		return 1;
	}

	int signalFD = signalfd(-1, &forwardedSignals, SFD_CLOEXEC | SFD_NONBLOCK);

	if (signalFD < 0) {
		std::cerr << "darlingserver: failed to create signalfd: " << strerror(errno) << std::endl;
		removeRuntimeDirectory();
		return 1;
	}

	// The server has to exist before the child runs: the constructor is what binds the socket, and
	// mldr connects to it before it has even loaded dyld.
	auto server = new DarlingServer::Server(root, socketPath, logPath);

	// Establish the vchroot server-side rather than running the `vchroot` helper in the child, so
	// that the program inherits the caller's working directory. Paths under the root still resolve
	// into it, and relative paths still resolve against the directory the user ran us from.
	int rootFD = open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);

	if (rootFD < 0) {
		std::cerr << "darlingserver: failed to open " << root << ": " << strerror(errno) << std::endl;
		removeRuntimeDirectory();
		return 1;
	}

	server->setDefaultVchrootDirectory(std::make_shared<DarlingServer::FD>(rootFD));

	pid_t supervisorPID = getpid();
	pid_t child = fork();

	if (child < 0) {
		std::cerr << "darlingserver: failed to fork: " << strerror(errno) << std::endl;
		removeRuntimeDirectory();
		return 1;
	} else if (child == 0) {
		execViaMldr(root, socketPath, program, &argv[argi + 1], supervisorPID);
	}

	// Watch the child with a pidfd through the server's own event loop, so that we find out it
	// exited without needing a second thread or a signal handler racing the loop.
	int pidfd = syscall(SYS_pidfd_open, child, 0);

	if (pidfd < 0) {
		std::cerr << "darlingserver: failed to open pidfd for child: " << strerror(errno) << std::endl;
		kill(child, SIGKILL);
		removeRuntimeDirectory();
		return 1;
	}

	auto childFD = std::make_shared<DarlingServer::FD>(pidfd);

	server->addMonitor(std::make_shared<DarlingServer::Monitor>(childFD, DarlingServer::Monitor::Event::Readable, false, true, [child](std::shared_ptr<DarlingServer::Monitor> monitor, DarlingServer::Monitor::Event events) {
		int status = 0;

		if (waitpid(child, &status, 0) < 0) {
			status = 0;
		}

		removeRuntimeDirectory();

		// _exit() rather than exit(): this callback runs on the server's event loop thread while
		// duct-tape microthreads and workqueue threads are still live, and running their static
		// destructors underneath them crashes. There is nothing left to flush.
		_exit(WIFSIGNALED(status) ? (128 + WTERMSIG(status)) : WEXITSTATUS(status));
	}));

	// Forward the blocked signals to the program and let it decide what to do with them. If it dies,
	// the pidfd monitor above reports the status; if it chooses to ignore them, we keep running,
	// which is what a shell would do.
	server->addMonitor(std::make_shared<DarlingServer::Monitor>(std::make_shared<DarlingServer::FD>(signalFD), DarlingServer::Monitor::Event::Readable, false, false, [child](std::shared_ptr<DarlingServer::Monitor> monitor, DarlingServer::Monitor::Event events) {
		struct signalfd_siginfo info;

		while (read(monitor->fd()->fd(), &info, sizeof(info)) == sizeof(info)) {
			kill(child, info.ssi_signo);
		}
	}));

	server->start();

	// start() only returns on error
	std::cerr << "darlingserver: server exited main loop" << std::endl;
	kill(child, SIGKILL);
	removeRuntimeDirectory();
	return 1;
}
