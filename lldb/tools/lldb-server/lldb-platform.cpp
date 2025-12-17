//===-- lldb-platform.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cerrno>
#if defined(__APPLE__)
#include <netinet/in.h>
#endif
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if !defined(_WIN32)
#include <sys/wait.h>
#endif
#include <fstream>
#include <optional>

#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include "LLDBServerUtilities.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerPlatform.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostGetOpt.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Host/Socket.h"
#include "lldb/Host/common/TCPSocket.h"
#if LLDB_ENABLE_POSIX
#include "lldb/Host/posix/DomainSocket.h"
#endif
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/UriParser.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;
using namespace llvm;

// The test suite makes many connections in parallel, let's not miss any.
// The highest this should get reasonably is a function of the number
// of target CPUs. For now, let's just use 100.
static const int backlog = 100;
static const int socket_error = -1;

namespace {
using namespace llvm::opt;

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "PlatformOptions.inc"
#undef OPTION
};

#define OPTTABLE_STR_TABLE_CODE
#include "PlatformOptions.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "PlatformOptions.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

static constexpr opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "PlatformOptions.inc"
#undef OPTION
};

class PlatformOptTable : public opt::GenericOptTable {
public:
  PlatformOptTable()
      : opt::GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable) {}

  void PrintHelp(llvm::StringRef Name) {
    std::string Usage =
        (Name + " [options] --listen <[host]:port> [[--] program args...]")
            .str();

    std::string Title = "lldb-server platform";

    OptTable::printHelp(llvm::outs(), Usage.c_str(), Title.c_str());

    llvm::outs() << R"(
DESCRIPTION
  Acts as a platform server for remote debugging. When LLDB clients connect,
  the platform server handles platform operations (file transfers, process
  launching) and spawns debug server instances (lldb-server gdbserver) to
  handle actual debugging sessions.

  By default, the server exits after handling one connection. Use --server
  to keep running and accept multiple connections sequentially.

EXAMPLES
  # Listen on port 1234, exit after first connection
  lldb-server platform --listen tcp://0.0.0.0:1234

  # Listen on port 5555, accept multiple connections
  lldb-server platform --server --listen tcp://localhost:5555

  # Listen on Unix domain socket
  lldb-server platform --listen unix:///tmp/lldb-server.sock

)";
  }
};
} // namespace

#if defined(__APPLE__)
#define LOW_PORT (IPPORT_RESERVED)
#define HIGH_PORT (IPPORT_HIFIRSTAUTO)
#else
#define LOW_PORT (1024u)
#define HIGH_PORT (49151u)
#endif

#if !defined(_WIN32)
// Watch for signals
static void signal_handler(int signo) {
  switch (signo) {
  case SIGHUP:
    // Use SIGINT first, if that does not work, use SIGHUP as a last resort.
    // And we should not call exit() here because it results in the global
    // destructors to be invoked and wreaking havoc on the threads still
    // running.
    llvm::errs() << "SIGHUP received, exiting lldb-server...\n";
    abort();
    break;
  }
}
#endif

static void display_usage(PlatformOptTable &Opts, const char *progname,
                          const char *subcommand) {
  std::string Name =
      (llvm::sys::path::filename(progname) + " " + subcommand).str();
  Opts.PrintHelp(Name);
}

static Status parse_listen_host_port(Socket::SocketProtocol &protocol,
                                     const std::string &listen_host_port,
                                     std::string &address,
                                     uint16_t &platform_port,
                                     std::string &gdb_address,
                                     const uint16_t gdbserver_port) {
  std::string hostname;
  // Try to match socket name as URL - e.g., tcp://localhost:5555
  if (std::optional<URI> uri = URI::Parse(listen_host_port)) {
    if (!Socket::FindProtocolByScheme(uri->scheme.str().c_str(), protocol)) {
      return Status::FromErrorStringWithFormat(
          "Unknown protocol scheme \"%s\".", uri->scheme.str().c_str());
    }
    if (protocol == Socket::ProtocolTcp) {
      hostname = uri->hostname;
      if (uri->port) {
        platform_port = *(uri->port);
      }
    } else
      address = listen_host_port.substr(uri->scheme.size() + strlen("://"));
  } else {
    // Try to match socket name as $host:port - e.g., localhost:5555
    llvm::Expected<Socket::HostAndPort> host_port =
        Socket::DecodeHostAndPort(listen_host_port);
    if (!llvm::errorToBool(host_port.takeError())) {
      protocol = Socket::ProtocolTcp;
      hostname = host_port->hostname;
      platform_port = host_port->port;
    } else
      address = listen_host_port;
  }

  if (protocol == Socket::ProtocolTcp) {
    if (platform_port != 0 && platform_port == gdbserver_port) {
      return Status::FromErrorStringWithFormat(
          "The same platform and gdb ports %u.", platform_port);
    }
    address = llvm::formatv("[{0}]:{1}", hostname, platform_port).str();
    gdb_address = llvm::formatv("[{0}]:{1}", hostname, gdbserver_port).str();
  } else {
    if (gdbserver_port) {
      return Status::FromErrorStringWithFormat(
          "--gdbserver-port %u is redundant for non-tcp protocol %s.",
          gdbserver_port, Socket::FindSchemeByProtocol(protocol));
    }
  }
  return Status();
}

static Status save_socket_id_to_file(const std::string &socket_id,
                                     const FileSpec &file_spec) {
  FileSpec temp_file_spec(file_spec.GetDirectory().GetStringRef());
  Status error(llvm::sys::fs::create_directory(temp_file_spec.GetPath()));
  if (error.Fail())
    return Status::FromErrorStringWithFormat(
        "Failed to create directory %s: %s", temp_file_spec.GetPath().c_str(),
        error.AsCString());

  Status status;
  if (auto Err = llvm::writeToOutput(file_spec.GetPath(),
                                     [&socket_id](llvm::raw_ostream &OS) {
                                       OS << socket_id;
                                       return llvm::Error::success();
                                     }))
    return Status::FromErrorStringWithFormat(
        "Failed to atomically write file %s: %s", file_spec.GetPath().c_str(),
        llvm::toString(std::move(Err)).c_str());
  return status;
}

static Status ListenGdbConnectionsIfNeeded(
    const Socket::SocketProtocol protocol, std::unique_ptr<TCPSocket> &gdb_sock,
    const std::string &gdb_address, uint16_t &gdbserver_port) {
  if (protocol != Socket::ProtocolTcp)
    return Status();

  gdb_sock = std::make_unique<TCPSocket>(/*should_close=*/true);
  Status error = gdb_sock->Listen(gdb_address, backlog);
  if (error.Fail())
    return error;

  if (gdbserver_port == 0)
    gdbserver_port = gdb_sock->GetLocalPortNumber();

  return Status();
}

static llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>>
AcceptGdbConnectionsIfNeeded(const FileSpec &debugserver_path,
                             const Socket::SocketProtocol protocol,
                             std::unique_ptr<TCPSocket> &gdb_sock,
                             MainLoop &main_loop, const uint16_t gdbserver_port,
                             const lldb_private::Args &args) {
  if (protocol != Socket::ProtocolTcp)
    return std::vector<MainLoopBase::ReadHandleUP>();

  return gdb_sock->Accept(main_loop, [debugserver_path, gdbserver_port,
                                      &args](std::unique_ptr<Socket> sock_up) {
    Log *log = GetLog(LLDBLog::Platform);
    Status error;
    SharedSocket shared_socket(sock_up.get(), error);
    if (error.Fail()) {
      LLDB_LOGF(log, "gdbserver SharedSocket failed: %s", error.AsCString());
      return;
    }
    lldb::pid_t child_pid = LLDB_INVALID_PROCESS_ID;
    std::string socket_name;
    GDBRemoteCommunicationServerPlatform platform(
        debugserver_path, Socket::ProtocolTcp, gdbserver_port);
    error = platform.LaunchGDBServer(args, child_pid, socket_name,
                                     shared_socket.GetSendableFD());
    if (error.Success() && child_pid != LLDB_INVALID_PROCESS_ID) {
      error = shared_socket.CompleteSending(child_pid);
      if (error.Fail()) {
        Host::Kill(child_pid, SIGTERM);
        LLDB_LOGF(log, "gdbserver CompleteSending failed: %s",
                  error.AsCString());
        return;
      }
    }
  });
}

static void client_handle(GDBRemoteCommunicationServerPlatform &platform,
                          const lldb_private::Args &args) {
  if (!platform.IsConnected())
    return;

  if (args.GetArgumentCount() > 0) {
    lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
    std::string socket_name;
    Status error = platform.LaunchGDBServer(args, pid, socket_name,
                                            SharedSocket::kInvalidFD);
    if (error.Success())
      platform.SetPendingGdbServer(socket_name);
    else
      fprintf(stderr, "failed to start gdbserver: %s\n", error.AsCString());
  }

  bool interrupt = false;
  bool done = false;
  Status error;
  while (!interrupt && !done) {
    if (platform.GetPacketAndSendResponse(std::nullopt, error, interrupt,
                                          done) !=
        GDBRemoteCommunication::PacketResult::Success)
      break;
  }

  printf("Disconnected.\n");
}

static Status spawn_process(const char *progname, const FileSpec &prog,
                            const Socket *conn_socket, uint16_t gdb_port,
                            const lldb_private::Args &args,
                            const std::string &log_file,
                            const StringRef log_channels, MainLoop &main_loop,
                            bool multi_client) {
  Status error;
  SharedSocket shared_socket(conn_socket, error);
  if (error.Fail())
    return error;

  ProcessLaunchInfo launch_info;

  launch_info.SetExecutableFile(prog, false);
  launch_info.SetArg0(progname);
  Args &self_args = launch_info.GetArguments();
  self_args.AppendArgument(progname);
  self_args.AppendArgument(llvm::StringRef("platform"));
  self_args.AppendArgument(llvm::StringRef("--child-platform-fd"));
  self_args.AppendArgument(llvm::to_string(shared_socket.GetSendableFD()));
  launch_info.AppendDuplicateFileAction((int64_t)shared_socket.GetSendableFD(),
                                        (int64_t)shared_socket.GetSendableFD());
  if (gdb_port) {
    self_args.AppendArgument(llvm::StringRef("--gdbserver-port"));
    self_args.AppendArgument(llvm::to_string(gdb_port));
  }
  if (!log_file.empty()) {
    self_args.AppendArgument(llvm::StringRef("--log-file"));
    self_args.AppendArgument(log_file);
  }
  if (!log_channels.empty()) {
    self_args.AppendArgument(llvm::StringRef("--log-channels"));
    self_args.AppendArgument(log_channels);
  }
  if (args.GetArgumentCount() > 0) {
    self_args.AppendArgument("--");
    self_args.AppendArguments(args);
  }

  launch_info.SetLaunchInSeparateProcessGroup(false);

  // Set up process monitor callback based on whether we're in server mode.
  if (multi_client)
    // In server mode: empty callback (don't terminate when child exits).
    launch_info.SetMonitorProcessCallback([](lldb::pid_t, int, int) {});
  else
    // In single-client mode: terminate main loop when child exits.
    launch_info.SetMonitorProcessCallback([&main_loop](lldb::pid_t, int, int) {
      main_loop.AddPendingCallback(
          [](MainLoopBase &loop) { loop.RequestTermination(); });
    });

  // Copy the current environment.
  launch_info.GetEnvironment() = Host::GetEnvironment();

  launch_info.GetFlags().Set(eLaunchFlagDisableSTDIO);

  // Close STDIN, STDOUT and STDERR.
  launch_info.AppendCloseFileAction(STDIN_FILENO);
  launch_info.AppendCloseFileAction(STDOUT_FILENO);
  launch_info.AppendCloseFileAction(STDERR_FILENO);

  // Redirect STDIN, STDOUT and STDERR to "/dev/null".
  launch_info.AppendSuppressFileAction(STDIN_FILENO, true, false);
  launch_info.AppendSuppressFileAction(STDOUT_FILENO, false, true);
  launch_info.AppendSuppressFileAction(STDERR_FILENO, false, true);

  std::string cmd;
  self_args.GetCommandString(cmd);

  error = Host::LaunchProcess(launch_info);
  if (error.Fail())
    return error;

  lldb::pid_t child_pid = launch_info.GetProcessID();
  if (child_pid == LLDB_INVALID_PROCESS_ID)
    return Status::FromErrorString("invalid pid");

  LLDB_LOG(GetLog(LLDBLog::Platform), "lldb-platform launched '{0}', pid={1}",
           cmd, child_pid);

  error = shared_socket.CompleteSending(child_pid);
  if (error.Fail()) {
    Host::Kill(child_pid, SIGTERM);
    return error;
  }

  return Status();
}

static FileSpec GetDebugserverPath() {
  if (const char *p = getenv("LLDB_DEBUGSERVER_PATH")) {
    FileSpec candidate(p);
    if (FileSystem::Instance().Exists(candidate))
      return candidate;
  }
#if defined(__APPLE__)
  FileSpec candidate = HostInfo::GetSupportExeDir();
  candidate.AppendPathComponent("debugserver");
  if (FileSystem::Instance().Exists(candidate))
    return candidate;
  return FileSpec();
#else
  // On non-apple platforms, *we* are the debug server.
  return HostInfo::GetProgramFileSpec();
#endif
}

// main
int main_platform(int argc, char *argv[]) {
  const char *progname = argv[0];
  const char *subcommand = argv[1];
  argc--;
  argv++;
#if !defined(_WIN32)
  signal(SIGPIPE, SIG_IGN);
  signal(SIGHUP, signal_handler);
#endif

  // Special handling for 'help' as first argument.
  if (argc > 0 && strcmp(argv[0], "help") == 0) {
    PlatformOptTable Opts;
    display_usage(Opts, progname, subcommand);
    return EXIT_SUCCESS;
  }

  Status error;
  shared_fd_t fd = SharedSocket::kInvalidFD;
  uint16_t gdbserver_port = 0;
  FileSpec socket_file;

  PlatformOptTable Opts;
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  bool HasError = false;

  opt::InputArgList Args =
      Opts.parseArgs(argc, argv, OPT_UNKNOWN, Saver, [&](llvm::StringRef Msg) {
        WithColor::error() << Msg << "\n";
        HasError = true;
      });

  std::string Name =
      (llvm::sys::path::filename(progname) + " " + subcommand).str();
  std::string HelpText =
      "Use '" + Name + " --help' for a complete list of options.\n";

  if (HasError) {
    llvm::errs() << HelpText;
    return EXIT_FAILURE;
  }

  if (Args.hasArg(OPT_help)) {
    display_usage(Opts, progname, subcommand);
    return EXIT_SUCCESS;
  }

  // Parse arguments.
  std::string listen_host_port = Args.getLastArgValue(OPT_listen).str();
  std::string log_file = Args.getLastArgValue(OPT_log_file).str();
  StringRef log_channels = Args.getLastArgValue(OPT_log_channels);
  bool multi_client = Args.hasArg(OPT_server);
  [[maybe_unused]] bool debug = Args.hasArg(OPT_debug);
  [[maybe_unused]] bool verbose = Args.hasArg(OPT_verbose);

  if (Args.hasArg(OPT_socket_file)) {
    socket_file.SetFile(Args.getLastArgValue(OPT_socket_file),
                        FileSpec::Style::native);
  }

  if (Args.hasArg(OPT_gdbserver_port)) {
    if (!llvm::to_integer(Args.getLastArgValue(OPT_gdbserver_port),
                          gdbserver_port)) {
      WithColor::error() << "invalid --gdbserver-port value\n";
      return EXIT_FAILURE;
    }
  }

  if (Args.hasArg(OPT_child_platform_fd)) {
    uint64_t _fd;
    if (!llvm::to_integer(Args.getLastArgValue(OPT_child_platform_fd), _fd)) {
      WithColor::error() << "invalid --child-platform-fd value\n";
      return EXIT_FAILURE;
    }
    fd = (shared_fd_t)_fd;
  }

  if (!LLDBServerUtilities::SetupLogging(log_file, log_channels, 0))
    return -1;

  // Print usage and exit if no listening port is specified.
  if (listen_host_port.empty() && fd == SharedSocket::kInvalidFD) {
    WithColor::error() << "either --listen or --child-platform-fd is required\n"
                       << HelpText;
    return EXIT_FAILURE;
  }

  // Get remaining arguments for inferior.
  std::vector<llvm::StringRef> Inputs;
  for (opt::Arg *Arg : Args.filtered(OPT_INPUT))
    Inputs.push_back(Arg->getValue());
  if (opt::Arg *Arg = Args.getLastArg(OPT_REM)) {
    for (const char *Val : Arg->getValues())
      Inputs.push_back(Val);
  }

  lldb_private::Args inferior_arguments;
  if (!Inputs.empty()) {
    std::vector<const char *> args_ptrs;
    for (const auto &Input : Inputs)
      args_ptrs.push_back(Input.data());
    inferior_arguments.SetArguments(args_ptrs.size(), args_ptrs.data());
  }

  FileSpec debugserver_path = GetDebugserverPath();
  if (!debugserver_path) {
    WithColor::error(errs()) << "Could not find debug server executable.";
    return EXIT_FAILURE;
  }

  Log *log = GetLog(LLDBLog::Platform);
  if (fd != SharedSocket::kInvalidFD) {
    // Child process will handle the connection and exit.
    NativeSocket sockfd;
    error = SharedSocket::GetNativeSocket(fd, sockfd);
    if (error.Fail()) {
      LLDB_LOGF(log, "lldb-platform child: %s", error.AsCString());
      return socket_error;
    }

    std::unique_ptr<Socket> socket;
    if (gdbserver_port) {
      socket = std::make_unique<TCPSocket>(sockfd, /*should_close=*/true);
    } else {
#if LLDB_ENABLE_POSIX
      llvm::Expected<std::unique_ptr<DomainSocket>> domain_socket =
          DomainSocket::FromBoundNativeSocket(sockfd, /*should_close=*/true);
      if (!domain_socket) {
        LLDB_LOG_ERROR(log, domain_socket.takeError(),
                       "Failed to create socket: {0}");
        return socket_error;
      }
      socket = std::move(domain_socket.get());
#else
      WithColor::error() << "lldb-platform child: Unix domain sockets are not "
                            "supported on this platform.";
      return socket_error;
#endif
    }

    GDBRemoteCommunicationServerPlatform platform(
        debugserver_path, socket->GetSocketProtocol(), gdbserver_port);
    platform.SetConnection(
        std::make_unique<ConnectionFileDescriptor>(std::move(socket)));
    client_handle(platform, inferior_arguments);
    return EXIT_SUCCESS;
  }

  if (gdbserver_port != 0 &&
      (gdbserver_port < LOW_PORT || gdbserver_port > HIGH_PORT)) {
    WithColor::error() << llvm::formatv("Port number {0} is not in the "
                                        "valid user port range of {1} - {2}\n",
                                        gdbserver_port, LOW_PORT, HIGH_PORT);
    return EXIT_FAILURE;
  }

  Socket::SocketProtocol protocol = Socket::ProtocolUnixDomain;
  std::string address;
  std::string gdb_address;
  uint16_t platform_port = 0;
  error = parse_listen_host_port(protocol, listen_host_port, address,
                                 platform_port, gdb_address, gdbserver_port);
  if (error.Fail()) {
    printf("Failed to parse listen address: %s\n", error.AsCString());
    return socket_error;
  }

  std::unique_ptr<Socket> platform_sock = Socket::Create(protocol, error);
  if (error.Fail()) {
    printf("Failed to create platform socket: %s\n", error.AsCString());
    return socket_error;
  }
  error = platform_sock->Listen(address, backlog);
  if (error.Fail()) {
    printf("Failed to listen platform: %s\n", error.AsCString());
    return socket_error;
  }
  if (protocol == Socket::ProtocolTcp && platform_port == 0)
    platform_port =
        static_cast<TCPSocket *>(platform_sock.get())->GetLocalPortNumber();

  if (socket_file) {
    error = save_socket_id_to_file(
        protocol == Socket::ProtocolTcp
            ? (platform_port ? llvm::to_string(platform_port) : "")
            : address,
        socket_file);
    if (error.Fail()) {
      fprintf(stderr, "failed to write socket id to %s: %s\n",
              socket_file.GetPath().c_str(), error.AsCString());
      return EXIT_FAILURE;
    }
  }

  std::unique_ptr<TCPSocket> gdb_sock;
  // Update gdbserver_port if it is still 0 and protocol is tcp.
  error = ListenGdbConnectionsIfNeeded(protocol, gdb_sock, gdb_address,
                                       gdbserver_port);
  if (error.Fail()) {
    printf("Failed to listen gdb: %s\n", error.AsCString());
    return socket_error;
  }

  MainLoop main_loop;
  {
    llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>> platform_handles =
        platform_sock->Accept(
            main_loop, [progname, gdbserver_port, &inferior_arguments, log_file,
                        log_channels, &main_loop, multi_client,
                        &platform_handles](std::unique_ptr<Socket> sock_up) {
              printf("Connection established.\n");
              Status error = spawn_process(
                  progname, HostInfo::GetProgramFileSpec(), sock_up.get(),
                  gdbserver_port, inferior_arguments, log_file, log_channels,
                  main_loop, multi_client);
              if (error.Fail()) {
                Log *log = GetLog(LLDBLog::Platform);
                LLDB_LOGF(log, "spawn_process failed: %s", error.AsCString());
                WithColor::error()
                    << "spawn_process failed: " << error.AsCString() << "\n";
                if (!multi_client)
                  main_loop.RequestTermination();
              }
              if (!multi_client)
                platform_handles->clear();
            });
    if (!platform_handles) {
      printf("Failed to accept platform: %s\n",
             llvm::toString(platform_handles.takeError()).c_str());
      return socket_error;
    }

    llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>> gdb_handles =
        AcceptGdbConnectionsIfNeeded(debugserver_path, protocol, gdb_sock,
                                     main_loop, gdbserver_port,
                                     inferior_arguments);
    if (!gdb_handles) {
      printf("Failed to accept gdb: %s\n",
             llvm::toString(gdb_handles.takeError()).c_str());
      return socket_error;
    }

    main_loop.Run();
  }

  fprintf(stderr, "lldb-server exiting...\n");

  return EXIT_SUCCESS;
}
