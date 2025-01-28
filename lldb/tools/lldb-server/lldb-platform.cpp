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

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include "LLDBServerUtilities.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerPlatform.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/HostGetOpt.h"
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
static int g_debug = 0;
static int g_verbose = 0;
static int g_server = 0;

// option descriptors for getopt_long_only()
static struct option g_long_options[] = {
    {"debug", no_argument, &g_debug, 1},
    {"verbose", no_argument, &g_verbose, 1},
    {"log-file", required_argument, nullptr, 'l'},
    {"log-channels", required_argument, nullptr, 'c'},
    {"listen", required_argument, nullptr, 'L'},
    {"gdbserver-port", required_argument, nullptr, 'P'},
    {"min-gdbserver-port", required_argument, nullptr, 'm'},
    {"max-gdbserver-port", required_argument, nullptr, 'M'},
    {"socket-file", required_argument, nullptr, 'f'},
    {"server", no_argument, &g_server, 1},
    {"child-platform-fd", required_argument, nullptr, 2},
    {nullptr, 0, nullptr, 0}};

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

static void display_usage(const char *progname, const char *subcommand) {
  fprintf(stderr, "Usage:\n  %s %s [--log-file log-file-name] [--log-channels "
                  "log-channel-list] [--port-file port-file-path] --server "
                  "--listen port\n",
          progname, subcommand);
  exit(0);
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
AcceptGdbConnectionsIfNeeded(const Socket::SocketProtocol protocol,
                             std::unique_ptr<TCPSocket> &gdb_sock,
                             MainLoop &main_loop, const uint16_t gdbserver_port,
                             const lldb_private::Args &args) {
  if (protocol != Socket::ProtocolTcp)
    return std::vector<MainLoopBase::ReadHandleUP>();

  return gdb_sock->Accept(main_loop, [gdbserver_port,
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
    GDBRemoteCommunicationServerPlatform platform(Socket::ProtocolTcp,
                                                  gdbserver_port);
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

static Status spawn_process(const char *progname, const Socket *conn_socket,
                            uint16_t gdb_port, const lldb_private::Args &args,
                            const std::string &log_file,
                            const StringRef log_channels, MainLoop &main_loop) {
  Status error;
  SharedSocket shared_socket(conn_socket, error);
  if (error.Fail())
    return error;

  ProcessLaunchInfo launch_info;

  FileSpec self_spec(progname, FileSpec::Style::native);
  launch_info.SetExecutableFile(self_spec, true);
  Args &self_args = launch_info.GetArguments();
  self_args.AppendArgument(llvm::StringRef("platform"));
  self_args.AppendArgument(llvm::StringRef("--child-platform-fd"));
  self_args.AppendArgument(llvm::to_string(shared_socket.GetSendableFD()));
#ifndef _WIN32
  launch_info.AppendDuplicateFileAction((int)shared_socket.GetSendableFD(),
                                        (int)shared_socket.GetSendableFD());
#endif
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

  if (g_server)
    launch_info.SetMonitorProcessCallback([](lldb::pid_t, int, int) {});
  else
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
  int long_option_index = 0;
  Status error;
  std::string listen_host_port;
  int ch;

  std::string log_file;
  StringRef
      log_channels; // e.g. "lldb process threads:gdb-remote default:linux all"

  shared_fd_t fd = SharedSocket::kInvalidFD;

  uint16_t gdbserver_port = 0;

  FileSpec socket_file;
  bool show_usage = false;
  int option_error = 0;

  std::string short_options(OptionParser::GetShortOptionString(g_long_options));

#if __GLIBC__
  optind = 0;
#else
  optreset = 1;
  optind = 1;
#endif

  while ((ch = getopt_long_only(argc, argv, short_options.c_str(),
                                g_long_options, &long_option_index)) != -1) {
    switch (ch) {
    case 0: // Any optional that auto set themselves will return 0
      break;

    case 'L':
      listen_host_port.append(optarg);
      break;

    case 'l': // Set Log File
      if (optarg && optarg[0])
        log_file.assign(optarg);
      break;

    case 'c': // Log Channels
      if (optarg && optarg[0])
        log_channels = StringRef(optarg);
      break;

    case 'f': // Socket file
      if (optarg && optarg[0])
        socket_file.SetFile(optarg, FileSpec::Style::native);
      break;

    case 'P':
    case 'm':
    case 'M': {
      uint16_t portnum;
      if (!llvm::to_integer(optarg, portnum)) {
        WithColor::error() << "invalid port number string " << optarg << "\n";
        option_error = 2;
        break;
      }
      // Note the condition gdbserver_port > HIGH_PORT is valid in case of using
      // --child-platform-fd. Check gdbserver_port later.
      if (ch == 'P')
        gdbserver_port = portnum;
      else if (gdbserver_port == 0)
        gdbserver_port = portnum;
    } break;

    case 2: {
      uint64_t _fd;
      if (!llvm::to_integer(optarg, _fd)) {
        WithColor::error() << "invalid fd " << optarg << "\n";
        option_error = 6;
      } else
        fd = (shared_fd_t)_fd;
    } break;

    case 'h': /* fall-through is intentional */
    case '?':
      show_usage = true;
      break;
    }
  }

  if (!LLDBServerUtilities::SetupLogging(log_file, log_channels, 0))
    return -1;

  // Print usage and exit if no listening port is specified.
  if (listen_host_port.empty() && fd == SharedSocket::kInvalidFD)
    show_usage = true;

  if (show_usage || option_error) {
    display_usage(progname, subcommand);
    exit(option_error);
  }

  // Skip any options we consumed with getopt_long_only.
  argc -= optind;
  argv += optind;
  lldb_private::Args inferior_arguments;
  inferior_arguments.SetArguments(argc, const_cast<const char **>(argv));

  Socket::SocketProtocol protocol = Socket::ProtocolUnixDomain;

  if (fd != SharedSocket::kInvalidFD) {
    // Child process will handle the connection and exit.
    if (gdbserver_port)
      protocol = Socket::ProtocolTcp;

    Log *log = GetLog(LLDBLog::Platform);

    NativeSocket sockfd;
    error = SharedSocket::GetNativeSocket(fd, sockfd);
    if (error.Fail()) {
      LLDB_LOGF(log, "lldb-platform child: %s", error.AsCString());
      return socket_error;
    }

    GDBRemoteCommunicationServerPlatform platform(protocol, gdbserver_port);
    Socket *socket;
    if (protocol == Socket::ProtocolTcp)
      socket = new TCPSocket(sockfd, /*should_close=*/true);
    else {
#if LLDB_ENABLE_POSIX
      socket = new DomainSocket(sockfd, /*should_close=*/true);
#else
      WithColor::error() << "lldb-platform child: Unix domain sockets are not "
                            "supported on this platform.";
      return socket_error;
#endif
    }
    platform.SetConnection(
        std::unique_ptr<Connection>(new ConnectionFileDescriptor(socket)));
    client_handle(platform, inferior_arguments);
    return 0;
  }

  if (gdbserver_port != 0 &&
      (gdbserver_port < LOW_PORT || gdbserver_port > HIGH_PORT)) {
    WithColor::error() << llvm::formatv("Port number {0} is not in the "
                                        "valid user port range of {1} - {2}\n",
                                        gdbserver_port, LOW_PORT, HIGH_PORT);
    return 1;
  }

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
      return 1;
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
                        log_channels, &main_loop,
                        &platform_handles](std::unique_ptr<Socket> sock_up) {
              printf("Connection established.\n");
              Status error = spawn_process(progname, sock_up.get(),
                                           gdbserver_port, inferior_arguments,
                                           log_file, log_channels, main_loop);
              if (error.Fail()) {
                Log *log = GetLog(LLDBLog::Platform);
                LLDB_LOGF(log, "spawn_process failed: %s", error.AsCString());
                WithColor::error()
                    << "spawn_process failed: " << error.AsCString() << "\n";
                if (!g_server)
                  main_loop.RequestTermination();
              }
              if (!g_server)
                platform_handles->clear();
            });
    if (!platform_handles) {
      printf("Failed to accept platform: %s\n",
             llvm::toString(platform_handles.takeError()).c_str());
      return socket_error;
    }

    llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>> gdb_handles =
        AcceptGdbConnectionsIfNeeded(protocol, gdb_sock, main_loop,
                                     gdbserver_port, inferior_arguments);
    if (!gdb_handles) {
      printf("Failed to accept gdb: %s\n",
             llvm::toString(gdb_handles.takeError()).c_str());
      return socket_error;
    }

    main_loop.Run();
  }

  fprintf(stderr, "lldb-server exiting...\n");

  return 0;
}
