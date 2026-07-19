//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Multiplexer.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Host/Socket.h"
#include "lldb/Protocol/MCP/Client.h"
#include "lldb/Protocol/MCP/Server.h"
#include "lldb/Protocol/MCP/Transport.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/UriParser.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>

#if defined(_WIN32)
#include <fcntl.h>
#endif

using namespace llvm;
using namespace lldb;
using namespace lldb_protocol::mcp;

using lldb_private::Environment;
using lldb_private::File;
using lldb_private::FileSpec;
using lldb_private::FileSystem;
using lldb_private::Host;
using lldb_private::MainLoop;
using lldb_private::MainLoopBase;
using lldb_private::NativeFile;

namespace {

#if defined(_WIN32)
constexpr StringLiteral kDriverName = "lldb.exe";
#else
constexpr StringLiteral kDriverName = "lldb";
#endif

inline void exitWithError(llvm::Error Err, StringRef Prefix = "") {
  handleAllErrors(std::move(Err), [&](ErrorInfoBase &Info) {
    WithColor::error(errs(), Prefix) << Info.message() << '\n';
  });
  std::exit(EXIT_FAILURE);
}

FileSpec driverPath() {
  Environment host_env = Host::GetEnvironment();

  // Check if an override for which lldb we're using exists, otherwise look next
  // to the current binary.
  std::string lldb_exe_path = host_env.lookup("LLDB_EXE_PATH");
  auto &fs = FileSystem::Instance();
  if (fs.Exists(lldb_exe_path))
    return FileSpec(lldb_exe_path);

  FileSpec lldb_exec_spec = lldb_private::HostInfo::GetProgramFileSpec();
  lldb_exec_spec.SetFilename(kDriverName);
  return lldb_exec_spec;
}

llvm::Error launch() {
  FileSpec lldb_exec = driverPath();
  lldb_private::ProcessLaunchInfo info;
  info.SetMonitorProcessCallback(
      &lldb_private::ProcessLaunchInfo::NoOpMonitorCallback);
  info.SetExecutableFile(lldb_exec,
                         /*add_exe_file_as_first_arg=*/true);
  info.GetArguments().AppendArgument("-O");
  info.GetArguments().AppendArgument("protocol start MCP");
  return Host::LaunchProcess(info).takeError();
}

Expected<std::vector<ServerInfo>> loadOrStart(
    // FIXME: This should become a CLI arg.
    lldb_private::Timeout<std::micro> timeout = std::chrono::seconds(30)) {
  using namespace std::chrono;
  bool started = false;

  const auto deadline = steady_clock::now() + *timeout;
  while (steady_clock::now() < deadline) {
    Expected<std::vector<ServerInfo>> servers = ServerInfo::Load();
    if (!servers)
      return servers.takeError();

    if (servers->empty()) {
      if (!started) {
        started = true;
        if (llvm::Error err = launch())
          return std::move(err);
      }

      // FIXME: Can we use MainLoop to watch the directory?
      std::this_thread::sleep_for(microseconds(250));
      continue;
    }

    return std::move(*servers);
  }

  return createStringError("timed out waiting for MCP server to start");
}

/// Connects to the MCP server described by \p info and returns the connected
/// socket.
Expected<IOObjectSP> connectToServer(const ServerInfo &info) {
  auto uri = lldb_private::URI::Parse(info.connection_uri);
  if (!uri)
    return createStringError("invalid connection_uri");

  std::optional<lldb_private::Socket::ProtocolModePair> protocol_and_mode =
      lldb_private::Socket::GetProtocolAndMode(uri->scheme);
  if (!protocol_and_mode)
    return createStringError("unknown protocol scheme");

  lldb_private::Status status;
  std::unique_ptr<lldb_private::Socket> sock =
      lldb_private::Socket::Create(protocol_and_mode->first, status);
  if (status.Fail())
    return status.takeError();

  if (uri->port && !uri->hostname.empty())
    status = sock->Connect(
        llvm::formatv("[{0}]:{1}", uri->hostname, *uri->port).str());
  else
    status = sock->Connect(uri->path);
  if (status.Fail())
    return status.takeError();

  return IOObjectSP(std::move(sock));
}

/// Returns a log callback that traces to stderr when LLDB_MCP_LOG is set in the
/// environment, since stdout is reserved for the MCP protocol.
lldb_protocol::mcp::LogCallback makeLogCallback() {
  if (!std::getenv("LLDB_MCP_LOG"))
    return {};
  return [](StringRef message) { errs() << message << '\n'; };
}

} // namespace

int main(int argc, char *argv[]) {
  llvm::InitLLVM IL(argc, argv, /*InstallPipeSignalExitHandler=*/false);
#if !defined(__APPLE__)
  llvm::setBugReportMsg("PLEASE submit a bug report to " LLDB_BUG_REPORT_URL
                        " and include the crash backtrace.\n");
#else
  llvm::setBugReportMsg("PLEASE submit a bug report to " LLDB_BUG_REPORT_URL
                        " and include the crash report from "
                        "~/Library/Logs/DiagnosticReports/.\n");
#endif

#if defined(_WIN32)
  // Windows opens stdout and stdin in text mode which converts \n to 13,10
  // while the value is just 10 on Darwin/Linux. Setting the file mode to
  // binary fixes this.
  int result = _setmode(fileno(stdout), _O_BINARY);
  assert(result);
  result = _setmode(fileno(stdin), _O_BINARY);
  UNUSED_IF_ASSERT_DISABLED(result);
  assert(result);
#endif

  lldb_private::FileSystem::Initialize();
  lldb_private::HostInfo::Initialize();
  if (llvm::Error error = lldb_private::Socket::Initialize())
    exitWithError(std::move(error));

  llvm::scope_exit cleanup([] {
    lldb_private::Socket::Terminate();
    lldb_private::HostInfo::Terminate();
    lldb_private::FileSystem::Terminate();
  });

  IOObjectSP input_sp = std::make_shared<NativeFile>(
      fileno(stdin), File::eOpenOptionReadOnly, NativeFile::Unowned);

  IOObjectSP output_sp = std::make_shared<NativeFile>(
      fileno(stdout), File::eOpenOptionWriteOnly, NativeFile::Unowned);

  Expected<std::vector<ServerInfo>> servers = loadOrStart();
  if (!servers)
    exitWithError(servers.takeError());

  static MainLoop loop;
  sys::SetInterruptFunction([]() {
    loop.AddPendingCallback(
        [](MainLoopBase &loop) { loop.RequestTermination(); });
  });

  // Present a unified MCP server to the client over stdio.
  auto client_transport = std::make_unique<lldb_protocol::mcp::Transport>(
      loop, input_sp, output_sp, makeLogCallback());
  lldb_mcp::Multiplexer multiplexer(std::move(client_transport),
                                    makeLogCallback());

  // A stale registry entry left by a crashed instance fails to connect; skip it
  // rather than aborting.
  LogCallback log = makeLogCallback();
  size_t connected = 0;
  for (const ServerInfo &info : *servers) {
    Expected<IOObjectSP> backend_io = connectToServer(info);
    if (!backend_io) {
      std::string reason = toString(backend_io.takeError());
      if (log)
        log(formatv("skipping unreachable MCP server (pid {0}): {1}", info.pid,
                    reason)
                .str());
      continue;
    }

    auto backend_transport = std::make_unique<lldb_protocol::mcp::Transport>(
        loop, *backend_io, *backend_io, makeLogCallback());
    auto backend = std::make_unique<lldb_protocol::mcp::Client>(
        std::move(backend_transport), makeLogCallback());
    if (llvm::Error error = backend->Run()) {
      std::string reason = toString(std::move(error));
      if (log)
        log(formatv("skipping MCP server (pid {0}): {1}", info.pid, reason)
                .str());
      continue;
    }

    multiplexer.AddBackend(info.pid, std::move(backend));
    ++connected;
  }

  if (connected == 0)
    exitWithError(createStringError("failed to connect to any MCP server"));

  multiplexer.SetDisconnectHandler([]() { loop.RequestTermination(); });
  if (llvm::Error error = multiplexer.Run())
    exitWithError(std::move(error));

  if (llvm::Error error = loop.Run().takeError())
    exitWithError(std::move(error));

  // The client is gone; fail any requests still in flight to a backend so their
  // replies are satisfied rather than destroyed unanswered during teardown.
  multiplexer.Shutdown();
  return EXIT_SUCCESS;
}
