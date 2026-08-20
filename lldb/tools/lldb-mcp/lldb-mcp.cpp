//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Multiplexer.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBProtocolServer.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "lldb/Host/Socket.h"
#include "lldb/Protocol/MCP/Client.h"
#include "lldb/Protocol/MCP/Server.h"
#include "lldb/Protocol/MCP/Transport.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/UriParser.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include <memory>

#if defined(_WIN32)
#include <fcntl.h>
#endif

using namespace llvm;
using namespace lldb;
using namespace lldb_protocol::mcp;

using lldb_private::File;
using lldb_private::FileSystem;
using lldb_private::HostInfo;
using lldb_private::MainLoop;
using lldb_private::MainLoopBase;
using lldb_private::NativeFile;
using lldb_private::Socket;

namespace {

inline void exitWithError(llvm::Error Err, StringRef Prefix = "") {
  handleAllErrors(std::move(Err), [&](ErrorInfoBase &Info) {
    WithColor::error(errs(), Prefix) << Info.message() << '\n';
  });
  std::exit(EXIT_FAILURE);
}

/// Returns a log callback that traces to stderr when LLDB_MCP_LOG is set in the
/// environment, since stdout is reserved for the MCP protocol.
lldb_protocol::mcp::LogCallback makeLogCallback() {
  if (!std::getenv("LLDB_MCP_LOG"))
    return {};
  return [](StringRef message) { errs() << message << '\n'; };
}

/// Connects to the MCP server described by \p info and returns the connected
/// socket.
Expected<IOObjectSP> connectToServer(const ServerInfo &info) {
  auto uri = lldb_private::URI::Parse(info.connection_uri);
  if (!uri)
    return createStringError("invalid connection_uri");

  std::optional<Socket::ProtocolModePair> protocol_and_mode =
      Socket::GetProtocolAndMode(uri->scheme);
  if (!protocol_and_mode)
    return createStringError("unknown protocol scheme");

  lldb_private::Status status;
  std::unique_ptr<Socket> sock =
      Socket::Create(protocol_and_mode->first, status);
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

/// Connects to the server described by \p info and adds it to \p multiplexer,
/// as the in-process local backend when \p local, otherwise as a remote one.
llvm::Error connectBackend(lldb_mcp::Multiplexer &multiplexer, MainLoop &loop,
                           const ServerInfo &info, bool local) {
  Expected<IOObjectSP> io = connectToServer(info);
  if (!io)
    return io.takeError();

  auto transport = std::make_unique<lldb_protocol::mcp::Transport>(
      loop, *io, *io, makeLogCallback());
  auto client = std::make_unique<lldb_protocol::mcp::Client>(
      std::move(transport), makeLogCallback());
  if (llvm::Error error = client->Run())
    return error;

  if (local)
    multiplexer.AddLocalBackend(info.pid, std::move(client));
  else
    multiplexer.AddBackend(info.pid, std::move(client));
  return llvm::Error::success();
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

  // Bring up the debug engine (through the public SB API) so lldb-mcp can host
  // debug sessions in its own process.
  SBDebugger::Initialize();
  auto terminate_debugger = llvm::scope_exit([]() { SBDebugger::Terminate(); });

  // lldb-mcp also uses the host and transport libraries directly (which link as
  // a separate copy from the one inside liblldb), so initialize the subsystems
  // it needs itself.
  FileSystem::Initialize();
  HostInfo::Initialize();
  if (llvm::Error error = Socket::Initialize())
    exitWithError(std::move(error));
  auto terminate_host = llvm::scope_exit([]() {
    Socket::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  });

  IOObjectSP input_sp = std::make_shared<NativeFile>(
      fileno(stdin), File::eOpenOptionReadOnly, NativeFile::Unowned);
  IOObjectSP output_sp = std::make_shared<NativeFile>(
      fileno(stdout), File::eOpenOptionWriteOnly, NativeFile::Unowned);

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

  // Start an in-process MCP server that hosts our own debug sessions, and drive
  // it as the local backend. This reuses the engine's MCP tools rather than
  // reimplementing them.
  const lldb::pid_t self_pid = llvm::sys::Process::getProcessId();
  SBError sb_error;
  SBProtocolServer protocol_server = SBProtocolServer::Create("MCP", sb_error);
  if (sb_error.Fail())
    exitWithError(createStringError(sb_error.GetCString()));
  if (SBError error = protocol_server.Start("listen://[localhost]:0");
      error.Fail())
    exitWithError(createStringError(error.GetCString()));
  const char *local_uri = protocol_server.GetConnectionURI();
  if (!local_uri)
    exitWithError(createStringError("in-process MCP server is not listening"));

  ServerInfo local_info{local_uri, self_pid};
  if (llvm::Error error =
          connectBackend(multiplexer, loop, local_info, /*local=*/true))
    exitWithError(std::move(error));

  // Connect to every other discovered LLDB MCP server as a remote backend
  // (skipping our own in-process one). A registry entry is written only after
  // its server is listening, so an entry that fails to connect is from a
  // crashed instance and gets pruned. A failed discovery is non-fatal.
  LogCallback log = makeLogCallback();
  if (Expected<std::vector<ServerInfo>> servers = ServerInfo::Load()) {
    for (const ServerInfo &info : *servers) {
      if (info.pid == self_pid)
        continue;
      if (llvm::Error error =
              connectBackend(multiplexer, loop, info, /*local=*/false)) {
        std::string reason = toString(std::move(error));
        if (log)
          log(formatv("pruning unreachable MCP server (pid {0}): {1}", info.pid,
                      reason)
                  .str());
        consumeError(ServerInfo::Remove(info.pid));
      }
    }
  } else {
    consumeError(servers.takeError());
  }

  multiplexer.SetDisconnectHandler([]() { loop.RequestTermination(); });
  if (llvm::Error error = multiplexer.Run())
    exitWithError(std::move(error));

  if (llvm::Error error = loop.Run().takeError())
    exitWithError(std::move(error));

  // The client disconnected. Fail any requests still in flight so abandoned
  // replies are satisfied, then stop the in-process server.
  multiplexer.Shutdown();
  protocol_server.Stop();
  return EXIT_SUCCESS;
}
