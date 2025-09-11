//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/Host/File.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "lldb/Host/Socket.h"
#include "lldb/Initialization/SystemInitializerCommon.h"
#include "lldb/Initialization/SystemLifetimeManager.h"
#include "lldb/Protocol/MCP/Server.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/UriParser.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include <cstdlib>
#include <memory>

#if defined(_WIN32)
#include <fcntl.h>
#endif

using namespace llvm;
using namespace lldb;
using namespace lldb_protocol::mcp;

using lldb_private::File;
using lldb_private::MainLoop;
using lldb_private::MainLoopBase;
using lldb_private::NativeFile;

namespace {

inline void exitWithError(llvm::Error Err, StringRef Prefix = "") {
  handleAllErrors(std::move(Err), [&](ErrorInfoBase &Info) {
    WithColor::error(errs(), Prefix) << Info.message() << '\n';
  });
  std::exit(EXIT_FAILURE);
}

constexpr size_t kForwardIOBufferSize = 1024;

void forwardIO(lldb_private::MainLoopBase &loop, lldb::IOObjectSP &from,
               lldb::IOObjectSP &to) {
  char buf[kForwardIOBufferSize];
  size_t num_bytes = sizeof(buf);

  if (llvm::Error err = from->Read(buf, num_bytes).takeError())
    exitWithError(std::move(err));

  // EOF reached.
  if (num_bytes == 0)
    return loop.RequestTermination();

  if (llvm::Error err = to->Write(buf, num_bytes).takeError())
    exitWithError(std::move(err));
}

void connectAndForwardIO(lldb_private::MainLoop &loop, ServerInfo &info,
                         IOObjectSP &input_sp, IOObjectSP &output_sp) {
  auto uri = lldb_private::URI::Parse(info.connection_uri);
  if (!uri)
    exitWithError(createStringError("invalid connection_uri"));

  std::optional<lldb_private::Socket::ProtocolModePair> protocol_and_mode =
      lldb_private::Socket::GetProtocolAndMode(uri->scheme);

  lldb_private::Status status;
  std::unique_ptr<lldb_private::Socket> sock =
      lldb_private::Socket::Create(protocol_and_mode->first, status);

  if (status.Fail())
    exitWithError(status.takeError());

  if (uri->port && !uri->hostname.empty())
    status = sock->Connect(
        llvm::formatv("[{0}]:{1}", uri->hostname, *uri->port).str());
  else
    status = sock->Connect(uri->path);
  if (status.Fail())
    exitWithError(status.takeError());

  IOObjectSP sock_sp = std::move(sock);
  auto input_handle = loop.RegisterReadObject(
      input_sp, std::bind(forwardIO, std::placeholders::_1, input_sp, sock_sp),
      status);
  if (status.Fail())
    exitWithError(status.takeError());

  auto socket_handle = loop.RegisterReadObject(
      sock_sp, std::bind(forwardIO, std::placeholders::_1, sock_sp, output_sp),
      status);
  if (status.Fail())
    exitWithError(status.takeError());

  status = loop.Run();
  if (status.Fail())
    exitWithError(status.takeError());
}

llvm::ManagedStatic<lldb_private::SystemLifetimeManager> g_debugger_lifetime;

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

  if (llvm::Error err = g_debugger_lifetime->Initialize(
          std::make_unique<lldb_private::SystemInitializerCommon>(nullptr)))
    exitWithError(std::move(err));

  auto cleanup = make_scope_exit([] { g_debugger_lifetime->Terminate(); });

  IOObjectSP input_sp = std::make_shared<NativeFile>(
      fileno(stdin), File::eOpenOptionReadOnly, NativeFile::Unowned);

  IOObjectSP output_sp = std::make_shared<NativeFile>(
      fileno(stdout), File::eOpenOptionWriteOnly, NativeFile::Unowned);

  static MainLoop loop;

  sys::SetInterruptFunction([]() {
    loop.AddPendingCallback(
        [](MainLoopBase &loop) { loop.RequestTermination(); });
  });

  auto existing_servers = ServerInfo::Load();

  if (!existing_servers)
    exitWithError(existing_servers.takeError());

  // FIXME: Launch `lldb -o 'protocol start MCP'`.
  if (existing_servers->empty())
    exitWithError(createStringError("No MCP servers running"));

  // FIXME: Support selecting a specific server.
  if (existing_servers->size() != 1)
    exitWithError(
        createStringError("To many MCP servers running, picking a specific "
                          "one is not yet implemented."));

  ServerInfo &info = existing_servers->front();
  connectAndForwardIO(loop, info, input_sp, output_sp);

  return EXIT_SUCCESS;
}
