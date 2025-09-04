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
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Server.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"

#if defined(_WIN32)
#include <fcntl.h>
#endif

using namespace lldb_protocol::mcp;

using lldb_private::File;
using lldb_private::MainLoop;
using lldb_private::MainLoopBase;
using lldb_private::NativeFile;

static constexpr llvm::StringLiteral kName = "lldb-mcp";
static constexpr llvm::StringLiteral kVersion = "0.1.0";

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

  lldb::IOObjectSP input = std::make_shared<NativeFile>(
      fileno(stdin), File::eOpenOptionReadOnly, NativeFile::Unowned);

  lldb::IOObjectSP output = std::make_shared<NativeFile>(
      fileno(stdout), File::eOpenOptionWriteOnly, NativeFile::Unowned);

  constexpr llvm::StringLiteral client_name = "stdio";
  static MainLoop loop;

  llvm::sys::SetInterruptFunction([]() {
    loop.AddPendingCallback(
        [](MainLoopBase &loop) { loop.RequestTermination(); });
  });

  auto transport_up = std::make_unique<lldb_protocol::mcp::Transport>(
      input, output, [&](llvm::StringRef message) {
        llvm::errs() << formatv("{0}: {1}", client_name, message) << '\n';
      });

  auto instance_up = std::make_unique<lldb_protocol::mcp::Server>(
      std::string(kName), std::string(kVersion), std::move(transport_up), loop);

  if (llvm::Error error = instance_up->Run()) {
    llvm::logAllUnhandledErrors(std::move(error), llvm::WithColor::error(),
                                "MCP error: ");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
