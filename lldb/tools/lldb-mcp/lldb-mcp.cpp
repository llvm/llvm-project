//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Host/Socket.h"
#include "lldb/Initialization/SystemInitializerCommon.h"
#include "lldb/Initialization/SystemLifetimeManager.h"
#include "lldb/Protocol/MCP/Server.h"
#include "lldb/Utility/FileSpec.h"
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

constexpr size_t kForwardIOBufferSize = 1024;

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

Expected<ServerInfo> loadOrStart(
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

    // FIXME: Support selecting / multiplexing a specific lldb instance.
    if (servers->size() > 1)
      return createStringError("too many MCP servers running, picking a "
                               "specific one is not yet implemented");

    return servers->front();
  }

  return createStringError("timed out waiting for MCP server to start");
}

void forwardIO(MainLoopBase &loop, IOObjectSP &from, IOObjectSP &to) {
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

llvm::Error connectAndForwardIO(lldb_private::MainLoop &loop, ServerInfo &info,
                                IOObjectSP &input_sp, IOObjectSP &output_sp) {
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

  IOObjectSP sock_sp = std::move(sock);
  auto input_handle = loop.RegisterReadObject(
      input_sp, std::bind(forwardIO, std::placeholders::_1, input_sp, sock_sp),
      status);
  if (status.Fail())
    return status.takeError();

  auto socket_handle = loop.RegisterReadObject(
      sock_sp, std::bind(forwardIO, std::placeholders::_1, sock_sp, output_sp),
      status);
  if (status.Fail())
    return status.takeError();

  return loop.Run().takeError();
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

  Expected<ServerInfo> server_info = loadOrStart();
  if (!server_info)
    exitWithError(server_info.takeError());

  static MainLoop loop;
  sys::SetInterruptFunction([]() {
    loop.AddPendingCallback(
        [](MainLoopBase &loop) { loop.RequestTermination(); });
  });

  if (llvm::Error error =
          connectAndForwardIO(loop, *server_info, input_sp, output_sp))
    exitWithError(std::move(error));

  return EXIT_SUCCESS;
}
