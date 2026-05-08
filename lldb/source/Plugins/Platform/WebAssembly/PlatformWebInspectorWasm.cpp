//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformWebInspectorWasm.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorExtras.h"

#include <chrono>
#include <csignal>
#include <thread>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(PlatformWebInspectorWasm)

static constexpr llvm::StringLiteral kServerBinary =
    "/System/Cryptexes/App/usr/libexec/webinspector-wasm-lldb-platform";
static constexpr uint8_t kConnectAttempts = 5;
static constexpr auto kConnectDelay = std::chrono::milliseconds(100);

llvm::StringRef PlatformWebInspectorWasm::GetPluginDescriptionStatic() {
  return "Platform for debugging Wasm via WebInspector";
}

void PlatformWebInspectorWasm::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(),
                                PlatformWebInspectorWasm::CreateInstance);
}

void PlatformWebInspectorWasm::Terminate() {
  PluginManager::UnregisterPlugin(PlatformWebInspectorWasm::CreateInstance);
}

PlatformSP PlatformWebInspectorWasm::CreateInstance(bool force,
                                                    const ArchSpec *arch) {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOG(log, "force = {0}, arch = ({1}, {2})", force,
           arch ? arch->GetArchitectureName() : "<null>",
           arch ? arch->GetTriple().getTriple() : "<null>");
  return force ? PlatformSP(new PlatformWebInspectorWasm()) : PlatformSP();
}

PlatformWebInspectorWasm::~PlatformWebInspectorWasm() {
  if (m_server_pid != LLDB_INVALID_PROCESS_ID)
    Host::Kill(m_server_pid, SIGTERM);
}

llvm::Error PlatformWebInspectorWasm::LaunchPlatformServer() {
  Log *log = GetLog(LLDBLog::Platform);

  if (!FileSystem::Instance().Exists(FileSpec(kServerBinary)))
    return llvm::createStringErrorV("platform binary not found: {0}",
                                    kServerBinary);

  // Find two free TCP ports.
  llvm::Expected<uint16_t> expected_platform_port = FindFreeTCPPort();
  if (!expected_platform_port)
    return expected_platform_port.takeError();
  uint16_t platform_port = *expected_platform_port;

  llvm::Expected<uint16_t> expected_debugserver_port = FindFreeTCPPort();
  if (!expected_debugserver_port)
    return expected_debugserver_port.takeError();
  uint16_t debugserver_port = *expected_debugserver_port;

  ProcessLaunchInfo launch_info;
  launch_info.SetExecutableFile(FileSpec(kServerBinary),
                                /*add_exe_file_as_first_arg=*/true);
  Args args;
  args.AppendArgument(kServerBinary);
  args.AppendArgument("--platform");
  args.AppendArgument(llvm::utostr(platform_port));
  args.AppendArgument("--debugserver");
  args.AppendArgument(llvm::utostr(debugserver_port));
  launch_info.SetArguments(args, /*first_arg_is_executable=*/true);
  launch_info.SetLaunchInSeparateProcessGroup(true);
  launch_info.GetFlags().Clear(eLaunchFlagDebug);

  launch_info.SetMonitorProcessCallback(
      [log](lldb::pid_t pid, int signal, int status) {
        LLDB_LOG(log,
                 "Platform exited: pid = {0}, signal = "
                 "{1}, status = {2}",
                 pid, signal, status);
      });

  LLDB_LOG(log, "{0}", GetArgRange(launch_info.GetArguments()));

  Status status = Host::LaunchProcess(launch_info);
  if (status.Fail())
    return status.takeError();

  m_server_pid = launch_info.GetProcessID();
  LLDB_LOG(log, "Platform launched: pid = {0}", m_server_pid);

  Args connect_args;
  connect_args.AppendArgument(
      llvm::formatv("connect://localhost:{0}", platform_port).str());

  // The platform may need some time to bind a socket to the requested port.
  for (uint8_t attempt = 0; attempt < kConnectAttempts; attempt++) {
    status = PlatformWasm::ConnectRemote(connect_args);
    if (status.Success())
      return llvm::Error::success();

    LLDB_LOG(
        log,
        "[{0}/{1}] platform not yet listening on port {2}: trying again in {3}",
        attempt, kConnectAttempts, platform_port, kConnectDelay);
    std::this_thread::sleep_for(kConnectDelay);
  }
  return status.takeError();
}

llvm::Error PlatformWebInspectorWasm::EnsureConnected() {
  if (m_remote_platform_sp)
    return llvm::Error::success();
  return LaunchPlatformServer();
}

Status PlatformWebInspectorWasm::ConnectRemote(Args &args) {
  if (args.GetArgumentCount() == 0)
    return Status::FromError(LaunchPlatformServer());
  return PlatformWasm::ConnectRemote(args);
}

ProcessSP PlatformWebInspectorWasm::Attach(ProcessAttachInfo &attach_info,
                                           Debugger &debugger, Target *target,
                                           Status &status) {
  status = Status::FromError(EnsureConnected());
  if (status.Fail())
    return nullptr;
  return PlatformWasm::Attach(attach_info, debugger, target, status);
}

uint32_t PlatformWebInspectorWasm::FindProcesses(
    const ProcessInstanceInfoMatch &match_info,
    ProcessInstanceInfoList &proc_infos) {
  if (llvm::Error err = EnsureConnected()) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Platform), std::move(err),
                   "EnsureConnected failed: {0}");
    return 0;
  }
  return PlatformWasm::FindProcesses(match_info, proc_infos);
}

bool PlatformWebInspectorWasm::GetProcessInfo(lldb::pid_t pid,
                                              ProcessInstanceInfo &proc_info) {
  if (llvm::Error err = EnsureConnected()) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Platform), std::move(err),
                   "EnsureConnected failed: {0}");
    return false;
  }
  return PlatformWasm::GetProcessInfo(pid, proc_info);
}

lldb::ProcessSP
PlatformWebInspectorWasm::DebugProcess(ProcessLaunchInfo &launch_info,
                                       Debugger &debugger, Target &target,
                                       Status &error) {
  error = Status::FromErrorStringWithFormatv("{0} does not support launching",
                                             GetPluginNameStatic());
  return nullptr;
}
