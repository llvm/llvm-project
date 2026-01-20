//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/WebAssembly/PlatformWasm.h"
#include "Plugins/Platform/WebAssembly/PlatformWasmRemoteGDBServer.h"
#include "Plugins/Process/wasm/ProcessWasm.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Listener.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorExtras.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(PlatformWasm)

namespace {
#define LLDB_PROPERTIES_platformwasm
#include "PlatformWasmProperties.inc"

enum {
#define LLDB_PROPERTIES_platformwasm
#include "PlatformWasmPropertiesEnum.inc"
};

class PluginProperties : public Properties {
public:
  PluginProperties() {
    m_collection_sp = std::make_shared<OptionValueProperties>(
        PlatformWasm::GetPluginNameStatic());
    m_collection_sp->Initialize(g_platformwasm_properties);
  }

  FileSpec GetRuntimePath() const {
    return GetPropertyAtIndexAs<FileSpec>(ePropertyRuntimePath, {});
  }

  Args GetRuntimeArgs() const {
    Args result;
    m_collection_sp->GetPropertyAtIndexAsArgs(ePropertyRuntimeArgs, result);
    return result;
  }

  llvm::StringRef GetPortArg() const {
    return GetPropertyAtIndexAs<llvm::StringRef>(ePropertyPortArg, {});
  }
};

} // namespace

static PluginProperties &GetGlobalProperties() {
  static PluginProperties g_settings;
  return g_settings;
}

llvm::StringRef PlatformWasm::GetPluginDescriptionStatic() {
  return "Platform for debugging Wasm";
}

void PlatformWasm::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), GetPluginDescriptionStatic(),
      PlatformWasm::CreateInstance, PlatformWasm::DebuggerInitialize);
}

void PlatformWasm::Terminate() {
  PluginManager::UnregisterPlugin(PlatformWasm::CreateInstance);
}

void PlatformWasm::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForPlatformPlugin(debugger,
                                                  GetPluginNameStatic())) {
    PluginManager::CreateSettingForPlatformPlugin(
        debugger, GetGlobalProperties().GetValueProperties(),
        "Properties for the wasm platform plugin.",
        /*is_global_property=*/true);
  }
}

PlatformSP PlatformWasm::CreateInstance(bool force, const ArchSpec *arch) {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOG(log, "force = {0}, arch = ({1}, {2})", force,
           arch ? arch->GetArchitectureName() : "<null>",
           arch ? arch->GetTriple().getTriple() : "<null>");

  bool create = force;
  if (!create && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getArch()) {
    case llvm::Triple::wasm32:
    case llvm::Triple::wasm64:
      create = true;
      break;
    default:
      break;
    }
  }

  LLDB_LOG(log, "create = {0}", create);
  return create ? PlatformSP(new PlatformWasm()) : PlatformSP();
}

std::vector<ArchSpec>
PlatformWasm::GetSupportedArchitectures(const ArchSpec &process_host_arch) {
  return {ArchSpec("wasm32-unknown-unknown-wasm"),
          ArchSpec("wasm64-unknown-unknown-wasm")};
}

static auto get_arg_range(const Args &args) {
  return llvm::make_range(args.GetArgumentArrayRef().begin(),
                          args.GetArgumentArrayRef().end());
}

lldb::ProcessSP PlatformWasm::Attach(ProcessAttachInfo &attach_info,
                                     Debugger &debugger, Target *target,
                                     Status &status) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->Attach(attach_info, debugger, target, status);

  status = Status::FromErrorString(
      "attaching is only supported when connected to a remote Wasm platform");
  return nullptr;
}

lldb::ProcessSP PlatformWasm::DebugProcess(ProcessLaunchInfo &launch_info,
                                           Debugger &debugger, Target &target,
                                           Status &error) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->DebugProcess(launch_info, debugger, target,
                                              error);

  Log *log = GetLog(LLDBLog::Platform);

  const PluginProperties &properties = GetGlobalProperties();

  FileSpec runtime = properties.GetRuntimePath();
  FileSystem::Instance().ResolveExecutableLocation(runtime);

  if (!FileSystem::Instance().Exists(runtime)) {
    error = Status::FromErrorStringWithFormatv(
        "WebAssembly runtime does not exist: {0}", runtime.GetPath());
    return nullptr;
  }

  uint16_t port = 0;
  {
    // Get the next available port by binding a socket to port 0.
    TCPSocket listen_socket(true);
    error = listen_socket.Listen("localhost:0", /*backlog=*/5);
    if (error.Fail())
      return nullptr;
    port = listen_socket.GetLocalPortNumber();
  }

  if (error.Fail())
    return nullptr;

  Args args({runtime.GetPath(),
             llvm::formatv("{0}{1}", properties.GetPortArg(), port).str()});
  args.AppendArguments(properties.GetRuntimeArgs());
  args.AppendArguments(launch_info.GetArguments());

  launch_info.SetArguments(args, true);
  launch_info.SetLaunchInSeparateProcessGroup(true);
  launch_info.GetFlags().Clear(eLaunchFlagDebug);

  auto exit_code = std::make_shared<std::optional<int>>();
  launch_info.SetMonitorProcessCallback(
      [=](lldb::pid_t pid, int signal, int status) {
        LLDB_LOG(
            log,
            "WebAssembly runtime exited: pid = {0}, signal = {1}, status = {2}",
            pid, signal, status);
        exit_code->emplace(status);
      });

  // This is automatically done for host platform in
  // Target::FinalizeFileActions, but we're not a host platform.
  llvm::Error Err = launch_info.SetUpPtyRedirection();
  LLDB_LOG_ERROR(log, std::move(Err), "SetUpPtyRedirection failed: {0}");

  LLDB_LOG(log, "{0}", get_arg_range(launch_info.GetArguments()));
  error = Host::LaunchProcess(launch_info);
  if (error.Fail())
    return nullptr;

  ProcessSP process_sp = target.CreateProcess(
      launch_info.GetListener(), wasm::ProcessWasm::GetPluginNameStatic(),
      nullptr, true);
  if (!process_sp) {
    error = Status::FromErrorString("failed to create WebAssembly process");
    return nullptr;
  }

  process_sp->HijackProcessEvents(launch_info.GetHijackListener());

  error = process_sp->ConnectRemote(
      llvm::formatv("connect://localhost:{0}", port).str());
  if (error.Fail()) {
    // If we know the runtime has exited, that's a better error message than
    // failing to connect.
    if (*exit_code)
      error = Status::FromError(llvm::joinErrors(
          llvm::createStringErrorV(
              "WebAssembly runtime exited with exit code {0}", **exit_code),
          error.takeError()));

    return nullptr;
  }
#ifndef _WIN32
  if (launch_info.GetPTY().GetPrimaryFileDescriptor() !=
      PseudoTerminal::invalid_fd)
    process_sp->SetSTDIOFileDescriptor(
        launch_info.GetPTY().ReleasePrimaryFileDescriptor());
#endif
  return process_sp;
}

Status PlatformWasm::ConnectRemote(Args &args) {
  if (IsHost())
    return Status::FromErrorString(
        "can't connect to the host platform, always connected");

  if (!m_remote_platform_sp)
    m_remote_platform_sp = PlatformSP(new PlatformWasmRemoteGDBServer());

  return m_remote_platform_sp->ConnectRemote(args);
}
