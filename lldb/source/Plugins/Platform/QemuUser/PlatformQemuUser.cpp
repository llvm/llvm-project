//===-- PlatformQemuUser.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/QemuUser/PlatformQemuUser.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Listener.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(PlatformQemuUser)

#define LLDB_PROPERTIES_platformqemuuser
#include "PlatformQemuUserProperties.inc"

enum {
#define LLDB_PROPERTIES_platformqemuuser
#include "PlatformQemuUserPropertiesEnum.inc"
};

class PluginProperties : public Properties {
public:
  PluginProperties() {
    m_collection_sp = std::make_shared<OptionValueProperties>(
        ConstString(PlatformQemuUser::GetPluginNameStatic()));
    m_collection_sp->Initialize(g_platformqemuuser_properties);
  }

  llvm::StringRef GetArchitecture() {
    return m_collection_sp->GetPropertyAtIndexAsString(
        nullptr, ePropertyArchitecture, "");
  }

  FileSpec GetEmulatorPath() {
    return m_collection_sp->GetPropertyAtIndexAsFileSpec(nullptr,
                                                         ePropertyEmulatorPath);
  }
};

static PluginProperties &GetGlobalProperties() {
  static PluginProperties g_settings;
  return g_settings;
}

llvm::StringRef PlatformQemuUser::GetPluginDescriptionStatic() {
  return "Platform for debugging binaries under user mode qemu";
}

void PlatformQemuUser::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), GetPluginDescriptionStatic(),
      PlatformQemuUser::CreateInstance, PlatformQemuUser::DebuggerInitialize);
}

void PlatformQemuUser::Terminate() {
  PluginManager::UnregisterPlugin(PlatformQemuUser::CreateInstance);
}

void PlatformQemuUser::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForPlatformPlugin(
          debugger, ConstString(GetPluginNameStatic()))) {
    PluginManager::CreateSettingForPlatformPlugin(
        debugger, GetGlobalProperties().GetValueProperties(),
        ConstString("Properties for the qemu-user platform plugin."),
        /*is_global_property=*/true);
  }
}

PlatformSP PlatformQemuUser::CreateInstance(bool force, const ArchSpec *arch) {
  if (force)
    return PlatformSP(new PlatformQemuUser());
  return nullptr;
}

std::vector<ArchSpec> PlatformQemuUser::GetSupportedArchitectures() {
  llvm::Triple triple = HostInfo::GetArchitecture().GetTriple();
  triple.setEnvironment(llvm::Triple::UnknownEnvironment);
  triple.setArchName(GetGlobalProperties().GetArchitecture());
  if (triple.getArch() != llvm::Triple::UnknownArch)
    return {ArchSpec(triple)};
  return {};
}

static auto get_arg_range(const Args &args) {
  return llvm::make_range(args.GetArgumentArrayRef().begin(),
                          args.GetArgumentArrayRef().end());
}

lldb::ProcessSP PlatformQemuUser::DebugProcess(ProcessLaunchInfo &launch_info,
                                               Debugger &debugger,
                                               Target &target, Status &error) {
  Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);

  std::string qemu = GetGlobalProperties().GetEmulatorPath().GetPath();

  llvm::SmallString<0> socket_model, socket_path;
  HostInfo::GetProcessTempDir().GetPath(socket_model);
  llvm::sys::path::append(socket_model, "qemu-%%%%%%%%.socket");
  do {
    llvm::sys::fs::createUniquePath(socket_model, socket_path, false);
  } while (FileSystem::Instance().Exists(socket_path));

  Args args(
      {qemu, "-g", socket_path, launch_info.GetExecutableFile().GetPath()});
  for (size_t i = 1; i < launch_info.GetArguments().size(); ++i)
    args.AppendArgument(launch_info.GetArguments()[i].ref());

  LLDB_LOG(log, "{0} -> {1}", get_arg_range(launch_info.GetArguments()),
           get_arg_range(args));

  launch_info.SetArguments(args, true);
  launch_info.SetLaunchInSeparateProcessGroup(true);
  launch_info.GetFlags().Clear(eLaunchFlagDebug);
  launch_info.SetMonitorProcessCallback(ProcessLaunchInfo::NoOpMonitorCallback,
                                        false);

  error = Host::LaunchProcess(launch_info);
  if (error.Fail())
    return nullptr;

  ProcessSP process_sp = target.CreateProcess(
      launch_info.GetListener(),
      process_gdb_remote::ProcessGDBRemote::GetPluginNameStatic(), nullptr,
      true);
  ListenerSP listener_sp =
      Listener::MakeListener("lldb.platform_qemu_user.debugprocess");
  launch_info.SetHijackListener(listener_sp);
  Process::ProcessEventHijacker hijacker(*process_sp, listener_sp);

  error = process_sp->ConnectRemote(("unix-connect://" + socket_path).str());
  if (error.Fail())
    return nullptr;

  process_sp->WaitForProcessToStop(llvm::None, nullptr, false, listener_sp);
  return process_sp;
}
