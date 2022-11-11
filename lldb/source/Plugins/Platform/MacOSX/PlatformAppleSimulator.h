//===-- PlatformAppleSimulator.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMAPPLESIMULATOR_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMAPPLESIMULATOR_H

#include "Plugins/Platform/MacOSX/PlatformDarwin.h"
#include "Plugins/Platform/MacOSX/objcxx/PlatformiOSSimulatorCoreSimulatorSupport.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/XcodeSDK.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"

#include <mutex>
#include <vector>

namespace lldb_private {
class ArchSpec;
class Args;
class Debugger;
class FileSpecList;
class ModuleSpec;
class Process;
class ProcessLaunchInfo;
class Stream;
class Target;
class UUID;

class PlatformAppleSimulator : public PlatformDarwin {
public:
  // Class Functions
  static void Initialize();

  static void Terminate();

  // Class Methods
  PlatformAppleSimulator(
      const char *class_name, const char *description, ConstString plugin_name,
      llvm::Triple::OSType preferred_os,
      llvm::SmallVector<llvm::StringRef, 4> supported_triples,
      llvm::StringRef sdk, XcodeSDK::Type sdk_type,
      CoreSimulatorSupport::DeviceType::ProductFamilyID kind);

  static lldb::PlatformSP
  CreateInstance(const char *class_name, const char *description,
                 ConstString plugin_name,
                 llvm::SmallVector<llvm::Triple::ArchType, 4> supported_arch,
                 llvm::Triple::OSType preferred_os,
                 llvm::SmallVector<llvm::Triple::OSType, 4> supported_os,
                 llvm::SmallVector<llvm::StringRef, 4> supported_triples,
                 std::string sdk_name_preferred, std::string sdk_name_secondary,
                 XcodeSDK::Type sdk_type,
                 CoreSimulatorSupport::DeviceType::ProductFamilyID kind,
                 bool force, const ArchSpec *arch);

  ~PlatformAppleSimulator() override;

  llvm::StringRef GetPluginName() override {
    return m_plugin_name.GetStringRef();
  }
  llvm::StringRef GetDescription() override { return m_description; }

  Status LaunchProcess(ProcessLaunchInfo &launch_info) override;

  void GetStatus(Stream &strm) override;

  Status ConnectRemote(Args &args) override;

  Status DisconnectRemote() override;

  lldb::ProcessSP DebugProcess(ProcessLaunchInfo &launch_info,
                               Debugger &debugger, Target &target,
                               Status &error) override;

  std::vector<ArchSpec>
  GetSupportedArchitectures(const ArchSpec &process_host_arch) override;

  Status
  ResolveExecutable(const ModuleSpec &module_spec, lldb::ModuleSP &module_sp,
                    const FileSpecList *module_search_paths_ptr) override;

  Status GetSharedModule(const ModuleSpec &module_spec, Process *process,
                         lldb::ModuleSP &module_sp,
                         const FileSpecList *module_search_paths_ptr,
                         llvm::SmallVectorImpl<lldb::ModuleSP> *old_modules,
                         bool *did_create_ptr) override;

  uint32_t FindProcesses(const ProcessInstanceInfoMatch &match_info,
                         ProcessInstanceInfoList &process_infos) override;

  void
  AddClangModuleCompilationOptions(Target *target,
                                   std::vector<std::string> &options) override {
    return PlatformDarwin::AddClangModuleCompilationOptionsForSDKType(
        target, options, m_sdk_type);
  }

protected:
  const char *m_class_name;
  const char *m_description;
  ConstString m_plugin_name;
  std::mutex m_core_sim_path_mutex;
  llvm::Optional<FileSpec> m_core_simulator_framework_path;
  llvm::Optional<CoreSimulatorSupport::Device> m_device;
  CoreSimulatorSupport::DeviceType::ProductFamilyID m_kind;

  FileSpec GetCoreSimulatorPath();

  llvm::Triple::OSType m_os_type = llvm::Triple::UnknownOS;
  llvm::SmallVector<llvm::StringRef, 4> m_supported_triples = {};
  llvm::StringRef m_sdk;
  XcodeSDK::Type m_sdk_type;

  void LoadCoreSimulator();

#if defined(__APPLE__)
  CoreSimulatorSupport::Device GetSimulatorDevice();
#endif

private:
  PlatformAppleSimulator(const PlatformAppleSimulator &) = delete;
  const PlatformAppleSimulator &
  operator=(const PlatformAppleSimulator &) = delete;
  Status

  GetSymbolFile(const FileSpec &platform_file, const UUID *uuid_ptr,
                FileSpec &local_file);
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMAPPLESIMULATOR_H
