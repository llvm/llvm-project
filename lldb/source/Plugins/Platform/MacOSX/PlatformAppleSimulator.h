//===-- PlatformAppleSimulator.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMAPPLESIMULATOR_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMAPPLESIMULATOR_H

#include <mutex>

#include "Plugins/Platform/MacOSX/PlatformDarwin.h"
#include "Plugins/Platform/MacOSX/objcxx/PlatformiOSSimulatorCoreSimulatorSupport.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/FileSpec.h"

#include "llvm/ADT/Optional.h"

class PlatformAppleSimulator : public PlatformDarwin {
public:
  // Class Functions
  static void Initialize();

  static void Terminate();

  // Class Methods
  PlatformAppleSimulator(
      const char *class_name, const char *description,
      lldb_private::ConstString plugin_name, llvm::Triple::OSType preferred_os,
      llvm::SmallVector<llvm::StringRef, 4> supported_triples,
      llvm::StringRef sdk, lldb_private::XcodeSDK::Type sdk_type,
      CoreSimulatorSupport::DeviceType::ProductFamilyID kind);

  static lldb::PlatformSP
  CreateInstance(const char *class_name, const char *description,
                 lldb_private::ConstString plugin_name,
                 llvm::SmallVector<llvm::Triple::ArchType, 4> supported_arch,
                 llvm::Triple::OSType preferred_os,
                 llvm::SmallVector<llvm::Triple::OSType, 4> supported_os,
                 llvm::SmallVector<llvm::StringRef, 4> supported_triples,
                 llvm::StringRef sdk, lldb_private::XcodeSDK::Type sdk_type,
                 CoreSimulatorSupport::DeviceType::ProductFamilyID kind,
                 bool force, const lldb_private::ArchSpec *arch);

  virtual ~PlatformAppleSimulator();

  lldb_private::ConstString GetPluginName() override { return m_plugin_name; }
  const char *GetDescription() override { return m_description; }
  uint32_t GetPluginVersion() override { return 1; }

  lldb_private::Status
  LaunchProcess(lldb_private::ProcessLaunchInfo &launch_info) override;

  void GetStatus(lldb_private::Stream &strm) override;

  lldb_private::Status ConnectRemote(lldb_private::Args &args) override;

  lldb_private::Status DisconnectRemote() override;

  lldb::ProcessSP DebugProcess(lldb_private::ProcessLaunchInfo &launch_info,
                               lldb_private::Debugger &debugger,
                               lldb_private::Target *target,
                               lldb_private::Status &error) override;

  bool GetSupportedArchitectureAtIndex(uint32_t idx,
                                       lldb_private::ArchSpec &arch) override;

  lldb_private::Status ResolveExecutable(
      const lldb_private::ModuleSpec &module_spec, lldb::ModuleSP &module_sp,
      const lldb_private::FileSpecList *module_search_paths_ptr) override;

  lldb_private::Status
  GetSharedModule(const lldb_private::ModuleSpec &module_spec,
                  lldb_private::Process *process, lldb::ModuleSP &module_sp,
                  const lldb_private::FileSpecList *module_search_paths_ptr,
                  llvm::SmallVectorImpl<lldb::ModuleSP> *old_modules,
                  bool *did_create_ptr) override;

  uint32_t
  FindProcesses(const lldb_private::ProcessInstanceInfoMatch &match_info,
                lldb_private::ProcessInstanceInfoList &process_infos) override;

  void
  AddClangModuleCompilationOptions(lldb_private::Target *target,
                                   std::vector<std::string> &options) override {
    return PlatformDarwin::AddClangModuleCompilationOptionsForSDKType(
        target, options, m_sdk_type);
  }

protected:
  const char *m_class_name;
  const char *m_description;
  lldb_private::ConstString m_plugin_name;
  std::mutex m_core_sim_path_mutex;
  llvm::Optional<lldb_private::FileSpec> m_core_simulator_framework_path;
  llvm::Optional<CoreSimulatorSupport::Device> m_device;
  CoreSimulatorSupport::DeviceType::ProductFamilyID m_kind;

  lldb_private::FileSpec GetCoreSimulatorPath();

  llvm::Triple::OSType m_os_type = llvm::Triple::UnknownOS;
  llvm::SmallVector<llvm::StringRef, 4> m_supported_triples = {};
  llvm::StringRef m_sdk;
  lldb_private::XcodeSDK::Type m_sdk_type;

  void LoadCoreSimulator();

#if defined(__APPLE__)
  CoreSimulatorSupport::Device GetSimulatorDevice();
#endif

private:
  PlatformAppleSimulator(const PlatformAppleSimulator &) = delete;
  const PlatformAppleSimulator &
  operator=(const PlatformAppleSimulator &) = delete;
  lldb_private::Status

  GetSymbolFile(const lldb_private::FileSpec &platform_file,
                const lldb_private::UUID *uuid_ptr,
                lldb_private::FileSpec &local_file);
};

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMAPPLESIMULATOR_H
