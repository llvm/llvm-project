//===-- PlatformAppleWatchSimulator.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMAPPLEWATCHSIMULATOR_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMAPPLEWATCHSIMULATOR_H

#include "PlatformAppleSimulator.h"

class PlatformAppleWatchSimulator : public PlatformAppleSimulator {
public:
  // Class Functions
  static lldb::PlatformSP CreateInstance(bool force,
                                         const lldb_private::ArchSpec *arch);

  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetDescriptionStatic();

  // Class Methods
  PlatformAppleWatchSimulator();

  virtual ~PlatformAppleWatchSimulator();

  // lldb_private::PluginInterface functions
  lldb_private::ConstString GetPluginName() override {
    return GetPluginNameStatic();
  }

  uint32_t GetPluginVersion() override { return 1; }

  // lldb_private::Platform functions
  lldb_private::Status ResolveExecutable(
      const lldb_private::ModuleSpec &module_spec, lldb::ModuleSP &module_sp,
      const lldb_private::FileSpecList *module_search_paths_ptr) override;

  const char *GetDescription() override { return GetDescriptionStatic(); }

  void GetStatus(lldb_private::Stream &strm) override;

  virtual lldb_private::Status
  GetSymbolFile(const lldb_private::FileSpec &platform_file,
                const lldb_private::UUID *uuid_ptr,
                lldb_private::FileSpec &local_file);

  lldb_private::Status
  GetSharedModule(const lldb_private::ModuleSpec &module_spec,
                  lldb_private::Process *process, lldb::ModuleSP &module_sp,
                  const lldb_private::FileSpecList *module_search_paths_ptr,
                  lldb::ModuleSP *old_module_sp_ptr,
                  bool *did_create_ptr) override;

  uint32_t
  FindProcesses(const lldb_private::ProcessInstanceInfoMatch &match_info,
                lldb_private::ProcessInstanceInfoList &process_infos) override;

  void
  AddClangModuleCompilationOptions(lldb_private::Target *target,
                                   std::vector<std::string> &options) override {
    return PlatformDarwin::AddClangModuleCompilationOptionsForSDKType(
        target, options, lldb_private::XcodeSDK::Type::iPhoneSimulator);
  }

protected:
  std::mutex m_sdk_dir_mutex;
  std::string m_sdk_directory;
  std::string m_build_update;

  llvm::StringRef GetSDKDirectoryAsCString();

private:
  PlatformAppleWatchSimulator(const PlatformAppleWatchSimulator &) = delete;
  const PlatformAppleWatchSimulator &
  operator=(const PlatformAppleWatchSimulator &) = delete;
};

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMAPPLEWATCHSIMULATOR_H
