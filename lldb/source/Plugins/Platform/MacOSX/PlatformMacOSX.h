//===-- PlatformMacOSX.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMMACOSX_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMMACOSX_H

#include "PlatformDarwin.h"

class PlatformMacOSX : public PlatformDarwin {
public:
  PlatformMacOSX(bool is_host);

  ~PlatformMacOSX() override;

  // Class functions
  static lldb::PlatformSP CreateInstance(bool force,
                                         const lldb_private::ArchSpec *arch);

  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic(bool is_host);

  static const char *GetDescriptionStatic(bool is_host);

  // lldb_private::PluginInterface functions
  lldb_private::ConstString GetPluginName() override {
    return GetPluginNameStatic(IsHost());
  }

  uint32_t GetPluginVersion() override { return 1; }

  lldb_private::Status
  GetSharedModule(const lldb_private::ModuleSpec &module_spec,
                  lldb_private::Process *process, lldb::ModuleSP &module_sp,
                  const lldb_private::FileSpecList *module_search_paths_ptr,
                  lldb::ModuleSP *old_module_sp_ptr,
                  bool *did_create_ptr) override;

  const char *GetDescription() override {
    return GetDescriptionStatic(IsHost());
  }

  lldb_private::Status
  GetSymbolFile(const lldb_private::FileSpec &platform_file,
                const lldb_private::UUID *uuid_ptr,
                lldb_private::FileSpec &local_file);

  lldb_private::Status
  GetFile(const lldb_private::FileSpec &source,
          const lldb_private::FileSpec &destination) override {
    return PlatformDarwin::GetFile(source, destination);
  }

  lldb_private::Status
  GetFileWithUUID(const lldb_private::FileSpec &platform_file,
                  const lldb_private::UUID *uuid_ptr,
                  lldb_private::FileSpec &local_file) override;

  bool GetSupportedArchitectureAtIndex(uint32_t idx,
                                       lldb_private::ArchSpec &arch) override;

  lldb_private::ConstString
  GetSDKDirectory(lldb_private::Target &target) override;

  void
  AddClangModuleCompilationOptions(lldb_private::Target *target,
                                   std::vector<std::string> &options) override {
    return PlatformDarwin::AddClangModuleCompilationOptionsForSDKType(
        target, options, lldb_private::XcodeSDK::Type::MacOSX);
  }

private:
  PlatformMacOSX(const PlatformMacOSX &) = delete;
  const PlatformMacOSX &operator=(const PlatformMacOSX &) = delete;

#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
  uint32_t m_num_arm_arches = 0;
#endif
};

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMMACOSX_H
