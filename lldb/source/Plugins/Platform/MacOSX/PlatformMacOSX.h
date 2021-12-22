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
  PlatformMacOSX();

  // Class functions
  static lldb::PlatformSP CreateInstance(bool force,
                                         const lldb_private::ArchSpec *arch);

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return Platform::GetHostPlatformName();
  }

  static llvm::StringRef GetDescriptionStatic();

  // lldb_private::PluginInterface functions
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  lldb_private::Status
  GetSharedModule(const lldb_private::ModuleSpec &module_spec,
                  lldb_private::Process *process, lldb::ModuleSP &module_sp,
                  const lldb_private::FileSpecList *module_search_paths_ptr,
                  llvm::SmallVectorImpl<lldb::ModuleSP> *old_modules,
                  bool *did_create_ptr) override;

  llvm::StringRef GetDescription() override { return GetDescriptionStatic(); }

  lldb_private::Status
  GetFile(const lldb_private::FileSpec &source,
          const lldb_private::FileSpec &destination) override {
    return PlatformDarwin::GetFile(source, destination);
  }

  std::vector<lldb_private::ArchSpec> GetSupportedArchitectures() override;

  lldb_private::ConstString
  GetSDKDirectory(lldb_private::Target &target) override;

  void
  AddClangModuleCompilationOptions(lldb_private::Target *target,
                                   std::vector<std::string> &options) override {
    return PlatformDarwin::AddClangModuleCompilationOptionsForSDKType(
        target, options, lldb_private::XcodeSDK::Type::MacOSX);
  }
};

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMMACOSX_H
