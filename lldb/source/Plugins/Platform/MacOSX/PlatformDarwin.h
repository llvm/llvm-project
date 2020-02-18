//===-- PlatformDarwin.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMDARWIN_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMDARWIN_H

#include "Plugins/Platform/POSIX/PlatformPOSIX.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Utility/FileSpec.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"

#include <string>
#include <tuple>

class PlatformDarwin : public PlatformPOSIX {
public:
  PlatformDarwin(bool is_host);

  ~PlatformDarwin() override;

  // lldb_private::Platform functions
  lldb_private::Status
  ResolveSymbolFile(lldb_private::Target &target,
                    const lldb_private::ModuleSpec &sym_spec,
                    lldb_private::FileSpec &sym_file) override;

  lldb_private::FileSpecList LocateExecutableScriptingResources(
      lldb_private::Target *target, lldb_private::Module &module,
      lldb_private::Stream *feedback_stream) override;

  lldb_private::Status
  GetSharedModule(const lldb_private::ModuleSpec &module_spec,
                  lldb_private::Process *process, lldb::ModuleSP &module_sp,
                  const lldb_private::FileSpecList *module_search_paths_ptr,
                  lldb::ModuleSP *old_module_sp_ptr,
                  bool *did_create_ptr) override;

  size_t GetSoftwareBreakpointTrapOpcode(
      lldb_private::Target &target,
      lldb_private::BreakpointSite *bp_site) override;

  lldb::BreakpointSP
  SetThreadCreationBreakpoint(lldb_private::Target &target) override;

  bool ModuleIsExcludedForUnconstrainedSearches(
      lldb_private::Target &target, const lldb::ModuleSP &module_sp) override;

  bool ARMGetSupportedArchitectureAtIndex(uint32_t idx,
                                          lldb_private::ArchSpec &arch);

  bool x86GetSupportedArchitectureAtIndex(uint32_t idx,
                                          lldb_private::ArchSpec &arch);

  int32_t GetResumeCountForLaunchInfo(
      lldb_private::ProcessLaunchInfo &launch_info) override;

  void CalculateTrapHandlerSymbolNames() override;

  llvm::VersionTuple
  GetOSVersion(lldb_private::Process *process = nullptr) override;

  bool SupportsModules() override { return true; }

  lldb_private::ConstString
  GetFullNameForDylib(lldb_private::ConstString basename) override;

  lldb_private::FileSpec LocateExecutable(const char *basename) override;

  lldb_private::Status
  LaunchProcess(lldb_private::ProcessLaunchInfo &launch_info) override;

  static std::tuple<llvm::VersionTuple, llvm::StringRef>
  ParseVersionBuildDir(llvm::StringRef str);

  enum SDKType : unsigned {
    MacOSX = 0,
    iPhoneSimulator,
    iPhoneOS,
  };

protected:
  void ReadLibdispatchOffsetsAddress(lldb_private::Process *process);

  void ReadLibdispatchOffsets(lldb_private::Process *process);

  virtual lldb_private::Status GetSharedModuleWithLocalCache(
      const lldb_private::ModuleSpec &module_spec, lldb::ModuleSP &module_sp,
      const lldb_private::FileSpecList *module_search_paths_ptr,
      lldb::ModuleSP *old_module_sp_ptr, bool *did_create_ptr);

  static bool SDKSupportsModules(SDKType sdk_type, llvm::VersionTuple version);

  static bool SDKSupportsModules(SDKType desired_type,
                                 const lldb_private::FileSpec &sdk_path);

  struct SDKEnumeratorInfo {
    lldb_private::FileSpec found_path;
    SDKType sdk_type;
  };

  static lldb_private::FileSystem::EnumerateDirectoryResult
  DirectoryEnumerator(void *baton, llvm::sys::fs::file_type file_type,
                      llvm::StringRef path);

  static lldb_private::FileSpec
  FindSDKInXcodeForModules(SDKType sdk_type,
                           const lldb_private::FileSpec &sdks_spec);

  static lldb_private::FileSpec
  GetSDKDirectoryForModules(PlatformDarwin::SDKType sdk_type);

  void
  AddClangModuleCompilationOptionsForSDKType(lldb_private::Target *target,
                                             std::vector<std::string> &options,
                                             SDKType sdk_type);

  const char *GetDeveloperDirectory();

  lldb_private::Status
  FindBundleBinaryInExecSearchPaths (const lldb_private::ModuleSpec &module_spec, lldb_private::Process *process,
                                     lldb::ModuleSP &module_sp, const lldb_private::FileSpecList *module_search_paths_ptr, 
                                     lldb::ModuleSP *old_module_sp_ptr, bool *did_create_ptr);

  std::string m_developer_directory;


private:
  DISALLOW_COPY_AND_ASSIGN(PlatformDarwin);
};

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMDARWIN_H
