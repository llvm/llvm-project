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
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/Utility/XcodeSDK.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"

#include <string>
#include <tuple>

class PlatformDarwin : public PlatformPOSIX {
public:
  PlatformDarwin(bool is_host);

  ~PlatformDarwin() override;

  lldb_private::Status PutFile(const lldb_private::FileSpec &source,
                               const lldb_private::FileSpec &destination,
                               uint32_t uid = UINT32_MAX,
                               uint32_t gid = UINT32_MAX) override;

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
                  llvm::SmallVectorImpl<lldb::ModuleSP> *old_modules,
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

  uint32_t GetResumeCountForLaunchInfo(
      lldb_private::ProcessLaunchInfo &launch_info) override;

  lldb::ProcessSP DebugProcess(lldb_private::ProcessLaunchInfo &launch_info,
                               lldb_private::Debugger &debugger,
                               lldb_private::Target &target,
                               lldb_private::Status &error) override;

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

  llvm::Expected<lldb_private::StructuredData::DictionarySP>
  FetchExtendedCrashInformation(lldb_private::Process &process) override;

  /// Return the toolchain directory the current LLDB instance is located in.
  static lldb_private::FileSpec GetCurrentToolchainDirectory();

  /// Return the command line tools directory the current LLDB instance is
  /// located in.
  static lldb_private::FileSpec GetCurrentCommandLineToolsDirectory();

protected:
  struct CrashInfoAnnotations {
    uint64_t version;          // unsigned long
    uint64_t message;          // char *
    uint64_t signature_string; // char *
    uint64_t backtrace;        // char *
    uint64_t message2;         // char *
    uint64_t thread;           // uint64_t
    uint64_t dialog_mode;      // unsigned int
    uint64_t abort_cause;      // unsigned int
  };

  /// Extract the `__crash_info` annotations from each of of the target's
  /// modules.
  ///
  /// If the platform have a crashed processes with a `__crash_info` section,
  /// extract the section to gather the messages annotations and the abort
  /// cause.
  ///
  /// \param[in] process
  ///     The crashed process.
  ///
  /// \return
  ///     A  structured data array containing at each entry in each entry, the
  ///     module spec, its UUID, the crash messages and the abort cause.
  ///     \b nullptr if process has no crash information annotations.
  lldb_private::StructuredData::ArraySP
  ExtractCrashInfoAnnotations(lldb_private::Process &process);

  void ReadLibdispatchOffsetsAddress(lldb_private::Process *process);

  void ReadLibdispatchOffsets(lldb_private::Process *process);

  virtual lldb_private::Status GetSharedModuleWithLocalCache(
      const lldb_private::ModuleSpec &module_spec, lldb::ModuleSP &module_sp,
      const lldb_private::FileSpecList *module_search_paths_ptr,
      llvm::SmallVectorImpl<lldb::ModuleSP> *old_modules, bool *did_create_ptr);

  struct SDKEnumeratorInfo {
    lldb_private::FileSpec found_path;
    lldb_private::XcodeSDK::Type sdk_type;
  };

  static lldb_private::FileSystem::EnumerateDirectoryResult
  DirectoryEnumerator(void *baton, llvm::sys::fs::file_type file_type,
                      llvm::StringRef path);

  static lldb_private::FileSpec
  FindSDKInXcodeForModules(lldb_private::XcodeSDK::Type sdk_type,
                           const lldb_private::FileSpec &sdks_spec);

  static lldb_private::FileSpec
  GetSDKDirectoryForModules(lldb_private::XcodeSDK::Type sdk_type);

  void AddClangModuleCompilationOptionsForSDKType(
      lldb_private::Target *target, std::vector<std::string> &options,
      lldb_private::XcodeSDK::Type sdk_type);

  lldb_private::Status FindBundleBinaryInExecSearchPaths(
      const lldb_private::ModuleSpec &module_spec,
      lldb_private::Process *process, lldb::ModuleSP &module_sp,
      const lldb_private::FileSpecList *module_search_paths_ptr,
      llvm::SmallVectorImpl<lldb::ModuleSP> *old_modules, bool *did_create_ptr);

  static std::string FindComponentInPath(llvm::StringRef path,
                                         llvm::StringRef component);

  std::string m_developer_directory;
  llvm::StringMap<std::string> m_sdk_path;
  std::mutex m_sdk_path_mutex;

private:
  PlatformDarwin(const PlatformDarwin &) = delete;
  const PlatformDarwin &operator=(const PlatformDarwin &) = delete;
};

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMDARWIN_H
