//===-- PlatformDarwin.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformDarwin_h_
#define liblldb_PlatformDarwin_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
// Project includes
#include "Plugins/Platform/POSIX/PlatformPOSIX.h"
#include "lldb/Host/FileSpec.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <tuple>

class PlatformDarwin : public PlatformPOSIX {
public:
  PlatformDarwin(bool is_host);

  ~PlatformDarwin() override;

  //------------------------------------------------------------
  // lldb_private::Platform functions
  //------------------------------------------------------------
  lldb_private::Error ResolveExecutable(
      const lldb_private::ModuleSpec &module_spec, lldb::ModuleSP &module_sp,
      const lldb_private::FileSpecList *module_search_paths_ptr) override;

  lldb_private::Error
  ResolveSymbolFile(lldb_private::Target &target,
                    const lldb_private::ModuleSpec &sym_spec,
                    lldb_private::FileSpec &sym_file) override;

  lldb_private::FileSpecList LocateExecutableScriptingResources(
      lldb_private::Target *target, lldb_private::Module &module,
      lldb_private::Stream *feedback_stream) override;

  lldb_private::Error
  GetSharedModule(const lldb_private::ModuleSpec &module_spec,
                  lldb_private::Process *process, lldb::ModuleSP &module_sp,
                  const lldb_private::FileSpecList *module_search_paths_ptr,
                  lldb::ModuleSP *old_module_sp_ptr,
                  bool *did_create_ptr) override;

  size_t GetSoftwareBreakpointTrapOpcode(
      lldb_private::Target &target,
      lldb_private::BreakpointSite *bp_site) override;

  bool GetProcessInfo(lldb::pid_t pid,
                      lldb_private::ProcessInstanceInfo &proc_info) override;

  lldb::BreakpointSP
  SetThreadCreationBreakpoint(lldb_private::Target &target) override;

  uint32_t
  FindProcesses(const lldb_private::ProcessInstanceInfoMatch &match_info,
                lldb_private::ProcessInstanceInfoList &process_infos) override;

  bool ModuleIsExcludedForUnconstrainedSearches(
      lldb_private::Target &target, const lldb::ModuleSP &module_sp) override;

  bool ARMGetSupportedArchitectureAtIndex(uint32_t idx,
                                          lldb_private::ArchSpec &arch);

  bool x86GetSupportedArchitectureAtIndex(uint32_t idx,
                                          lldb_private::ArchSpec &arch);

  int32_t GetResumeCountForLaunchInfo(
      lldb_private::ProcessLaunchInfo &launch_info) override;

  void CalculateTrapHandlerSymbolNames() override;

  bool GetOSVersion(uint32_t &major, uint32_t &minor, uint32_t &update,
                    lldb_private::Process *process = nullptr) override;

  bool SupportsModules() override { return true; }

  lldb_private::ConstString
  GetFullNameForDylib(lldb_private::ConstString basename) override;

  lldb_private::FileSpec LocateExecutable(const char *basename) override;

  lldb_private::Error
  LaunchProcess(lldb_private::ProcessLaunchInfo &launch_info) override;

  static std::tuple<uint32_t, uint32_t, uint32_t, llvm::StringRef>
  ParseVersionBuildDir(llvm::StringRef str);

protected:
  void ReadLibdispatchOffsetsAddress(lldb_private::Process *process);

  void ReadLibdispatchOffsets(lldb_private::Process *process);

  virtual lldb_private::Error GetSharedModuleWithLocalCache(
      const lldb_private::ModuleSpec &module_spec, lldb::ModuleSP &module_sp,
      const lldb_private::FileSpecList *module_search_paths_ptr,
      lldb::ModuleSP *old_module_sp_ptr, bool *did_create_ptr);

  enum class SDKType {
    MacOSX = 0,
    iPhoneSimulator,
    iPhoneOS,
  };

  static bool SDKSupportsModules(SDKType sdk_type, uint32_t major,
                                 uint32_t minor, uint32_t micro);

  static bool SDKSupportsModules(SDKType desired_type,
                                 const lldb_private::FileSpec &sdk_path);

  struct SDKEnumeratorInfo {
    lldb_private::FileSpec found_path;
    SDKType sdk_type;
  };

  static lldb_private::FileSpec::EnumerateDirectoryResult
  DirectoryEnumerator(void *baton, lldb_private::FileSpec::FileType file_type,
                      const lldb_private::FileSpec &spec);

  static lldb_private::FileSpec
  FindSDKInXcodeForModules(SDKType sdk_type,
                           const lldb_private::FileSpec &sdks_spec);

  static lldb_private::FileSpec
  GetSDKDirectoryForModules(PlatformDarwin::SDKType sdk_type);

  void
  AddClangModuleCompilationOptionsForSDKType(lldb_private::Target *target,
                                             std::vector<std::string> &options,
                                             SDKType sdk_type);

  std::string m_developer_directory;

  const char *GetDeveloperDirectory();

private:
  DISALLOW_COPY_AND_ASSIGN(PlatformDarwin);
};

#endif // liblldb_PlatformDarwin_h_
