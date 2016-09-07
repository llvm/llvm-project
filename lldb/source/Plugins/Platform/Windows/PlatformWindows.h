//===-- PlatformWindows.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformWindows_h_
#define liblldb_PlatformWindows_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"

namespace lldb_private {

class PlatformWindows : public Platform {
public:
  PlatformWindows(bool is_host);

  ~PlatformWindows() override;

  static void Initialize();

  static void Terminate();

  //------------------------------------------------------------
  // lldb_private::PluginInterface functions
  //------------------------------------------------------------
  static lldb::PlatformSP CreateInstance(bool force,
                                         const lldb_private::ArchSpec *arch);

  static lldb_private::ConstString GetPluginNameStatic(bool is_host);

  static const char *GetPluginDescriptionStatic(bool is_host);

  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override { return 1; }

  //------------------------------------------------------------
  // lldb_private::Platform functions
  //------------------------------------------------------------
  bool GetModuleSpec(const lldb_private::FileSpec &module_file_spec,
                     const lldb_private::ArchSpec &arch,
                     lldb_private::ModuleSpec &module_spec) override;

  Error ResolveExecutable(const lldb_private::ModuleSpec &module_spec,
                          lldb::ModuleSP &module_sp,
                          const FileSpecList *module_search_paths_ptr) override;

  const char *GetDescription() override {
    return GetPluginDescriptionStatic(IsHost());
  }

  bool GetRemoteOSVersion() override;

  bool GetRemoteOSBuildString(std::string &s) override;

  bool GetRemoteOSKernelDescription(std::string &s) override;

  // Remote Platform subclasses need to override this function
  lldb_private::ArchSpec GetRemoteSystemArchitecture() override;

  bool IsConnected() const override;

  lldb_private::Error ConnectRemote(lldb_private::Args &args) override;

  lldb_private::Error DisconnectRemote() override;

  const char *GetHostname() override;

  const char *GetUserName(uint32_t uid) override;

  const char *GetGroupName(uint32_t gid) override;

  bool GetProcessInfo(lldb::pid_t pid,
                      lldb_private::ProcessInstanceInfo &proc_info) override;

  uint32_t
  FindProcesses(const lldb_private::ProcessInstanceInfoMatch &match_info,
                lldb_private::ProcessInstanceInfoList &process_infos) override;

  lldb_private::Error
  LaunchProcess(lldb_private::ProcessLaunchInfo &launch_info) override;

  lldb::ProcessSP DebugProcess(lldb_private::ProcessLaunchInfo &launch_info,
                               lldb_private::Debugger &debugger,
                               lldb_private::Target *target,
                               lldb_private::Error &error) override;

  lldb::ProcessSP Attach(lldb_private::ProcessAttachInfo &attach_info,
                         lldb_private::Debugger &debugger,
                         lldb_private::Target *target,
                         lldb_private::Error &error) override;

  lldb_private::Error
  GetFileWithUUID(const lldb_private::FileSpec &platform_file,
                  const lldb_private::UUID *uuid,
                  lldb_private::FileSpec &local_file) override;

  lldb_private::Error
  GetSharedModule(const lldb_private::ModuleSpec &module_spec,
                  lldb_private::Process *process, lldb::ModuleSP &module_sp,
                  const lldb_private::FileSpecList *module_search_paths_ptr,
                  lldb::ModuleSP *old_module_sp_ptr,
                  bool *did_create_ptr) override;

  bool GetSupportedArchitectureAtIndex(uint32_t idx,
                                       lldb_private::ArchSpec &arch) override;

  void GetStatus(lldb_private::Stream &strm) override;

  bool CanDebugProcess() override;

  size_t GetEnvironment(StringList &env) override;

  // FIXME not sure what the _sigtramp equivalent would be on this platform
  void CalculateTrapHandlerSymbolNames() override {}

  ConstString GetFullNameForDylib(ConstString basename) override;

protected:
  lldb::PlatformSP m_remote_platform_sp;

private:
  DISALLOW_COPY_AND_ASSIGN(PlatformWindows);
};

} // namespace lldb_private

#endif // liblldb_PlatformWindows_h_
