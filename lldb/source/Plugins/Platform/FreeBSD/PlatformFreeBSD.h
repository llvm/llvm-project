//===-- PlatformFreeBSD.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformFreeBSD_h_
#define liblldb_PlatformFreeBSD_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"

namespace lldb_private {
namespace platform_freebsd {

class PlatformFreeBSD : public Platform {
public:
  PlatformFreeBSD(bool is_host);

  ~PlatformFreeBSD() override;

  //------------------------------------------------------------
  // Class functions
  //------------------------------------------------------------
  static lldb::PlatformSP CreateInstance(bool force, const ArchSpec *arch);

  static void Initialize();

  static void Terminate();

  static ConstString GetPluginNameStatic(bool is_host);

  static const char *GetDescriptionStatic(bool is_host);

  //------------------------------------------------------------
  // lldb_private::PluginInterface functions
  //------------------------------------------------------------
  ConstString GetPluginName() override { return GetPluginNameStatic(IsHost()); }

  uint32_t GetPluginVersion() override { return 1; }

  const char *GetDescription() override {
    return GetDescriptionStatic(IsHost());
  }

  //------------------------------------------------------------
  // lldb_private::Platform functions
  //------------------------------------------------------------
  bool GetModuleSpec(const FileSpec &module_file_spec, const ArchSpec &arch,
                     ModuleSpec &module_spec) override;

  Error RunShellCommand(const char *command, const FileSpec &working_dir,
                        int *status_ptr, int *signo_ptr,
                        std::string *command_output,
                        uint32_t timeout_sec) override;

  Error ResolveExecutable(const ModuleSpec &module_spec,
                          lldb::ModuleSP &module_sp,
                          const FileSpecList *module_search_paths_ptr) override;

  size_t GetSoftwareBreakpointTrapOpcode(Target &target,
                                         BreakpointSite *bp_site) override;

  bool GetRemoteOSVersion() override;

  bool GetRemoteOSBuildString(std::string &s) override;

  bool GetRemoteOSKernelDescription(std::string &s) override;

  // Remote Platform subclasses need to override this function
  ArchSpec GetRemoteSystemArchitecture() override;

  bool IsConnected() const override;

  Error ConnectRemote(Args &args) override;

  Error DisconnectRemote() override;

  const char *GetHostname() override;

  const char *GetUserName(uint32_t uid) override;

  const char *GetGroupName(uint32_t gid) override;

  bool GetProcessInfo(lldb::pid_t pid, ProcessInstanceInfo &proc_info) override;

  uint32_t FindProcesses(const ProcessInstanceInfoMatch &match_info,
                         ProcessInstanceInfoList &process_infos) override;

  Error LaunchProcess(ProcessLaunchInfo &launch_info) override;

  lldb::ProcessSP Attach(ProcessAttachInfo &attach_info, Debugger &debugger,
                         Target *target, Error &error) override;

  // FreeBSD processes can not be launched by spawning and attaching.
  bool CanDebugProcess() override { return false; }

  // Only on PlatformMacOSX:
  Error GetFileWithUUID(const FileSpec &platform_file, const UUID *uuid,
                        FileSpec &local_file) override;

  Error GetSharedModule(const ModuleSpec &module_spec, Process *process,
                        lldb::ModuleSP &module_sp,
                        const FileSpecList *module_search_paths_ptr,
                        lldb::ModuleSP *old_module_sp_ptr,
                        bool *did_create_ptr) override;

  bool GetSupportedArchitectureAtIndex(uint32_t idx, ArchSpec &arch) override;

  void GetStatus(Stream &strm) override;

  void CalculateTrapHandlerSymbolNames() override;

protected:
  lldb::PlatformSP m_remote_platform_sp; // Allow multiple ways to connect to a
                                         // remote freebsd OS

private:
  DISALLOW_COPY_AND_ASSIGN(PlatformFreeBSD);
};

} // namespace platform_freebsd
} // namespace lldb_private

#endif // liblldb_PlatformFreeBSD_h_
