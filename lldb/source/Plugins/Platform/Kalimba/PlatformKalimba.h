//===-- PlatformKalimba.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformKalimba_h_
#define liblldb_PlatformKalimba_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"

namespace lldb_private {

class PlatformKalimba : public Platform {
public:
  PlatformKalimba(bool is_host);

  ~PlatformKalimba() override;

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
  Error ResolveExecutable(const lldb_private::ModuleSpec &module_spec,
                          lldb::ModuleSP &module_sp,
                          const FileSpecList *module_search_paths_ptr) override;

  const char *GetDescription() override {
    return GetPluginDescriptionStatic(IsHost());
  }

  void GetStatus(Stream &strm) override;

  Error GetFileWithUUID(const FileSpec &platform_file, const UUID *uuid,
                        FileSpec &local_file) override;

  bool GetProcessInfo(lldb::pid_t pid, ProcessInstanceInfo &proc_info) override;

  bool GetSupportedArchitectureAtIndex(uint32_t idx, ArchSpec &arch) override;

  size_t GetSoftwareBreakpointTrapOpcode(Target &target,
                                         BreakpointSite *bp_site) override;

  lldb_private::Error
  LaunchProcess(lldb_private::ProcessLaunchInfo &launch_info) override;

  lldb::ProcessSP Attach(ProcessAttachInfo &attach_info, Debugger &debugger,
                         Target *target, Error &error) override;

  // Kalimba processes can not be launched by spawning and attaching.
  bool CanDebugProcess() override { return false; }

  void CalculateTrapHandlerSymbolNames() override;

protected:
  lldb::PlatformSP m_remote_platform_sp;

private:
  DISALLOW_COPY_AND_ASSIGN(PlatformKalimba);
};

} // namespace lldb_private

#endif // liblldb_PlatformKalimba_h_
