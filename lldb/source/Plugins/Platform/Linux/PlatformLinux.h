//===-- PlatformLinux.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformLinux_h_
#define liblldb_PlatformLinux_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "Plugins/Platform/POSIX/PlatformPOSIX.h"

namespace lldb_private {
namespace platform_linux {

class PlatformLinux : public PlatformPOSIX {
public:
  PlatformLinux(bool is_host);

  ~PlatformLinux() override;

  static void DebuggerInitialize(Debugger &debugger);

  static void Initialize();

  static void Terminate();

  //------------------------------------------------------------
  // lldb_private::PluginInterface functions
  //------------------------------------------------------------
  static lldb::PlatformSP CreateInstance(bool force, const ArchSpec *arch);

  static ConstString GetPluginNameStatic(bool is_host);

  static const char *GetPluginDescriptionStatic(bool is_host);

  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override { return 1; }

  //------------------------------------------------------------
  // lldb_private::Platform functions
  //------------------------------------------------------------
  Error ResolveExecutable(const ModuleSpec &module_spec,
                          lldb::ModuleSP &module_sp,
                          const FileSpecList *module_search_paths_ptr) override;

  const char *GetDescription() override {
    return GetPluginDescriptionStatic(IsHost());
  }

  void GetStatus(Stream &strm) override;

  Error GetFileWithUUID(const FileSpec &platform_file, const UUID *uuid,
                        FileSpec &local_file) override;

  bool GetProcessInfo(lldb::pid_t pid, ProcessInstanceInfo &proc_info) override;

  uint32_t FindProcesses(const ProcessInstanceInfoMatch &match_info,
                         ProcessInstanceInfoList &process_infos) override;

  bool GetSupportedArchitectureAtIndex(uint32_t idx, ArchSpec &arch) override;

  int32_t GetResumeCountForLaunchInfo(ProcessLaunchInfo &launch_info) override;

  bool CanDebugProcess() override;

  lldb::ProcessSP DebugProcess(ProcessLaunchInfo &launch_info,
                               Debugger &debugger, Target *target,
                               Error &error) override;

  void CalculateTrapHandlerSymbolNames() override;

  uint64_t ConvertMmapFlagsToPlatform(const ArchSpec &arch,
                                      unsigned flags) override;

  ConstString GetFullNameForDylib(ConstString basename) override;

private:
  DISALLOW_COPY_AND_ASSIGN(PlatformLinux);
};

} // namespace platform_linux
} // namespace lldb_private

#endif // liblldb_PlatformLinux_h_
