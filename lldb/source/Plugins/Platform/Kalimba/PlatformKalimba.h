//===-- PlatformKalimba.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformKalimba_h_
#define liblldb_PlatformKalimba_h_

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
  const char *GetDescription() override {
    return GetPluginDescriptionStatic(IsHost());
  }

  void GetStatus(Stream &strm) override;

  bool GetSupportedArchitectureAtIndex(uint32_t idx, ArchSpec &arch) override;

  size_t GetSoftwareBreakpointTrapOpcode(Target &target,
                                         BreakpointSite *bp_site) override;

  lldb_private::Status
  LaunchProcess(lldb_private::ProcessLaunchInfo &launch_info) override;

  lldb::ProcessSP Attach(ProcessAttachInfo &attach_info, Debugger &debugger,
                         Target *target, Status &error) override;

  // Kalimba processes can not be launched by spawning and attaching.
  bool CanDebugProcess() override { return false; }

  void CalculateTrapHandlerSymbolNames() override;

  UserIDResolver &GetUserIDResolver() override {
    return UserIDResolver::GetNoopResolver();
  }

protected:
  lldb::PlatformSP m_remote_platform_sp;

private:
  DISALLOW_COPY_AND_ASSIGN(PlatformKalimba);
};

} // namespace lldb_private

#endif // liblldb_PlatformKalimba_h_
