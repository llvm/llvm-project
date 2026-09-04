//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_FREEBSD_PLATFORMFREEBSDKERNEL_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_FREEBSD_PLATFORMFREEBSDKERNEL_H

#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Platform.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <utility>
#include <vector>

namespace lldb_private {
namespace platform_freebsdkernel {

class PlatformFreeBSDKernel : public Platform {
public:
  PlatformFreeBSDKernel();

  static void Initialize();

  static void Terminate();

  static lldb::PlatformSP CreateInstance(bool force, const ArchSpec *arch);

  static llvm::StringRef GetPluginNameStatic() { return "freebsd-kernel"; }

  static llvm::StringRef GetPluginDescriptionStatic() {
    return "FreeBSD Kernel platform plug-in.";
  }

  // lldb_private::PluginInterface functions
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  // lldb_private::Platform functions
  llvm::StringRef GetDescription() override {
    return GetPluginDescriptionStatic();
  }

  void GetStatus(Stream &strm) override;

  std::vector<ArchSpec>
  GetSupportedArchitectures(const ArchSpec &process_host_arch) override;

  Status LaunchProcess(ProcessLaunchInfo &launch_info) override {
    return Status::FromErrorString("kernel platform cannot launch processes");
  }

  bool CanDebugProcess() override { return false; }

  lldb::ProcessSP Attach(ProcessAttachInfo &attach_info, Debugger &debugger,
                         Target *target, Status &error) override {
    error =
        Status::FromErrorString("kernel platform cannot attach to processes");
    return {};
  }

  lldb::UnwindPlanSP GetTrapHandlerUnwindPlan(const ArchSpec &arch,
                                              ConstString name) override;

  void CalculateTrapHandlerSymbolNames() override;

  // Called by ProcessFreeBSDKernelCore::DoLoadCore() to populate
  // m_trap_handlers.
  void PopulateTrapHandlerNames(Target &target);

private:
  lldb::UnwindPlanSP
  BuildTrapframeUnwindPlan(llvm::StringRef source_name, uint32_t cfa_dwarf_reg,
                           int32_t cfa_offset,
                           llvm::ArrayRef<std::pair<uint32_t, int32_t>> regs);

  // Per-architecture unwind plan builders.
  // Implemented in TrapframeUnwindPlan_<arch>.cpp.
  lldb::UnwindPlanSP GetTrapframeUnwindPlan_arm64(ConstString name);
  lldb::UnwindPlanSP GetTrapframeUnwindPlan_ppc64le(ConstString name);
  lldb::UnwindPlanSP GetTrapframeUnwindPlan_riscv64(ConstString name);
  lldb::UnwindPlanSP GetTrapframeUnwindPlan_x86_64(ConstString name);

  std::vector<ArchSpec> m_supported_architectures;
};

} // namespace platform_freebsdkernel
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_FREEBSD_PLATFORMFREEBSDKERNEL_H
