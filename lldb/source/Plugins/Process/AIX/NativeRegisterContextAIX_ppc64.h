//===-- NativeRegisterContextAIX_ppc64.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This implementation is related to the OpenPOWER ABI for Power Architecture
// 64-bit ELF V2 ABI

#if defined(__powerpc64__)

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_AIX_NATIVEREGISTERCONTEXTAIX_PPC64_H
#define LLDB_SOURCE_PLUGINS_PROCESS_AIX_NATIVEREGISTERCONTEXTAIX_PPC64_H

#include "Plugins/Process/AIX/NativeRegisterContextAIX.h"
#include "Plugins/Process/Utility/lldb-ppc64-register-enums.h"

#define DECLARE_REGISTER_INFOS_PPC64_STRUCT
#include "Plugins/Process/Utility/RegisterInfos_ppc64.h"
#undef DECLARE_REGISTER_INFOS_PPC64_STRUCT

namespace lldb_private {
namespace process_aix {

class NativeProcessAIX;

class NativeRegisterContextAIX_ppc64 : public NativeRegisterContextAIX {
public:
  NativeRegisterContextAIX_ppc64(const ArchSpec &target_arch,
                                     NativeThreadProtocol &native_thread);

  uint32_t GetRegisterSetCount() const override;

  uint32_t GetUserRegisterCount() const override;

  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  Status ReadRegister(const RegisterInfo *reg_info,
                      RegisterValue &reg_value) override;

  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  // Hardware watchpoint management functions

  uint32_t NumSupportedHardwareWatchpoints() override;

  uint32_t SetHardwareWatchpoint(lldb::addr_t addr, size_t size,
                                 uint32_t watch_flags) override;

  bool ClearHardwareWatchpoint(uint32_t hw_index) override;

  Status GetWatchpointHitIndex(uint32_t &wp_index,
                               lldb::addr_t trap_addr) override;

  lldb::addr_t GetWatchpointHitAddress(uint32_t wp_index) override;

  lldb::addr_t GetWatchpointAddress(uint32_t wp_index) override;

  uint32_t GetWatchpointSize(uint32_t wp_index);

  bool WatchpointIsEnabled(uint32_t wp_index);

protected:
  bool IsVMX(unsigned reg);

  bool IsVSX(unsigned reg);

  Status ReadVMX();

  Status WriteVMX();

  Status ReadVSX();

  Status WriteVSX();

  void *GetGPRBuffer() override { return &m_gpr_ppc64; }

  void *GetFPRBuffer() override { return &m_fpr_ppc64; }

  size_t GetFPRSize() override { return sizeof(m_fpr_ppc64); }

private:
  GPR_PPC64 m_gpr_ppc64; // 64-bit general purpose registers.
  FPR_PPC64 m_fpr_ppc64; // floating-point registers including extended register.
  VMX_PPC64 m_vmx_ppc64; // VMX registers.
  VSX_PPC64 m_vsx_ppc64; // Last lower bytes from first VSX registers.

  bool IsGPR(unsigned reg) const;

  bool IsFPR(unsigned reg) const;

  bool IsVMX(unsigned reg) const;

  bool IsVSX(unsigned reg) const;

  uint32_t CalculateFprOffset(const RegisterInfo *reg_info) const;

  uint32_t CalculateVmxOffset(const RegisterInfo *reg_info) const;

  uint32_t CalculateVsxOffset(const RegisterInfo *reg_info) const;

  Status ReadHardwareDebugInfo();

  Status WriteHardwareDebugRegs();

  // Debug register info for hardware watchpoints management.
  struct DREG {
    lldb::addr_t address;   // Breakpoint/watchpoint address value.
    lldb::addr_t hit_addr;  // Address at which last watchpoint trigger
                            // exception occurred.
    lldb::addr_t real_addr; // Address value that should cause target to stop.
    uint32_t control;       // Breakpoint/watchpoint control value.
    uint32_t refcount;      // Serves as enable/disable and reference counter.
    long slot;              // Saves the value returned from PTRACE_SETHWDEBUG.
    int mode;               // Defines if watchpoint is read/write/access.
  };

  std::array<DREG, 4> m_hwp_regs;

  // 16 is just a maximum value, query hardware for actual watchpoint count
  uint32_t m_max_hwp_supported = 16;
  uint32_t m_max_hbp_supported = 16;
  bool m_refresh_hwdebug_info = true;
};

} // namespace process_aix
} // namespace lldb_private

#endif // #ifndef LLDB_SOURCE_PLUGINS_PROCESS_AIX_NATIVEREGISTECONTXTAIX_PPC64_H 

#endif // defined(__powerpc64__)
