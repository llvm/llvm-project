//===-- NativeRegisterContextLinux_arm64.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__arm64__) || defined(__aarch64__)

#ifndef lldb_NativeRegisterContextLinux_arm64_h
#define lldb_NativeRegisterContextLinux_arm64_h

#include "Plugins/Process/Linux/NativeRegisterContextLinux.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_arm64.h"

#include <asm/ptrace.h>

namespace lldb_private {
namespace process_linux {

class NativeProcessLinux;

class NativeRegisterContextLinux_arm64 : public NativeRegisterContextLinux {
public:
  NativeRegisterContextLinux_arm64(const ArchSpec &target_arch,
                                   NativeThreadProtocol &native_thread);

  uint32_t GetRegisterSetCount() const override;

  uint32_t GetUserRegisterCount() const override;

  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  Status ReadRegister(const RegisterInfo *reg_info,
                      RegisterValue &reg_value) override;

  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::DataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  void InvalidateAllRegisters() override;

  // Hardware breakpoints/watchpoint management functions

  uint32_t NumSupportedHardwareBreakpoints() override;

  uint32_t SetHardwareBreakpoint(lldb::addr_t addr, size_t size) override;

  bool ClearHardwareBreakpoint(uint32_t hw_idx) override;

  Status ClearAllHardwareBreakpoints() override;

  Status GetHardwareBreakHitIndex(uint32_t &bp_index,
                                  lldb::addr_t trap_addr) override;

  uint32_t NumSupportedHardwareWatchpoints() override;

  uint32_t SetHardwareWatchpoint(lldb::addr_t addr, size_t size,
                                 uint32_t watch_flags) override;

  bool ClearHardwareWatchpoint(uint32_t hw_index) override;

  Status ClearAllHardwareWatchpoints() override;

  Status GetWatchpointHitIndex(uint32_t &wp_index,
                               lldb::addr_t trap_addr) override;

  lldb::addr_t GetWatchpointHitAddress(uint32_t wp_index) override;

  lldb::addr_t GetWatchpointAddress(uint32_t wp_index) override;

  uint32_t GetWatchpointSize(uint32_t wp_index);

  bool WatchpointIsEnabled(uint32_t wp_index);

  // Debug register type select
  enum DREGType { eDREGTypeWATCH = 0, eDREGTypeBREAK };

protected:

  Status ReadGPR() override;

  Status WriteGPR() override;

  Status ReadFPR() override;

  Status WriteFPR() override;

  void *GetGPRBuffer() override { return &m_gpr_arm64; }

  void *GetFPRBuffer() override { return &m_fpr; }

  size_t GetFPRSize() override { return sizeof(m_fpr); }

private:
  bool m_gpr_is_valid;
  bool m_fpu_is_valid;
  bool m_sve_buffer_is_valid;

  bool m_sve_header_is_valid;

  RegisterInfoPOSIX_arm64::GPR m_gpr_arm64; // 64-bit general purpose registers.

  RegisterInfoPOSIX_arm64::FPU
      m_fpr; // floating-point registers including extended register sets.

  SVEState m_sve_state;
  struct user_sve_header m_sve_header;
  std::vector<uint8_t> m_sve_ptrace_payload;

  // Debug register info for hardware breakpoints and watchpoints management.
  struct DREG {
    lldb::addr_t address;  // Breakpoint/watchpoint address value.
    lldb::addr_t hit_addr; // Address at which last watchpoint trigger exception
                           // occurred.
    lldb::addr_t real_addr; // Address value that should cause target to stop.
    uint32_t control;       // Breakpoint/watchpoint control value.
    uint32_t refcount;      // Serves as enable/disable and reference counter.
  };

  struct DREG m_hbr_regs[16]; // Arm native linux hardware breakpoints
  struct DREG m_hwp_regs[16]; // Arm native linux hardware watchpoints

  uint32_t m_max_hwp_supported;
  uint32_t m_max_hbp_supported;
  bool m_refresh_hwdebug_info;

  bool IsGPR(unsigned reg) const;

  bool IsFPR(unsigned reg) const;

  Status ReadAllSVE();

  Status WriteAllSVE();

  Status ReadSVEHeader();

  Status WriteSVEHeader();

  bool IsSVE(unsigned reg) const;

  uint64_t GetSVERegVG() { return m_sve_header.vl / 8; }

  void SetSVERegVG(uint64_t vg) { m_sve_header.vl = vg * 8; }

  void *GetSVEHeader() { return &m_sve_header; }

  void *GetSVEBuffer();

  size_t GetSVEHeaderSize() { return sizeof(m_sve_header); }

  size_t GetSVEBufferSize() { return m_sve_ptrace_payload.size(); }

  Status ReadHardwareDebugInfo();

  Status WriteHardwareDebugRegs(int hwbType);

  uint32_t CalculateFprOffset(const RegisterInfo *reg_info) const;

  RegisterInfoPOSIX_arm64 &GetRegisterInfo() const;

  void ConfigureRegisterContext();

  uint32_t CalculateSVEOffset(const RegisterInfo *reg_info) const;
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_arm64_h

#endif // defined (__arm64__) || defined (__aarch64__)
