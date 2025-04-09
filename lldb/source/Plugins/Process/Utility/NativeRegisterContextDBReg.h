//===-- NativeRegisterContextDBReg.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_NativeRegisterContextDBReg_h
#define lldb_NativeRegisterContextDBReg_h

#include "Plugins/Process/Utility/NativeRegisterContextRegisterInfo.h"

#include <array>

// Common utilities for hardware breakpoints and hardware watchpoints on AArch64
// and LoongArch.

namespace lldb_private {

class NativeRegisterContextDBReg
    : public virtual NativeRegisterContextRegisterInfo {
public:
  explicit NativeRegisterContextDBReg(uint32_t enable_bit)
      : m_hw_dbg_enable_bit(enable_bit) {}

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

protected:
  // Debug register type select
  enum DREGType { eDREGTypeWATCH = 0, eDREGTypeBREAK };

  /// Debug register info for hardware breakpoints and watchpoints management.
  struct DREG {
    lldb::addr_t address;  // Breakpoint/watchpoint address value.
    lldb::addr_t hit_addr; // Address at which last watchpoint trigger exception
                           // occurred.
    lldb::addr_t real_addr; // Address value that should cause target to stop.
    uint32_t control;       // Breakpoint/watchpoint control value.
  };

  std::array<struct DREG, 16> m_hbp_regs; // hardware breakpoints
  std::array<struct DREG, 16> m_hwp_regs; // hardware watchpoints

  uint32_t m_max_hbp_supported;
  uint32_t m_max_hwp_supported;
  const uint32_t m_hw_dbg_enable_bit;

  bool WatchpointIsEnabled(uint32_t wp_index);
  bool BreakpointIsEnabled(uint32_t bp_index);

  // On AArch64 and Loongarch the hardware breakpoint length size is 4, and the
  // target address must 4-byte alignment.
  bool ValidateBreakpoint(size_t size, lldb::addr_t addr) {
    return (size == 4) && !(addr & 0x3);
  }
  struct WatchpointDetails {
    size_t size;
    lldb::addr_t addr;
  };
  virtual std::optional<WatchpointDetails>
  AdjustWatchpoint(const WatchpointDetails &details) = 0;
  virtual uint32_t MakeBreakControlValue(size_t size) = 0;
  virtual uint32_t MakeWatchControlValue(size_t size, uint32_t watch_flags) = 0;
  virtual uint32_t GetWatchpointSize(uint32_t wp_index) = 0;

  virtual llvm::Error ReadHardwareDebugInfo() = 0;
  virtual llvm::Error WriteHardwareDebugRegs(DREGType hwbType) = 0;
  virtual lldb::addr_t FixWatchpointHitAddress(lldb::addr_t hit_addr) {
    return hit_addr;
  }
};

} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextDBReg_h
