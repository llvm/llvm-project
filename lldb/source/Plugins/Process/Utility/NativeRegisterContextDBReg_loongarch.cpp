//===-- NativeRegisterContextDBReg_loongarch.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextDBReg_loongarch.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"

using namespace lldb_private;

uint32_t
NativeRegisterContextDBReg_loongarch::SetHardwareBreakpoint(lldb::addr_t addr,
                                                            size_t size) {
  Log *log = GetLog(LLDBLog::Breakpoints);
  LLDB_LOG(log, "addr: {0:x}, size: {1:x}", addr, size);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();
  if (error.Fail()) {
    LLDB_LOG(log, "unable to set breakpoint: failed to read debug registers");
    return LLDB_INVALID_INDEX32;
  }

  uint32_t bp_index = 0;

  // Check if size has a valid hardware breakpoint length.
  if (size != 4)
    return LLDB_INVALID_INDEX32; // Invalid size for a LoongArch hardware
                                 // breakpoint

  // Check 4-byte alignment for hardware breakpoint target address.
  if (addr & 0x03)
    return LLDB_INVALID_INDEX32; // Invalid address, should be 4-byte aligned.

  // Iterate over stored breakpoints and find a free bp_index
  bp_index = LLDB_INVALID_INDEX32;
  for (uint32_t i = 0; i < m_max_hbp_supported; i++) {
    if (!BreakpointIsEnabled(i))
      bp_index = i; // Mark last free slot
    else if (m_hbp_regs[i].address == addr)
      return LLDB_INVALID_INDEX32; // We do not support duplicate breakpoints.
  }

  if (bp_index == LLDB_INVALID_INDEX32)
    return LLDB_INVALID_INDEX32;

  // Update breakpoint in local cache
  m_hbp_regs[bp_index].address = addr;
  m_hbp_regs[bp_index].control = m_hw_dbg_enable_bit;

  // PTRACE call to set corresponding hardware breakpoint register.
  error = WriteHardwareDebugRegs(eDREGTypeBREAK);

  if (error.Fail()) {
    m_hbp_regs[bp_index].address = 0;
    m_hbp_regs[bp_index].control &= ~m_hw_dbg_enable_bit;

    LLDB_LOG(log, "unable to set breakpoint: failed to write debug registers");
    return LLDB_INVALID_INDEX32;
  }

  return bp_index;
}

uint32_t
NativeRegisterContextDBReg_loongarch::GetWatchpointSize(uint32_t wp_index) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  switch ((m_hwp_regs[wp_index].control >> 10) & 0x3) {
  case 0x0:
    return 8;
  case 0x1:
    return 4;
  case 0x2:
    return 2;
  case 0x3:
    return 1;
  default:
    return 0;
  }
}

uint32_t NativeRegisterContextDBReg_loongarch::SetHardwareWatchpoint(
    lldb::addr_t addr, size_t size, uint32_t watch_flags) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "addr: {0:x}, size: {1:x} watch_flags: {2:x}", addr, size,
           watch_flags);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();

  if (error.Fail()) {
    LLDB_LOG(log, "unable to set watchpoint: failed to read debug registers");
    return LLDB_INVALID_INDEX32;
  }

  uint32_t control_value = 0, wp_index = 0;

  // Check if we are setting watchpoint other than read/write/access Update
  // watchpoint flag to match loongarch64 write-read bit configuration.
  switch (watch_flags) {
  case lldb::eWatchpointKindWrite:
    watch_flags = 2;
    break;
  case lldb::eWatchpointKindRead:
    watch_flags = 1;
    break;
  case (lldb::eWatchpointKindRead | lldb::eWatchpointKindWrite):
    break;
  default:
    return LLDB_INVALID_INDEX32;
  }

  // Check if size has a valid hardware watchpoint length.
  if (size != 1 && size != 2 && size != 4 && size != 8)
    return LLDB_INVALID_INDEX32;

  // Encode appropriate control register bits for the specified size
  // size encoded:
  // case 1 : 0b11
  // case 2 : 0b10
  // case 4 : 0b01
  // case 8 : 0b00
  auto EncodeSizeBits = [](int size) {
    return (3 - llvm::Log2_32(size)) << 10;
  };

  // Setup control value
  control_value = m_hw_dbg_enable_bit | EncodeSizeBits(size);
  control_value |= watch_flags << 8;

  // Iterate over stored watchpoints and find a free wp_index
  wp_index = LLDB_INVALID_INDEX32;
  for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
    if (!WatchpointIsEnabled(i)) {
      wp_index = i; // Mark last free slot
    } else if (m_hwp_regs[i].address == addr) {
      return LLDB_INVALID_INDEX32; // We do not support duplicate watchpoints.
    }
  }

  if (wp_index == LLDB_INVALID_INDEX32)
    return LLDB_INVALID_INDEX32;

  // Update watchpoint in local cache
  // Note: `NativeRegisterContextDBReg::GetWatchpointAddress` return `real_addr`
  m_hwp_regs[wp_index].real_addr = addr;
  m_hwp_regs[wp_index].address = addr;
  m_hwp_regs[wp_index].control = control_value;

  // PTRACE call to set corresponding watchpoint register.
  error = WriteHardwareDebugRegs(eDREGTypeWATCH);

  if (error.Fail()) {
    m_hwp_regs[wp_index].address = 0;
    m_hwp_regs[wp_index].control &= ~m_hw_dbg_enable_bit;

    LLDB_LOG(log, "unable to set watchpoint: failed to write debug registers");
    return LLDB_INVALID_INDEX32;
  }

  return wp_index;
}
