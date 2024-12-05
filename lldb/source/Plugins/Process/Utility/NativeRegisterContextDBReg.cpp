//===-- NativeRegisterContextDBReg.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextDBReg.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"

using namespace lldb_private;

uint32_t NativeRegisterContextDBReg::NumSupportedHardwareBreakpoints() {
  Log *log = GetLog(LLDBLog::Breakpoints);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();

  if (error.Fail()) {
    LLDB_LOG(log, "failed to read debug registers");
    return 0;
  }

  LLDB_LOG(log, "{0}", m_max_hbp_supported);
  return m_max_hbp_supported;
}

bool NativeRegisterContextDBReg::ClearHardwareBreakpoint(uint32_t hw_idx) {
  Log *log = GetLog(LLDBLog::Breakpoints);
  LLDB_LOG(log, "hw_idx: {0}", hw_idx);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();
  if (error.Fail()) {
    LLDB_LOG(log,
             "unable to clear breakpoint: failed to read debug registers: {0}");
    return false;
  }

  if (hw_idx >= m_max_hbp_supported)
    return false;

  // Create a backup we can revert to in case of failure.
  lldb::addr_t tempAddr = m_hbp_regs[hw_idx].address;
  uint32_t tempControl = m_hbp_regs[hw_idx].control;

  m_hbp_regs[hw_idx].control &= ~m_hw_dbg_enable_bit;
  m_hbp_regs[hw_idx].address = 0;

  // PTRACE call to clear corresponding hardware breakpoint register.
  error = WriteHardwareDebugRegs(eDREGTypeBREAK);

  if (error.Fail()) {
    m_hbp_regs[hw_idx].control = tempControl;
    m_hbp_regs[hw_idx].address = tempAddr;

    LLDB_LOG(log,
             "unable to clear breakpoint: failed to write debug registers");
    return false;
  }

  return true;
}

Status
NativeRegisterContextDBReg::GetHardwareBreakHitIndex(uint32_t &bp_index,
                                                     lldb::addr_t trap_addr) {
  Log *log = GetLog(LLDBLog::Breakpoints);

  LLDB_LOGF(log, "NativeRegisterContextDBReg::%s()", __FUNCTION__);

  lldb::addr_t break_addr;

  for (bp_index = 0; bp_index < m_max_hbp_supported; ++bp_index) {
    break_addr = m_hbp_regs[bp_index].address;

    if (BreakpointIsEnabled(bp_index) && trap_addr == break_addr) {
      m_hbp_regs[bp_index].hit_addr = trap_addr;
      return Status();
    }
  }

  bp_index = LLDB_INVALID_INDEX32;
  return Status();
}

Status NativeRegisterContextDBReg::ClearAllHardwareBreakpoints() {
  Log *log = GetLog(LLDBLog::Breakpoints);

  LLDB_LOGF(log, "NativeRegisterContextDBReg::%s()", __FUNCTION__);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();
  if (error.Fail())
    return error;

  for (uint32_t i = 0; i < m_max_hbp_supported; i++) {
    if (!BreakpointIsEnabled(i))
      continue;
    // Create a backup we can revert to in case of failure.
    lldb::addr_t tempAddr = m_hbp_regs[i].address;
    uint32_t tempControl = m_hbp_regs[i].control;

    // Clear watchpoints in local cache
    m_hbp_regs[i].control &= ~m_hw_dbg_enable_bit;
    m_hbp_regs[i].address = 0;

    // Ptrace call to update hardware debug registers
    error = WriteHardwareDebugRegs(eDREGTypeBREAK);

    if (error.Fail()) {
      m_hbp_regs[i].control = tempControl;
      m_hbp_regs[i].address = tempAddr;

      return error;
    }
  }

  return Status();
}

bool NativeRegisterContextDBReg::BreakpointIsEnabled(uint32_t bp_index) {
  return ((m_hbp_regs[bp_index].control & m_hw_dbg_enable_bit) != 0);
}

uint32_t NativeRegisterContextDBReg::NumSupportedHardwareWatchpoints() {
  Log *log = GetLog(LLDBLog::Watchpoints);
  Status error = ReadHardwareDebugInfo();
  if (error.Fail()) {
    LLDB_LOG(log, "failed to read debug registers");
    return 0;
  }

  return m_max_hwp_supported;
}

bool NativeRegisterContextDBReg::ClearHardwareWatchpoint(uint32_t wp_index) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();
  if (error.Fail()) {
    LLDB_LOG(log, "unable to set watchpoint: failed to read debug registers");
    return LLDB_INVALID_INDEX32;
  }

  if (wp_index >= m_max_hwp_supported)
    return false;

  // Create a backup we can revert to in case of failure.
  lldb::addr_t tempAddr = m_hwp_regs[wp_index].address;
  uint32_t tempControl = m_hwp_regs[wp_index].control;

  // Update watchpoint in local cache
  m_hwp_regs[wp_index].control &= ~m_hw_dbg_enable_bit;
  m_hwp_regs[wp_index].address = 0;

  // Ptrace call to update hardware debug registers
  error = WriteHardwareDebugRegs(eDREGTypeWATCH);

  if (error.Fail()) {
    m_hwp_regs[wp_index].control = tempControl;
    m_hwp_regs[wp_index].address = tempAddr;

    LLDB_LOG(log, "unable to clear watchpoint: failed to read debug registers");
    return false;
  }

  return true;
}

Status NativeRegisterContextDBReg::ClearAllHardwareWatchpoints() {
  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();
  if (error.Fail())
    return error;

  for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
    if (!WatchpointIsEnabled(i))
      continue;
    // Create a backup we can revert to in case of failure.
    lldb::addr_t tempAddr = m_hwp_regs[i].address;
    uint32_t tempControl = m_hwp_regs[i].control;

    // Clear watchpoints in local cache
    m_hwp_regs[i].control = 0;
    m_hwp_regs[i].address = 0;

    // Ptrace call to update hardware debug registers
    error = WriteHardwareDebugRegs(eDREGTypeWATCH);

    if (error.Fail()) {
      m_hwp_regs[i].control = tempControl;
      m_hwp_regs[i].address = tempAddr;

      return error;
    }
  }

  return Status();
}

Status
NativeRegisterContextDBReg::GetWatchpointHitIndex(uint32_t &wp_index,
                                                  lldb::addr_t trap_addr) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}, trap_addr: {1:x}", wp_index, trap_addr);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();
  if (error.Fail())
    return error;

  // Mask off ignored bits from watchpoint trap address.
  // aarch64
  trap_addr = FixWatchpointHitAddress(trap_addr);

  uint32_t watch_size;
  lldb::addr_t watch_addr;

  for (wp_index = 0; wp_index < m_max_hwp_supported; ++wp_index) {
    watch_size = GetWatchpointSize(wp_index);
    watch_addr = m_hwp_regs[wp_index].address;

    if (WatchpointIsEnabled(wp_index) && trap_addr >= watch_addr &&
        trap_addr < watch_addr + watch_size) {
      m_hwp_regs[wp_index].hit_addr = trap_addr;
      return Status();
    }
  }

  wp_index = LLDB_INVALID_INDEX32;
  return Status();
}

bool NativeRegisterContextDBReg::WatchpointIsEnabled(uint32_t wp_index) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);
  return ((m_hwp_regs[wp_index].control & m_hw_dbg_enable_bit) != 0);
}

lldb::addr_t
NativeRegisterContextDBReg::GetWatchpointAddress(uint32_t wp_index) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  if (wp_index >= m_max_hwp_supported)
    return LLDB_INVALID_ADDRESS;

  if (WatchpointIsEnabled(wp_index))
    return m_hwp_regs[wp_index].real_addr;
  return LLDB_INVALID_ADDRESS;
}

lldb::addr_t
NativeRegisterContextDBReg::GetWatchpointHitAddress(uint32_t wp_index) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  if (wp_index >= m_max_hwp_supported)
    return LLDB_INVALID_ADDRESS;

  if (WatchpointIsEnabled(wp_index))
    return m_hwp_regs[wp_index].hit_addr;
  return LLDB_INVALID_ADDRESS;
}
