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
  llvm::Error error = ReadHardwareDebugInfo();

  if (error) {
    LLDB_LOG_ERROR(log, std::move(error),
                   "failed to read debug registers: {0}");
    return 0;
  }

  LLDB_LOG(log, "{0}", m_max_hbp_supported);
  return m_max_hbp_supported;
}

uint32_t NativeRegisterContextDBReg::SetHardwareBreakpoint(lldb::addr_t addr,
                                                           size_t size) {
  Log *log = GetLog(LLDBLog::Breakpoints);
  LLDB_LOG(log, "addr: {0:x}, size: {1:x}", addr, size);

  // Read hardware breakpoint and watchpoint information.
  llvm::Error error = ReadHardwareDebugInfo();
  if (error) {
    LLDB_LOG_ERROR(
        log, std::move(error),
        "unable to set breakpoint: failed to read debug registers: {0}");
    return LLDB_INVALID_INDEX32;
  }

  if (!ValidateBreakpoint(size, addr))
    return LLDB_INVALID_INDEX32;

  uint32_t control_value = MakeBreakControlValue(size);
  auto details = AdjustBreakpoint({size, addr});
  size = details.size;
  addr = details.addr;

  // Iterate over stored breakpoints and find a free bp_index
  uint32_t bp_index = LLDB_INVALID_INDEX32;
  for (uint32_t i = 0; i < m_max_hbp_supported; i++) {
    if (!BreakpointIsEnabled(i))
      bp_index = i; // Mark last free slot
    else if (m_hbp_regs[i].address == addr)
      return LLDB_INVALID_INDEX32; // We do not support duplicate breakpoints.
  }

  if (bp_index == LLDB_INVALID_INDEX32)
    return LLDB_INVALID_INDEX32;

  // Update breakpoint in local cache
  m_hbp_regs[bp_index].real_addr = addr;
  m_hbp_regs[bp_index].address = addr;
  m_hbp_regs[bp_index].control = control_value;

  // PTRACE call to set corresponding hardware breakpoint register.
  error = WriteHardwareDebugRegs(eDREGTypeBREAK);

  if (error) {
    m_hbp_regs[bp_index].address = 0;
    m_hbp_regs[bp_index].control &= ~m_hw_dbg_enable_bit;

    LLDB_LOG_ERROR(
        log, std::move(error),
        "unable to set breakpoint: failed to write debug registers: {0}");
    return LLDB_INVALID_INDEX32;
  }

  return bp_index;
}

bool NativeRegisterContextDBReg::ClearHardwareBreakpoint(uint32_t hw_idx) {
  Log *log = GetLog(LLDBLog::Breakpoints);
  LLDB_LOG(log, "hw_idx: {0}", hw_idx);

  // Read hardware breakpoint and watchpoint information.
  llvm::Error error = ReadHardwareDebugInfo();
  if (error) {
    LLDB_LOG_ERROR(
        log, std::move(error),
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

  if (error) {
    m_hbp_regs[hw_idx].control = tempControl;
    m_hbp_regs[hw_idx].address = tempAddr;

    LLDB_LOG_ERROR(
        log, std::move(error),
        "unable to clear breakpoint: failed to write debug registers: {0}");
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
  llvm::Error error = ReadHardwareDebugInfo();
  if (error)
    return Status::FromError(std::move(error));

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

    if (error) {
      m_hbp_regs[i].control = tempControl;
      m_hbp_regs[i].address = tempAddr;

      return Status::FromError(std::move(error));
    }
  }

  return Status();
}

bool NativeRegisterContextDBReg::BreakpointIsEnabled(uint32_t bp_index) {
  return ((m_hbp_regs[bp_index].control & m_hw_dbg_enable_bit) != 0);
}

uint32_t NativeRegisterContextDBReg::NumSupportedHardwareWatchpoints() {
  Log *log = GetLog(LLDBLog::Watchpoints);
  llvm::Error error = ReadHardwareDebugInfo();
  if (error) {
    LLDB_LOG_ERROR(log, std::move(error),
                   "failed to read debug registers: {0}");
    return 0;
  }

  return m_max_hwp_supported;
}

uint32_t NativeRegisterContextDBReg::SetHardwareWatchpoint(
    lldb::addr_t addr, size_t size, uint32_t watch_flags) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "addr: {0:x}, size: {1:x} watch_flags: {2:x}", addr, size,
           watch_flags);

  // Read hardware breakpoint and watchpoint information.
  llvm::Error error = ReadHardwareDebugInfo();
  if (error) {
    LLDB_LOG_ERROR(
        log, std::move(error),
        "unable to set watchpoint: failed to read debug registers: {0}");
    return LLDB_INVALID_INDEX32;
  }

  uint32_t control_value = 0, wp_index = 0;
  lldb::addr_t real_addr = addr;
  WatchpointDetails details{size, addr};

  auto adjusted = AdjustWatchpoint(details);
  if (adjusted == std::nullopt)
    return LLDB_INVALID_INDEX32;
  size = adjusted->size;
  addr = adjusted->addr;

  // Check if we are setting watchpoint other than read/write/access Also
  // update watchpoint flag to match ARM/AArch64/LoongArch write-read bit
  // configuration.
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

  control_value = MakeWatchControlValue(size, watch_flags);

  // Iterate over stored watchpoints and find a free wp_index
  wp_index = LLDB_INVALID_INDEX32;
  for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
    if (!WatchpointIsEnabled(i))
      wp_index = i; // Mark last free slot
    else if (m_hwp_regs[i].address == addr) {
      return LLDB_INVALID_INDEX32; // We do not support duplicate watchpoints.
    }
  }

  if (wp_index == LLDB_INVALID_INDEX32)
    return LLDB_INVALID_INDEX32;

  // Update watchpoint in local cache
  m_hwp_regs[wp_index].real_addr = real_addr;
  m_hwp_regs[wp_index].address = addr;
  m_hwp_regs[wp_index].control = control_value;

  // PTRACE call to set corresponding watchpoint register.
  error = WriteHardwareDebugRegs(eDREGTypeWATCH);

  if (error) {
    m_hwp_regs[wp_index].address = 0;
    m_hwp_regs[wp_index].control &= ~m_hw_dbg_enable_bit;

    LLDB_LOG_ERROR(
        log, std::move(error),
        "unable to set watchpoint: failed to write debug registers: {0}");
    return LLDB_INVALID_INDEX32;
  }

  return wp_index;
}

bool NativeRegisterContextDBReg::ClearHardwareWatchpoint(uint32_t wp_index) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  // Read hardware breakpoint and watchpoint information.
  llvm::Error error = ReadHardwareDebugInfo();
  if (error) {
    LLDB_LOG_ERROR(
        log, std::move(error),
        "unable to set watchpoint: failed to read debug registers: {0}");
    return false;
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

  if (error) {
    m_hwp_regs[wp_index].control = tempControl;
    m_hwp_regs[wp_index].address = tempAddr;

    LLDB_LOG_ERROR(
        log, std::move(error),
        "unable to clear watchpoint: failed to read debug registers: {0}");
    return false;
  }

  return true;
}

Status NativeRegisterContextDBReg::ClearAllHardwareWatchpoints() {
  // Read hardware breakpoint and watchpoint information.
  llvm::Error error = ReadHardwareDebugInfo();
  if (error)
    return Status::FromError(std::move(error));

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

    if (error) {
      m_hwp_regs[i].control = tempControl;
      m_hwp_regs[i].address = tempAddr;

      return Status::FromError(std::move(error));
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
  llvm::Error error = ReadHardwareDebugInfo();
  if (error)
    return Status::FromError(std::move(error));

  // AArch64 need mask off ignored bits from watchpoint trap address.
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
