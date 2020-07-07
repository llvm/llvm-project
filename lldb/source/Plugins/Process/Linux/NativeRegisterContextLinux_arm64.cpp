//===-- NativeRegisterContextLinux_arm64.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__arm64__) || defined(__aarch64__)

#include "NativeRegisterContextLinux_arm.h"
#include "NativeRegisterContextLinux_arm64.h"


#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/Procfs.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"

// System includes - They have to be included after framework includes because
// they define some macros which collide with variable names in other modules
#include <sys/socket.h>
// NT_PRSTATUS and NT_FPREGSET definition
#include <elf.h>
// user_hwdebug_state definition
#include <asm/ptrace.h>

#define REG_CONTEXT_SIZE (GetGPRSize() + GetFPRSize())

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

std::unique_ptr<NativeRegisterContextLinux>
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::arm:
    return std::make_unique<NativeRegisterContextLinux_arm>(target_arch,
                                                             native_thread);
  case llvm::Triple::aarch64:
    return std::make_unique<NativeRegisterContextLinux_arm64>(target_arch,
                                                               native_thread);
  default:
    llvm_unreachable("have no register context for architecture");
  }
}

NativeRegisterContextLinux_arm64::NativeRegisterContextLinux_arm64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContextLinux(native_thread,
                                 new RegisterInfoPOSIX_arm64(target_arch)) {
  ::memset(&m_fpr, 0, sizeof(m_fpr));
  ::memset(&m_gpr_arm64, 0, sizeof(m_gpr_arm64));
  ::memset(&m_hwp_regs, 0, sizeof(m_hwp_regs));
  ::memset(&m_hbr_regs, 0, sizeof(m_hbr_regs));

  // 16 is just a maximum value, query hardware for actual watchpoint count
  m_max_hwp_supported = 16;
  m_max_hbp_supported = 16;
  m_refresh_hwdebug_info = true;

  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
}

RegisterInfoPOSIX_arm64 &
NativeRegisterContextLinux_arm64::GetRegisterInfo() const {
  return static_cast<RegisterInfoPOSIX_arm64 &>(*m_register_info_interface_up);
}

uint32_t NativeRegisterContextLinux_arm64::GetRegisterSetCount() const {
  return GetRegisterInfo().GetRegisterSetCount();
}

const RegisterSet *
NativeRegisterContextLinux_arm64::GetRegisterSet(uint32_t set_index) const {
  return GetRegisterInfo().GetRegisterSet(set_index);
}

uint32_t NativeRegisterContextLinux_arm64::GetUserRegisterCount() const {
  uint32_t count = 0;
  for (uint32_t set_index = 0; set_index < GetRegisterSetCount(); ++set_index)
    count += GetRegisterSet(set_index)->num_registers;
  return count;
}

Status
NativeRegisterContextLinux_arm64::ReadRegister(const RegisterInfo *reg_info,
                                               RegisterValue &reg_value) {
  Status error;

  if (!reg_info) {
    error.SetErrorString("reg_info NULL");
    return error;
  }

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (reg == LLDB_INVALID_REGNUM)
    return Status("no lldb regnum for %s", reg_info && reg_info->name
                                               ? reg_info->name
                                               : "<unknown register>");

  uint8_t *src;
  uint32_t offset;

  if (IsGPR(reg)) {
    if (!m_gpr_is_valid) {
      error = ReadGPR();
      if (error.Fail())
        return error;
    }

    offset = reg_info->byte_offset;
    assert(offset < GetGPRSize());
    src = (uint8_t *)GetGPRBuffer() + offset;

  } else if (IsFPR(reg)) {
    if (!m_fpu_is_valid) {

      error = ReadFPR();
      if (error.Fail())
        return error;
    }
    offset = CalculateFprOffset(reg_info);
    assert(offset < GetFPRSize());
    src = (uint8_t *)GetFPRBuffer() + offset;
  } else
    return Status("failed - register wasn't recognized to be a GPR or an FPR, "
                  "write strategy unknown");

  reg_value.SetFromMemoryData(reg_info, src, reg_info->byte_size,
                              eByteOrderLittle, error);

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  Status error;

  if (!reg_info)
    return Status("reg_info NULL");

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (reg == LLDB_INVALID_REGNUM)
    return Status("no lldb regnum for %s", reg_info && reg_info->name
                                               ? reg_info->name
                                               : "<unknown register>");

  uint8_t *dst;
  uint32_t offset;

  if (IsGPR(reg)) {
    if (!m_gpr_is_valid) {
      error = ReadGPR();
      if (error.Fail())
        return error;
    }

    offset = reg_info->byte_offset;
    assert(offset < GetGPRSize());
    dst = (uint8_t *)GetGPRBuffer() + offset;

    ::memcpy(dst, reg_value.GetBytes(), reg_info->byte_size);

    return WriteGPR();
  } else if (IsFPR(reg)) {
    if (!m_fpu_is_valid) {
      error = ReadFPR();
      if (error.Fail())
        return error;
    }
    offset = CalculateFprOffset(reg_info);
    assert(offset < GetFPRSize());
    dst = (uint8_t *)GetFPRBuffer() + offset;

    ::memcpy(dst, reg_value.GetBytes(), reg_info->byte_size);

    return WriteFPR();
  }

  return error;
}

Status NativeRegisterContextLinux_arm64::ReadAllRegisterValues(
    lldb::DataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(REG_CONTEXT_SIZE, 0));
  if (!m_gpr_is_valid) {
    error = ReadGPR();
    if (error.Fail())
      return error;
  }

  if (!m_fpu_is_valid) {
    error = ReadFPR();
    if (error.Fail())
      return error;
  }
  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, GetGPRBuffer(), GetGPRSize());
  dst += GetGPRSize();
  ::memcpy(dst, GetFPRBuffer(), GetFPRSize());

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error.SetErrorStringWithFormat(
        "NativeRegisterContextLinux_x86_64::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != REG_CONTEXT_SIZE) {
    error.SetErrorStringWithFormat(
        "NativeRegisterContextLinux_x86_64::%s data_sp contained mismatched "
        "data size, expected %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, REG_CONTEXT_SIZE, data_sp->GetByteSize());
    return error;
  }

  uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error.SetErrorStringWithFormat("NativeRegisterContextLinux_x86_64::%s "
                                   "DataBuffer::GetBytes() returned a null "
                                   "pointer",
                                   __FUNCTION__);
    return error;
  }
  ::memcpy(GetGPRBuffer(), src, GetRegisterInfoInterface().GetGPRSize());

  error = WriteGPR();
  if (error.Fail())
    return error;

  src += GetRegisterInfoInterface().GetGPRSize();
  ::memcpy(GetFPRBuffer(), src, GetFPRSize());

  error = WriteFPR();
  if (error.Fail())
    return error;

  return error;
}

bool NativeRegisterContextLinux_arm64::IsGPR(unsigned reg) const {
  if (GetRegisterInfo().GetRegisterSetFromRegisterIndex(reg) ==
      RegisterInfoPOSIX_arm64::GPRegSet)
    return true;
  return false;
}

bool NativeRegisterContextLinux_arm64::IsFPR(unsigned reg) const {
  if (GetRegisterInfo().GetRegisterSetFromRegisterIndex(reg) ==
      RegisterInfoPOSIX_arm64::FPRegSet)
    return true;
  return false;
}

uint32_t NativeRegisterContextLinux_arm64::NumSupportedHardwareBreakpoints() {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_BREAKPOINTS));

  LLDB_LOGF(log, "NativeRegisterContextLinux_arm64::%s()", __FUNCTION__);

  Status error;

  // Read hardware breakpoint and watchpoint information.
  error = ReadHardwareDebugInfo();

  if (error.Fail())
    return 0;

  return m_max_hbp_supported;
}

uint32_t
NativeRegisterContextLinux_arm64::SetHardwareBreakpoint(lldb::addr_t addr,
                                                        size_t size) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_BREAKPOINTS));
  LLDB_LOG(log, "addr: {0:x}, size: {1:x}", addr, size);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();

  if (error.Fail())
    return LLDB_INVALID_INDEX32;

  uint32_t control_value = 0, bp_index = 0;

  // Check if size has a valid hardware breakpoint length.
  if (size != 4)
    return LLDB_INVALID_INDEX32; // Invalid size for a AArch64 hardware
                                 // breakpoint

  // Check 4-byte alignment for hardware breakpoint target address.
  if (addr & 0x03)
    return LLDB_INVALID_INDEX32; // Invalid address, should be 4-byte aligned.

  // Setup control value
  control_value = 0;
  control_value |= ((1 << size) - 1) << 5;
  control_value |= (2 << 1) | 1;

  // Iterate over stored breakpoints and find a free bp_index
  bp_index = LLDB_INVALID_INDEX32;
  for (uint32_t i = 0; i < m_max_hbp_supported; i++) {
    if ((m_hbr_regs[i].control & 1) == 0) {
      bp_index = i; // Mark last free slot
    } else if (m_hbr_regs[i].address == addr) {
      return LLDB_INVALID_INDEX32; // We do not support duplicate breakpoints.
    }
  }

  if (bp_index == LLDB_INVALID_INDEX32)
    return LLDB_INVALID_INDEX32;

  // Update breakpoint in local cache
  m_hbr_regs[bp_index].real_addr = addr;
  m_hbr_regs[bp_index].address = addr;
  m_hbr_regs[bp_index].control = control_value;

  // PTRACE call to set corresponding hardware breakpoint register.
  error = WriteHardwareDebugRegs(eDREGTypeBREAK);

  if (error.Fail()) {
    m_hbr_regs[bp_index].address = 0;
    m_hbr_regs[bp_index].control &= ~1;

    return LLDB_INVALID_INDEX32;
  }

  return bp_index;
}

bool NativeRegisterContextLinux_arm64::ClearHardwareBreakpoint(
    uint32_t hw_idx) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_BREAKPOINTS));
  LLDB_LOG(log, "hw_idx: {0}", hw_idx);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();

  if (error.Fail())
    return false;

  if (hw_idx >= m_max_hbp_supported)
    return false;

  // Create a backup we can revert to in case of failure.
  lldb::addr_t tempAddr = m_hbr_regs[hw_idx].address;
  uint32_t tempControl = m_hbr_regs[hw_idx].control;

  m_hbr_regs[hw_idx].control &= ~1;
  m_hbr_regs[hw_idx].address = 0;

  // PTRACE call to clear corresponding hardware breakpoint register.
  error = WriteHardwareDebugRegs(eDREGTypeBREAK);

  if (error.Fail()) {
    m_hbr_regs[hw_idx].control = tempControl;
    m_hbr_regs[hw_idx].address = tempAddr;

    return false;
  }

  return true;
}

Status NativeRegisterContextLinux_arm64::GetHardwareBreakHitIndex(
    uint32_t &bp_index, lldb::addr_t trap_addr) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_BREAKPOINTS));

  LLDB_LOGF(log, "NativeRegisterContextLinux_arm64::%s()", __FUNCTION__);

  lldb::addr_t break_addr;

  for (bp_index = 0; bp_index < m_max_hbp_supported; ++bp_index) {
    break_addr = m_hbr_regs[bp_index].address;

    if ((m_hbr_regs[bp_index].control & 0x1) && (trap_addr == break_addr)) {
      m_hbr_regs[bp_index].hit_addr = trap_addr;
      return Status();
    }
  }

  bp_index = LLDB_INVALID_INDEX32;
  return Status();
}

Status NativeRegisterContextLinux_arm64::ClearAllHardwareBreakpoints() {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_BREAKPOINTS));

  LLDB_LOGF(log, "NativeRegisterContextLinux_arm64::%s()", __FUNCTION__);

  Status error;

  // Read hardware breakpoint and watchpoint information.
  error = ReadHardwareDebugInfo();

  if (error.Fail())
    return error;

  lldb::addr_t tempAddr = 0;
  uint32_t tempControl = 0;

  for (uint32_t i = 0; i < m_max_hbp_supported; i++) {
    if (m_hbr_regs[i].control & 0x01) {
      // Create a backup we can revert to in case of failure.
      tempAddr = m_hbr_regs[i].address;
      tempControl = m_hbr_regs[i].control;

      // Clear watchpoints in local cache
      m_hbr_regs[i].control &= ~1;
      m_hbr_regs[i].address = 0;

      // Ptrace call to update hardware debug registers
      error = WriteHardwareDebugRegs(eDREGTypeBREAK);

      if (error.Fail()) {
        m_hbr_regs[i].control = tempControl;
        m_hbr_regs[i].address = tempAddr;

        return error;
      }
    }
  }

  return Status();
}

uint32_t NativeRegisterContextLinux_arm64::NumSupportedHardwareWatchpoints() {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_WATCHPOINTS));

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();

  if (error.Fail())
    return 0;

  LLDB_LOG(log, "{0}", m_max_hwp_supported);
  return m_max_hwp_supported;
}

uint32_t NativeRegisterContextLinux_arm64::SetHardwareWatchpoint(
    lldb::addr_t addr, size_t size, uint32_t watch_flags) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_WATCHPOINTS));
  LLDB_LOG(log, "addr: {0:x}, size: {1:x} watch_flags: {2:x}", addr, size,
           watch_flags);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();

  if (error.Fail())
    return LLDB_INVALID_INDEX32;

  uint32_t control_value = 0, wp_index = 0;
  lldb::addr_t real_addr = addr;

  // Check if we are setting watchpoint other than read/write/access Also
  // update watchpoint flag to match AArch64 write-read bit configuration.
  switch (watch_flags) {
  case 1:
    watch_flags = 2;
    break;
  case 2:
    watch_flags = 1;
    break;
  case 3:
    break;
  default:
    return LLDB_INVALID_INDEX32;
  }

  // Check if size has a valid hardware watchpoint length.
  if (size != 1 && size != 2 && size != 4 && size != 8)
    return LLDB_INVALID_INDEX32;

  // Check 8-byte alignment for hardware watchpoint target address. Below is a
  // hack to recalculate address and size in order to make sure we can watch
  // non 8-byte aligned addresses as well.
  if (addr & 0x07) {
    uint8_t watch_mask = (addr & 0x07) + size;

    if (watch_mask > 0x08)
      return LLDB_INVALID_INDEX32;
    else if (watch_mask <= 0x02)
      size = 2;
    else if (watch_mask <= 0x04)
      size = 4;
    else
      size = 8;

    addr = addr & (~0x07);
  }

  // Setup control value
  control_value = watch_flags << 3;
  control_value |= ((1 << size) - 1) << 5;
  control_value |= (2 << 1) | 1;

  // Iterate over stored watchpoints and find a free wp_index
  wp_index = LLDB_INVALID_INDEX32;
  for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
    if ((m_hwp_regs[i].control & 1) == 0) {
      wp_index = i; // Mark last free slot
    } else if (m_hwp_regs[i].address == addr) {
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

  if (error.Fail()) {
    m_hwp_regs[wp_index].address = 0;
    m_hwp_regs[wp_index].control &= ~1;

    return LLDB_INVALID_INDEX32;
  }

  return wp_index;
}

bool NativeRegisterContextLinux_arm64::ClearHardwareWatchpoint(
    uint32_t wp_index) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_WATCHPOINTS));
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();

  if (error.Fail())
    return false;

  if (wp_index >= m_max_hwp_supported)
    return false;

  // Create a backup we can revert to in case of failure.
  lldb::addr_t tempAddr = m_hwp_regs[wp_index].address;
  uint32_t tempControl = m_hwp_regs[wp_index].control;

  // Update watchpoint in local cache
  m_hwp_regs[wp_index].control &= ~1;
  m_hwp_regs[wp_index].address = 0;

  // Ptrace call to update hardware debug registers
  error = WriteHardwareDebugRegs(eDREGTypeWATCH);

  if (error.Fail()) {
    m_hwp_regs[wp_index].control = tempControl;
    m_hwp_regs[wp_index].address = tempAddr;

    return false;
  }

  return true;
}

Status NativeRegisterContextLinux_arm64::ClearAllHardwareWatchpoints() {
  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();

  if (error.Fail())
    return error;

  lldb::addr_t tempAddr = 0;
  uint32_t tempControl = 0;

  for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
    if (m_hwp_regs[i].control & 0x01) {
      // Create a backup we can revert to in case of failure.
      tempAddr = m_hwp_regs[i].address;
      tempControl = m_hwp_regs[i].control;

      // Clear watchpoints in local cache
      m_hwp_regs[i].control &= ~1;
      m_hwp_regs[i].address = 0;

      // Ptrace call to update hardware debug registers
      error = WriteHardwareDebugRegs(eDREGTypeWATCH);

      if (error.Fail()) {
        m_hwp_regs[i].control = tempControl;
        m_hwp_regs[i].address = tempAddr;

        return error;
      }
    }
  }

  return Status();
}

uint32_t
NativeRegisterContextLinux_arm64::GetWatchpointSize(uint32_t wp_index) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_WATCHPOINTS));
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  switch ((m_hwp_regs[wp_index].control >> 5) & 0xff) {
  case 0x01:
    return 1;
  case 0x03:
    return 2;
  case 0x0f:
    return 4;
  case 0xff:
    return 8;
  default:
    return 0;
  }
}
bool NativeRegisterContextLinux_arm64::WatchpointIsEnabled(uint32_t wp_index) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_WATCHPOINTS));
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  if ((m_hwp_regs[wp_index].control & 0x1) == 0x1)
    return true;
  else
    return false;
}

Status NativeRegisterContextLinux_arm64::GetWatchpointHitIndex(
    uint32_t &wp_index, lldb::addr_t trap_addr) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_WATCHPOINTS));
  LLDB_LOG(log, "wp_index: {0}, trap_addr: {1:x}", wp_index, trap_addr);

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

lldb::addr_t
NativeRegisterContextLinux_arm64::GetWatchpointAddress(uint32_t wp_index) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_WATCHPOINTS));
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  if (wp_index >= m_max_hwp_supported)
    return LLDB_INVALID_ADDRESS;

  if (WatchpointIsEnabled(wp_index))
    return m_hwp_regs[wp_index].real_addr;
  else
    return LLDB_INVALID_ADDRESS;
}

lldb::addr_t
NativeRegisterContextLinux_arm64::GetWatchpointHitAddress(uint32_t wp_index) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_WATCHPOINTS));
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  if (wp_index >= m_max_hwp_supported)
    return LLDB_INVALID_ADDRESS;

  if (WatchpointIsEnabled(wp_index))
    return m_hwp_regs[wp_index].hit_addr;
  else
    return LLDB_INVALID_ADDRESS;
}

Status NativeRegisterContextLinux_arm64::ReadHardwareDebugInfo() {
  if (!m_refresh_hwdebug_info) {
    return Status();
  }

  ::pid_t tid = m_thread.GetID();

  int regset = NT_ARM_HW_WATCH;
  struct iovec ioVec;
  struct user_hwdebug_state dreg_state;
  Status error;

  ioVec.iov_base = &dreg_state;
  ioVec.iov_len = sizeof(dreg_state);
  error = NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET, tid, &regset,
                                            &ioVec, ioVec.iov_len);

  if (error.Fail())
    return error;

  m_max_hwp_supported = dreg_state.dbg_info & 0xff;

  regset = NT_ARM_HW_BREAK;
  error = NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET, tid, &regset,
                                            &ioVec, ioVec.iov_len);

  if (error.Fail())
    return error;

  m_max_hbp_supported = dreg_state.dbg_info & 0xff;
  m_refresh_hwdebug_info = false;

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteHardwareDebugRegs(int hwbType) {
  struct iovec ioVec;
  struct user_hwdebug_state dreg_state;
  Status error;

  memset(&dreg_state, 0, sizeof(dreg_state));
  ioVec.iov_base = &dreg_state;

  if (hwbType == eDREGTypeWATCH) {
    hwbType = NT_ARM_HW_WATCH;
    ioVec.iov_len = sizeof(dreg_state.dbg_info) + sizeof(dreg_state.pad) +
                    (sizeof(dreg_state.dbg_regs[0]) * m_max_hwp_supported);

    for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
      dreg_state.dbg_regs[i].addr = m_hwp_regs[i].address;
      dreg_state.dbg_regs[i].ctrl = m_hwp_regs[i].control;
    }
  } else {
    hwbType = NT_ARM_HW_BREAK;
    ioVec.iov_len = sizeof(dreg_state.dbg_info) + sizeof(dreg_state.pad) +
                    (sizeof(dreg_state.dbg_regs[0]) * m_max_hbp_supported);

    for (uint32_t i = 0; i < m_max_hbp_supported; i++) {
      dreg_state.dbg_regs[i].addr = m_hbr_regs[i].address;
      dreg_state.dbg_regs[i].ctrl = m_hbr_regs[i].control;
    }
  }

  return NativeProcessLinux::PtraceWrapper(PTRACE_SETREGSET, m_thread.GetID(),
                                           &hwbType, &ioVec, ioVec.iov_len);
}

Status NativeRegisterContextLinux_arm64::ReadGPR() {
  Status error;

  struct iovec ioVec;

  ioVec.iov_base = GetGPRBuffer();
  ioVec.iov_len = GetGPRSize();

  error = ReadRegisterSet(&ioVec, GetGPRSize(), NT_PRSTATUS);

  if (error.Success())
    m_gpr_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteGPR() {
  struct iovec ioVec;

  m_gpr_is_valid = false;

  ioVec.iov_base = GetGPRBuffer();
  ioVec.iov_len = GetGPRSize();

  return WriteRegisterSet(&ioVec, GetGPRSize(), NT_PRSTATUS);
}

Status NativeRegisterContextLinux_arm64::ReadFPR() {
  Status error;

  struct iovec ioVec;

  ioVec.iov_base = GetFPRBuffer();
  ioVec.iov_len = GetFPRSize();

  error = ReadRegisterSet(&ioVec, GetFPRSize(), NT_FPREGSET);

  if (error.Success())
    m_fpu_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_arm64::WriteFPR() {
  struct iovec ioVec;

  m_fpu_is_valid = false;

  ioVec.iov_base = GetFPRBuffer();
  ioVec.iov_len = GetFPRSize();

  return WriteRegisterSet(&ioVec, GetFPRSize(), NT_FPREGSET);
}

void NativeRegisterContextLinux_arm64::InvalidateAllRegisters() {
  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
}

uint32_t NativeRegisterContextLinux_arm64::CalculateFprOffset(
    const RegisterInfo *reg_info) const {
  return reg_info->byte_offset - GetGPRSize();
}

#endif // defined (__arm64__) || defined (__aarch64__)
