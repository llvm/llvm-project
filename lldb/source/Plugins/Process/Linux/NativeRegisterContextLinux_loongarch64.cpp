//===-- NativeRegisterContextLinux_loongarch64.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__loongarch__) && __loongarch_grlen == 64

#include "NativeRegisterContextLinux_loongarch64.h"

#include "lldb/Host/HostInfo.h"
#include "lldb/Host/linux/Ptrace.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/Procfs.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_loongarch64.h"
#include "Plugins/Process/Utility/lldb-loongarch-register-enums.h"

// NT_PRSTATUS and NT_FPREGSET definition
#include <elf.h>
// struct iovec definition
#include <sys/uio.h>

#define REG_CONTEXT_SIZE (GetGPRSize() + GetFPRSize())

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

// CTRL_PLV3_ENABLE, used to enable breakpoint/watchpoint
constexpr uint32_t g_enable_bit = 0x10;

// Returns appropriate control register bits for the specified size
// size encoded:
// case 1 : 0b11
// case 2 : 0b10
// case 4 : 0b01
// case 8 : 0b00
static inline uint64_t GetSizeBits(int size) {
  return (3 - llvm::Log2_32(size)) << 10;
}

std::unique_ptr<NativeRegisterContextLinux>
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(
    const ArchSpec &target_arch, NativeThreadLinux &native_thread) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::loongarch64: {
    Flags opt_regsets;
    auto register_info_up = std::make_unique<RegisterInfoPOSIX_loongarch64>(
        target_arch, opt_regsets);
    return std::make_unique<NativeRegisterContextLinux_loongarch64>(
        target_arch, native_thread, std::move(register_info_up));
  }
  default:
    llvm_unreachable("have no register context for architecture");
  }
}

llvm::Expected<ArchSpec>
NativeRegisterContextLinux::DetermineArchitecture(lldb::tid_t tid) {
  return HostInfo::GetArchitecture();
}

NativeRegisterContextLinux_loongarch64::NativeRegisterContextLinux_loongarch64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread,
    std::unique_ptr<RegisterInfoPOSIX_loongarch64> register_info_up)
    : NativeRegisterContextRegisterInfo(native_thread,
                                        register_info_up.release()),
      NativeRegisterContextLinux(native_thread) {
  ::memset(&m_fpr, 0, sizeof(m_fpr));
  ::memset(&m_gpr, 0, sizeof(m_gpr));
  ::memset(&m_hwp_regs, 0, sizeof(m_hwp_regs));
  ::memset(&m_hbp_regs, 0, sizeof(m_hbp_regs));

  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
}

const RegisterInfoPOSIX_loongarch64 &
NativeRegisterContextLinux_loongarch64::GetRegisterInfo() const {
  return static_cast<const RegisterInfoPOSIX_loongarch64 &>(
      NativeRegisterContextRegisterInfo::GetRegisterInfoInterface());
}

uint32_t NativeRegisterContextLinux_loongarch64::GetRegisterSetCount() const {
  return GetRegisterInfo().GetRegisterSetCount();
}

const RegisterSet *NativeRegisterContextLinux_loongarch64::GetRegisterSet(
    uint32_t set_index) const {
  return GetRegisterInfo().GetRegisterSet(set_index);
}

uint32_t NativeRegisterContextLinux_loongarch64::GetUserRegisterCount() const {
  uint32_t count = 0;
  for (uint32_t set_index = 0; set_index < GetRegisterSetCount(); ++set_index)
    count += GetRegisterSet(set_index)->num_registers;
  return count;
}

Status NativeRegisterContextLinux_loongarch64::ReadRegister(
    const RegisterInfo *reg_info, RegisterValue &reg_value) {
  Status error;

  if (!reg_info) {
    error = Status::FromErrorString("reg_info NULL");
    return error;
  }

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (reg == LLDB_INVALID_REGNUM)
    return Status::FromErrorStringWithFormat(
        "no lldb regnum for %s",
        reg_info && reg_info->name ? reg_info->name : "<unknown register>");

  uint8_t *src = nullptr;
  uint32_t offset = LLDB_INVALID_INDEX32;

  if (IsGPR(reg)) {
    error = ReadGPR();
    if (error.Fail())
      return error;

    offset = reg_info->byte_offset;
    assert(offset < GetGPRSize());
    src = (uint8_t *)GetGPRBuffer() + offset;

  } else if (IsFPR(reg)) {
    error = ReadFPR();
    if (error.Fail())
      return error;

    offset = CalculateFprOffset(reg_info);
    assert(offset < GetFPRSize());
    src = (uint8_t *)GetFPRBuffer() + offset;
  } else
    return Status::FromErrorString(
        "failed - register wasn't recognized to be a GPR or an FPR, "
        "write strategy unknown");

  reg_value.SetFromMemoryData(*reg_info, src, reg_info->byte_size,
                              eByteOrderLittle, error);

  return error;
}

Status NativeRegisterContextLinux_loongarch64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  Status error;

  if (!reg_info)
    return Status::FromErrorString("reg_info NULL");

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (reg == LLDB_INVALID_REGNUM)
    return Status::FromErrorStringWithFormat(
        "no lldb regnum for %s",
        reg_info->name != nullptr ? reg_info->name : "<unknown register>");

  uint8_t *dst = nullptr;
  uint32_t offset = LLDB_INVALID_INDEX32;

  if (IsGPR(reg)) {
    error = ReadGPR();
    if (error.Fail())
      return error;

    assert(reg_info->byte_offset < GetGPRSize());
    dst = (uint8_t *)GetGPRBuffer() + reg_info->byte_offset;
    ::memcpy(dst, reg_value.GetBytes(), reg_info->byte_size);

    return WriteGPR();
  } else if (IsFPR(reg)) {
    error = ReadFPR();
    if (error.Fail())
      return error;

    offset = CalculateFprOffset(reg_info);
    assert(offset < GetFPRSize());
    dst = (uint8_t *)GetFPRBuffer() + offset;
    ::memcpy(dst, reg_value.GetBytes(), reg_info->byte_size);

    return WriteFPR();
  }

  return Status::FromErrorString("Failed to write register value");
}

Status NativeRegisterContextLinux_loongarch64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(REG_CONTEXT_SIZE, 0));

  error = ReadGPR();
  if (error.Fail())
    return error;

  error = ReadFPR();
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, GetGPRBuffer(), GetGPRSize());
  dst += GetGPRSize();
  ::memcpy(dst, GetFPRBuffer(), GetFPRSize());

  return error;
}

Status NativeRegisterContextLinux_loongarch64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_loongarch64::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != REG_CONTEXT_SIZE) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_loongarch64::%s data_sp contained "
        "mismatched data size, expected %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, REG_CONTEXT_SIZE, data_sp->GetByteSize());
    return error;
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_loongarch64::%s "
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

bool NativeRegisterContextLinux_loongarch64::IsGPR(unsigned reg) const {
  return GetRegisterInfo().GetRegisterSetFromRegisterIndex(reg) ==
         RegisterInfoPOSIX_loongarch64::GPRegSet;
}

bool NativeRegisterContextLinux_loongarch64::IsFPR(unsigned reg) const {
  return GetRegisterInfo().GetRegisterSetFromRegisterIndex(reg) ==
         RegisterInfoPOSIX_loongarch64::FPRegSet;
}

Status NativeRegisterContextLinux_loongarch64::ReadGPR() {
  Status error;

  if (m_gpr_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetGPRBuffer();
  ioVec.iov_len = GetGPRSize();

  error = ReadRegisterSet(&ioVec, GetGPRSize(), NT_PRSTATUS);

  if (error.Success())
    m_gpr_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_loongarch64::WriteGPR() {
  Status error = ReadGPR();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetGPRBuffer();
  ioVec.iov_len = GetGPRSize();

  m_gpr_is_valid = false;

  return WriteRegisterSet(&ioVec, GetGPRSize(), NT_PRSTATUS);
}

Status NativeRegisterContextLinux_loongarch64::ReadFPR() {
  Status error;

  if (m_fpu_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetFPRBuffer();
  ioVec.iov_len = GetFPRSize();

  error = ReadRegisterSet(&ioVec, GetFPRSize(), NT_FPREGSET);

  if (error.Success())
    m_fpu_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_loongarch64::WriteFPR() {
  Status error = ReadFPR();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetFPRBuffer();
  ioVec.iov_len = GetFPRSize();

  m_fpu_is_valid = false;

  return WriteRegisterSet(&ioVec, GetFPRSize(), NT_FPREGSET);
}

void NativeRegisterContextLinux_loongarch64::InvalidateAllRegisters() {
  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
}

uint32_t NativeRegisterContextLinux_loongarch64::CalculateFprOffset(
    const RegisterInfo *reg_info) const {
  return reg_info->byte_offset - GetGPRSize();
}

std::vector<uint32_t>
NativeRegisterContextLinux_loongarch64::GetExpeditedRegisters(
    ExpeditedRegs expType) const {
  std::vector<uint32_t> expedited_reg_nums =
      NativeRegisterContext::GetExpeditedRegisters(expType);

  return expedited_reg_nums;
}

uint32_t
NativeRegisterContextLinux_loongarch64::NumSupportedHardwareBreakpoints() {
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

uint32_t
NativeRegisterContextLinux_loongarch64::SetHardwareBreakpoint(lldb::addr_t addr,
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
  m_hbp_regs[bp_index].control = g_enable_bit;

  // PTRACE call to set corresponding hardware breakpoint register.
  error = WriteHardwareDebugRegs(eDREGTypeBREAK);

  if (error.Fail()) {
    m_hbp_regs[bp_index].address = 0;
    m_hbp_regs[bp_index].control = 0;

    LLDB_LOG(log, "unable to set breakpoint: failed to write debug registers");
    return LLDB_INVALID_INDEX32;
  }

  return bp_index;
}
bool NativeRegisterContextLinux_loongarch64::ClearHardwareBreakpoint(
    uint32_t hw_idx) {
  Log *log = GetLog(LLDBLog::Breakpoints);
  LLDB_LOG(log, "hw_idx: {0}", hw_idx);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();
  if (error.Fail()) {
    LLDB_LOG(log, "unable to clear breakpoint: failed to read debug registers");
    return false;
  }

  if (hw_idx >= m_max_hbp_supported)
    return false;

  // Create a backup we can revert to in case of failure.
  lldb::addr_t tempAddr = m_hbp_regs[hw_idx].address;
  uint32_t tempControl = m_hbp_regs[hw_idx].control;

  m_hbp_regs[hw_idx].control = 0;
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
Status NativeRegisterContextLinux_loongarch64::GetHardwareBreakHitIndex(
    uint32_t &bp_index, lldb::addr_t trap_addr) {
  Log *log = GetLog(LLDBLog::Breakpoints);

  LLDB_LOGF(log, "NativeRegisterContextLinux_loongarch64::%s()", __FUNCTION__);

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
Status NativeRegisterContextLinux_loongarch64::ClearAllHardwareBreakpoints() {
  Log *log = GetLog(LLDBLog::Breakpoints);

  LLDB_LOGF(log, "NativeRegisterContextLinux_loongarch64::%s()", __FUNCTION__);

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
    m_hbp_regs[i].control = 0;
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
bool NativeRegisterContextLinux_loongarch64::BreakpointIsEnabled(
    uint32_t bp_index) {
  Log *log = GetLog(LLDBLog::Breakpoints);
  LLDB_LOG(log, "bp_index: {0}", bp_index);
  return ((m_hbp_regs[bp_index].control & g_enable_bit) != 0);
}

uint32_t
NativeRegisterContextLinux_loongarch64::NumSupportedHardwareWatchpoints() {
  Log *log = GetLog(LLDBLog::Watchpoints);
  Status error = ReadHardwareDebugInfo();
  if (error.Fail()) {
    LLDB_LOG(log, "failed to read debug registers");
    return 0;
  }

  return m_max_hwp_supported;
}

uint32_t NativeRegisterContextLinux_loongarch64::SetHardwareWatchpoint(
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
  case eWatchpointKindWrite:
    watch_flags = 2;
    break;
  case eWatchpointKindRead:
    watch_flags = 1;
    break;
  case (eWatchpointKindRead | eWatchpointKindWrite):
    break;
  default:
    return LLDB_INVALID_INDEX32;
  }

  // Check if size has a valid hardware watchpoint length.
  if (size != 1 && size != 2 && size != 4 && size != 8)
    return LLDB_INVALID_INDEX32;

  // Setup control value
  control_value = g_enable_bit | GetSizeBits(size);
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
  m_hwp_regs[wp_index].address = addr;
  m_hwp_regs[wp_index].control = control_value;

  // PTRACE call to set corresponding watchpoint register.
  error = WriteHardwareDebugRegs(eDREGTypeWATCH);

  if (error.Fail()) {
    m_hwp_regs[wp_index].address = 0;
    m_hwp_regs[wp_index].control = 0;

    LLDB_LOG(log, "unable to set watchpoint: failed to write debug registers");
    return LLDB_INVALID_INDEX32;
  }

  return wp_index;
}

bool NativeRegisterContextLinux_loongarch64::ClearHardwareWatchpoint(
    uint32_t wp_index) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();

  if (error.Fail()) {
    LLDB_LOG(log, "unable to clear watchpoint: failed to read debug registers");
    return false;
  }

  if (wp_index >= m_max_hwp_supported)
    return false;

  // Create a backup we can revert to in case of failure.
  lldb::addr_t tempAddr = m_hwp_regs[wp_index].address;
  uint32_t tempControl = m_hwp_regs[wp_index].control;

  // Update watchpoint in local cache
  m_hwp_regs[wp_index].control = 0;
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

Status NativeRegisterContextLinux_loongarch64::ClearAllHardwareWatchpoints() {
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

uint32_t
NativeRegisterContextLinux_loongarch64::GetWatchpointSize(uint32_t wp_index) {
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

bool NativeRegisterContextLinux_loongarch64::WatchpointIsEnabled(
    uint32_t wp_index) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);
  return ((m_hwp_regs[wp_index].control & g_enable_bit) != 0);
}

Status NativeRegisterContextLinux_loongarch64::GetWatchpointHitIndex(
    uint32_t &wp_index, lldb::addr_t trap_addr) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}, trap_addr: {1:x}", wp_index, trap_addr);

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();
  if (error.Fail())
    return error;

  uint32_t watch_size;
  lldb::addr_t watch_addr;

  for (wp_index = 0; wp_index < m_max_hwp_supported; ++wp_index) {
    watch_size = GetWatchpointSize(wp_index);
    watch_addr = m_hwp_regs[wp_index].address;

    if (WatchpointIsEnabled(wp_index) && trap_addr >= watch_addr &&
        trap_addr <= watch_addr + watch_size) {
      m_hwp_regs[wp_index].hit_addr = trap_addr;
      return Status();
    }
  }

  wp_index = LLDB_INVALID_INDEX32;
  return Status();
}

lldb::addr_t NativeRegisterContextLinux_loongarch64::GetWatchpointAddress(
    uint32_t wp_index) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  if (wp_index >= m_max_hwp_supported)
    return LLDB_INVALID_ADDRESS;

  if (WatchpointIsEnabled(wp_index))
    return m_hwp_regs[wp_index].address;

  return LLDB_INVALID_ADDRESS;
}

lldb::addr_t NativeRegisterContextLinux_loongarch64::GetWatchpointHitAddress(
    uint32_t wp_index) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  if (wp_index >= m_max_hwp_supported)
    return LLDB_INVALID_ADDRESS;

  if (WatchpointIsEnabled(wp_index))
    return m_hwp_regs[wp_index].hit_addr;

  return LLDB_INVALID_ADDRESS;
}

Status NativeRegisterContextLinux_loongarch64::ReadHardwareDebugInfo() {
  if (!m_refresh_hwdebug_info)
    return Status();

  ::pid_t tid = m_thread.GetID();

  int regset = NT_LOONGARCH_HW_WATCH;
  struct iovec ioVec;
  struct user_watch_state dreg_state;
  Status error;

  ioVec.iov_base = &dreg_state;
  ioVec.iov_len = sizeof(dreg_state);
  error = NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET, tid, &regset,
                                            &ioVec, ioVec.iov_len);
  if (error.Fail())
    return error;

  m_max_hwp_supported = dreg_state.dbg_info & 0x3f;

  regset = NT_LOONGARCH_HW_BREAK;
  error = NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET, tid, &regset,
                                            &ioVec, ioVec.iov_len);
  if (error.Fail())
    return error;

  m_max_hbp_supported = dreg_state.dbg_info & 0x3f;

  m_refresh_hwdebug_info = false;

  return error;
}

Status NativeRegisterContextLinux_loongarch64::WriteHardwareDebugRegs(
    DREGType hwbType) {
  struct iovec ioVec;
  struct user_watch_state dreg_state;
  int regset;

  memset(&dreg_state, 0, sizeof(dreg_state));
  ioVec.iov_base = &dreg_state;

  switch (hwbType) {
  case eDREGTypeWATCH:
    regset = NT_LOONGARCH_HW_WATCH;
    ioVec.iov_len = sizeof(dreg_state.dbg_info) +
                    (sizeof(dreg_state.dbg_regs[0]) * m_max_hwp_supported);

    for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
      dreg_state.dbg_regs[i].addr = m_hwp_regs[i].address;
      dreg_state.dbg_regs[i].ctrl = m_hwp_regs[i].control;
    }
    break;
  case eDREGTypeBREAK:
    regset = NT_LOONGARCH_HW_BREAK;
    ioVec.iov_len = sizeof(dreg_state.dbg_info) +
                    (sizeof(dreg_state.dbg_regs[0]) * m_max_hbp_supported);

    for (uint32_t i = 0; i < m_max_hbp_supported; i++) {
      dreg_state.dbg_regs[i].addr = m_hbp_regs[i].address;
      dreg_state.dbg_regs[i].ctrl = m_hbp_regs[i].control;
    }
    break;
  }

  return NativeProcessLinux::PtraceWrapper(PTRACE_SETREGSET, m_thread.GetID(),
                                           &regset, &ioVec, ioVec.iov_len);
}
#endif // defined(__loongarch__) && __loongarch_grlen == 64
