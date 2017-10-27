//===-- NativeRegisterContextLinux_ppc64le.cpp ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This implementation is related to the OpenPOWER ABI for Power Architecture
// 64-bit ELF V2 ABI

#if defined(__powerpc64__)

#include "NativeRegisterContextLinux_ppc64le.h"

#include "lldb/Core/RegisterValue.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/Procfs.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_ppc64le.h"

// System includes - They have to be included after framework includes because
// they define some
// macros which collide with variable names in other modules
#include <sys/socket.h>
#include <elf.h>
#include <asm/ptrace.h>

#define REG_CONTEXT_SIZE GetGPRSize()

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

static const uint32_t g_gpr_regnums_ppc64le[] = {
    gpr_r0_ppc64le,   gpr_r1_ppc64le,  gpr_r2_ppc64le,     gpr_r3_ppc64le,
    gpr_r4_ppc64le,   gpr_r5_ppc64le,  gpr_r6_ppc64le,     gpr_r7_ppc64le,
    gpr_r8_ppc64le,   gpr_r9_ppc64le,  gpr_r10_ppc64le,    gpr_r11_ppc64le,
    gpr_r12_ppc64le,  gpr_r13_ppc64le, gpr_r14_ppc64le,    gpr_r15_ppc64le,
    gpr_r16_ppc64le,  gpr_r17_ppc64le, gpr_r18_ppc64le,    gpr_r19_ppc64le,
    gpr_r20_ppc64le,  gpr_r21_ppc64le, gpr_r22_ppc64le,    gpr_r23_ppc64le,
    gpr_r24_ppc64le,  gpr_r25_ppc64le, gpr_r26_ppc64le,    gpr_r27_ppc64le,
    gpr_r28_ppc64le,  gpr_r29_ppc64le, gpr_r30_ppc64le,    gpr_r31_ppc64le,
    gpr_pc_ppc64le,   gpr_msr_ppc64le, gpr_origr3_ppc64le, gpr_ctr_ppc64le,
    gpr_lr_ppc64le,   gpr_xer_ppc64le, gpr_cr_ppc64le,     gpr_softe_ppc64le,
    gpr_trap_ppc64le,
};

namespace {
// Number of register sets provided by this context.
enum { k_num_register_sets = 1 };
}

static const RegisterSet g_reg_sets_ppc64le[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", k_num_gpr_registers_ppc64le,
     g_gpr_regnums_ppc64le},
};

NativeRegisterContextLinux *
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread,
    uint32_t concrete_frame_idx) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::ppc64le:
    return new NativeRegisterContextLinux_ppc64le(target_arch, native_thread,
                                              concrete_frame_idx);
  default:
    llvm_unreachable("have no register context for architecture");
  }
}

NativeRegisterContextLinux_ppc64le::NativeRegisterContextLinux_ppc64le(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread,
    uint32_t concrete_frame_idx)
    : NativeRegisterContextLinux(native_thread, concrete_frame_idx,
                                 new RegisterInfoPOSIX_ppc64le(target_arch)) {
  if (target_arch.GetMachine() != llvm::Triple::ppc64le) {
    llvm_unreachable("Unhandled target architecture.");
  }

  ::memset(&m_gpr_ppc64le, 0, sizeof(m_gpr_ppc64le));
  ::memset(&m_hwp_regs, 0, sizeof(m_hwp_regs));
}

uint32_t NativeRegisterContextLinux_ppc64le::GetRegisterSetCount() const {
  return k_num_register_sets;
}

const RegisterSet *
NativeRegisterContextLinux_ppc64le::GetRegisterSet(uint32_t set_index) const {
  if (set_index < k_num_register_sets)
    return &g_reg_sets_ppc64le[set_index];

  return nullptr;
}

uint32_t NativeRegisterContextLinux_ppc64le::GetUserRegisterCount() const {
  uint32_t count = 0;
  for (uint32_t set_index = 0; set_index < k_num_register_sets; ++set_index)
    count += g_reg_sets_ppc64le[set_index].num_registers;
  return count;
}

Status NativeRegisterContextLinux_ppc64le::ReadRegister(
    const RegisterInfo *reg_info, RegisterValue &reg_value) {
  Status error;

  if (!reg_info) {
    error.SetErrorString("reg_info NULL");
    return error;
  }

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (IsGPR(reg)) {
    error = ReadGPR();
    if (error.Fail())
      return error;

    uint8_t *src = (uint8_t *) &m_gpr_ppc64le + reg_info->byte_offset;
    reg_value.SetFromMemoryData(reg_info, src, reg_info->byte_size,
                                eByteOrderLittle, error);
  } else {
    return Status("failed - register wasn't recognized to be a GPR, "
                  "read strategy unknown");
  }

  return error;
}

Status NativeRegisterContextLinux_ppc64le::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  Status error;
  if (!reg_info)
    return Status("reg_info NULL");

  const uint32_t reg_index = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg_index == LLDB_INVALID_REGNUM)
    return Status("no lldb regnum for %s", reg_info && reg_info->name
                                               ? reg_info->name
                                               : "<unknown register>");

  if (IsGPR(reg_index)) {
      error = ReadGPR();
      if (error.Fail())
        return error;

      uint8_t *dst = (uint8_t *) &m_gpr_ppc64le + reg_info->byte_offset;
      ::memcpy(dst, reg_value.GetBytes(), reg_value.GetByteSize());

      error = WriteGPR();
      if (error.Fail())
        return error;

      return Status();
  }

  return Status("failed - register wasn't recognized to be a GPR, "
                "write strategy unknown");
}

Status NativeRegisterContextLinux_ppc64le::ReadAllRegisterValues(
    lldb::DataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(REG_CONTEXT_SIZE, 0));
  if (!data_sp)
    return Status("failed to allocate DataBufferHeap instance of size %" PRIu64,
                  REG_CONTEXT_SIZE);

  error = ReadGPR();
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  if (dst == nullptr) {
    error.SetErrorStringWithFormat("DataBufferHeap instance of size %" PRIu64
                                   " returned a null pointer",
                                   REG_CONTEXT_SIZE);
    return error;
  }

  ::memcpy(dst, &m_gpr_ppc64le, GetGPRSize());

  return error;
}

Status NativeRegisterContextLinux_ppc64le::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error.SetErrorStringWithFormat(
        "NativeRegisterContextLinux_ppc64le::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != REG_CONTEXT_SIZE) {
    error.SetErrorStringWithFormat(
        "NativeRegisterContextLinux_ppc64le::%s data_sp contained mismatched "
        "data size, expected %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, REG_CONTEXT_SIZE, data_sp->GetByteSize());
    return error;
  }

  uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error.SetErrorStringWithFormat("NativeRegisterContextLinux_ppc64le::%s "
                                   "DataBuffer::GetBytes() returned a null "
                                   "pointer",
                                   __FUNCTION__);
    return error;
  }

  ::memcpy(&m_gpr_ppc64le, src, GetGPRSize());
  error = WriteGPR();

  return error;
}

bool NativeRegisterContextLinux_ppc64le::IsGPR(unsigned reg) const {
  return reg <= k_last_gpr_ppc64le; // GPR's come first.
}

Status NativeRegisterContextLinux_ppc64le::DoReadGPR(
    void *buf, size_t buf_size) {
  int regset = NT_PRSTATUS;
  return NativeProcessLinux::PtraceWrapper(PTRACE_GETREGS, m_thread.GetID(),
                                           &regset, buf, buf_size);
}

Status NativeRegisterContextLinux_ppc64le::DoWriteGPR(
    void *buf, size_t buf_size) {
  int regset = NT_PRSTATUS;
  return NativeProcessLinux::PtraceWrapper(PTRACE_SETREGS, m_thread.GetID(),
                                           &regset, buf, buf_size);
}

uint32_t NativeRegisterContextLinux_ppc64le::NumSupportedHardwareWatchpoints() {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_WATCHPOINTS));

  // Read hardware breakpoint and watchpoint information.
  Status error = ReadHardwareDebugInfo();

  if (error.Fail())
    return 0;

  LLDB_LOG(log, "{0}", m_max_hwp_supported);
  return m_max_hwp_supported;
}

uint32_t NativeRegisterContextLinux_ppc64le::SetHardwareWatchpoint(
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
  uint32_t rw_mode = 0;

  // Check if we are setting watchpoint other than read/write/access
  // Update watchpoint flag to match ppc64le write-read bit configuration.
  switch (watch_flags) {
  case eWatchpointKindWrite:
    rw_mode = PPC_BREAKPOINT_TRIGGER_WRITE;
    watch_flags = 2;
    break;
  case eWatchpointKindRead:
    rw_mode = PPC_BREAKPOINT_TRIGGER_READ;
    watch_flags = 1;
    break;
  case (eWatchpointKindRead | eWatchpointKindWrite):
    rw_mode = PPC_BREAKPOINT_TRIGGER_RW;
    break;
  default:
    return LLDB_INVALID_INDEX32;
  }

  // Check if size has a valid hardware watchpoint length.
  if (size != 1 && size != 2 && size != 4 && size != 8)
    return LLDB_INVALID_INDEX32;

  // Check 8-byte alignment for hardware watchpoint target address.
  // Below is a hack to recalculate address and size in order to
  // make sure we can watch non 8-byte alligned addresses as well.
  if (addr & 0x07) {

    addr_t begin = llvm::alignDown(addr, 8);
    addr_t end = llvm::alignTo(addr + size, 8);
    size = llvm::PowerOf2Ceil(end - begin);

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
  m_hwp_regs[wp_index].mode = rw_mode;

  // PTRACE call to set corresponding watchpoint register.
  error = WriteHardwareDebugRegs();

  if (error.Fail()) {
    m_hwp_regs[wp_index].address = 0;
    m_hwp_regs[wp_index].control &= llvm::maskTrailingZeros<uint32_t>(1);

    return LLDB_INVALID_INDEX32;
  }

  return wp_index;
}

bool NativeRegisterContextLinux_ppc64le::ClearHardwareWatchpoint(
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
  long *tempSlot = reinterpret_cast<long *>(m_hwp_regs[wp_index].slot);

  // Update watchpoint in local cache
  m_hwp_regs[wp_index].control &= llvm::maskTrailingZeros<uint32_t>(1);
  m_hwp_regs[wp_index].address = 0;
  m_hwp_regs[wp_index].slot = 0;
  m_hwp_regs[wp_index].mode = 0;

  // Ptrace call to update hardware debug registers
  error = NativeProcessLinux::PtraceWrapper(PPC_PTRACE_DELHWDEBUG,
                                            m_thread.GetID(), 0, tempSlot);

  if (error.Fail()) {
    m_hwp_regs[wp_index].control = tempControl;
    m_hwp_regs[wp_index].address = tempAddr;
    m_hwp_regs[wp_index].slot = reinterpret_cast<long>(tempSlot);

    return false;
  }

  return true;
}

uint32_t
NativeRegisterContextLinux_ppc64le::GetWatchpointSize(uint32_t wp_index) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_WATCHPOINTS));
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  unsigned control = (m_hwp_regs[wp_index].control >> 5) & 0xff;
  assert(llvm::isPowerOf2_32(control + 1));
  return llvm::countPopulation(control);
}

bool NativeRegisterContextLinux_ppc64le::WatchpointIsEnabled(
    uint32_t wp_index) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_WATCHPOINTS));
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  return !!((m_hwp_regs[wp_index].control & 0x1) == 0x1);
}

Status NativeRegisterContextLinux_ppc64le::GetWatchpointHitIndex(
    uint32_t &wp_index, lldb::addr_t trap_addr) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_WATCHPOINTS));
  LLDB_LOG(log, "wp_index: {0}, trap_addr: {1:x}", wp_index, trap_addr);

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

lldb::addr_t
NativeRegisterContextLinux_ppc64le::GetWatchpointAddress(uint32_t wp_index) {
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
NativeRegisterContextLinux_ppc64le::GetWatchpointHitAddress(uint32_t wp_index) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_WATCHPOINTS));
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  if (wp_index >= m_max_hwp_supported)
    return LLDB_INVALID_ADDRESS;

  if (WatchpointIsEnabled(wp_index))
    return m_hwp_regs[wp_index].hit_addr;

  return LLDB_INVALID_ADDRESS;
}

Status NativeRegisterContextLinux_ppc64le::ReadHardwareDebugInfo() {
  if (!m_refresh_hwdebug_info) {
    return Status();
  }

  ::pid_t tid = m_thread.GetID();

  struct ppc_debug_info hwdebug_info;
  Status error;

  error = NativeProcessLinux::PtraceWrapper(
      PPC_PTRACE_GETHWDBGINFO, tid, 0, &hwdebug_info, sizeof(hwdebug_info));

  if (error.Fail())
    return error;

  m_max_hwp_supported = hwdebug_info.num_data_bps;
  m_max_hbp_supported = hwdebug_info.num_instruction_bps;
  m_refresh_hwdebug_info = false;

  return error;
}

Status NativeRegisterContextLinux_ppc64le::WriteHardwareDebugRegs() {
  struct ppc_hw_breakpoint reg_state;
  Status error;
  long ret;

  for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
    reg_state.addr = m_hwp_regs[i].address;
    reg_state.trigger_type = m_hwp_regs[i].mode;
    reg_state.version = 1;
    reg_state.addr_mode = PPC_BREAKPOINT_MODE_EXACT;
    reg_state.condition_mode = PPC_BREAKPOINT_CONDITION_NONE;
    reg_state.addr2 = 0;
    reg_state.condition_value = 0;

    error = NativeProcessLinux::PtraceWrapper(PPC_PTRACE_SETHWDEBUG,
                                              m_thread.GetID(), 0, &reg_state,
                                              sizeof(reg_state), &ret);

    if (error.Fail())
      return error;

    m_hwp_regs[i].slot = ret;
  }

  return error;
}

#endif // defined(__powerpc64__)
