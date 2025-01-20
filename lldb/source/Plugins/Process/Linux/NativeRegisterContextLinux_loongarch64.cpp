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

#ifndef NT_LOONGARCH_LSX
#define NT_LOONGARCH_LSX 0xa02 /* LoongArch SIMD eXtension registers */
#endif

#ifndef NT_LOONGARCH_LASX
#define NT_LOONGARCH_LASX                                                      \
  0xa03 /* LoongArch Advanced SIMD eXtension registers */
#endif

#define REG_CONTEXT_SIZE                                                       \
  (GetGPRSize() + GetFPRSize() + sizeof(m_lsx) + sizeof(m_lasx))

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

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
  ::memset(&m_lsx, 0, sizeof(m_lsx));
  ::memset(&m_lasx, 0, sizeof(m_lasx));

  ::memset(&m_hwp_regs, 0, sizeof(m_hwp_regs));
  ::memset(&m_hbp_regs, 0, sizeof(m_hbp_regs));

  // Refer to:
  // https://loongson.github.io/LoongArch-Documentation/LoongArch-Vol1-EN.html#control-and-status-registers-related-to-watchpoints
  // 14 is just a maximum value, query hardware for actual watchpoint count.
  m_max_hwp_supported = 14;
  m_max_hbp_supported = 14;
  m_refresh_hwdebug_info = true;

  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
  m_lsx_is_valid = false;
  m_lasx_is_valid = false;
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
  } else if (IsLSX(reg)) {
    error = ReadLSX();
    if (error.Fail())
      return error;

    offset = CalculateLsxOffset(reg_info);
    assert(offset < sizeof(m_lsx));
    src = (uint8_t *)&m_lsx + offset;
  } else if (IsLASX(reg)) {
    error = ReadLASX();
    if (error.Fail())
      return error;

    offset = CalculateLasxOffset(reg_info);
    assert(offset < sizeof(m_lasx));
    src = (uint8_t *)&m_lasx + offset;
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
  } else if (IsLSX(reg)) {
    error = ReadLSX();
    if (error.Fail())
      return error;

    offset = CalculateLsxOffset(reg_info);
    assert(offset < sizeof(m_lsx));
    dst = (uint8_t *)&m_lsx + offset;
    ::memcpy(dst, reg_value.GetBytes(), reg_info->byte_size);

    return WriteLSX();
  } else if (IsLASX(reg)) {
    error = ReadLASX();
    if (error.Fail())
      return error;

    offset = CalculateLasxOffset(reg_info);
    assert(offset < sizeof(m_lasx));
    dst = (uint8_t *)&m_lasx + offset;
    ::memcpy(dst, reg_value.GetBytes(), reg_info->byte_size);

    return WriteLASX();
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

  error = ReadLSX();
  if (error.Fail())
    return error;

  error = ReadLASX();
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, GetGPRBuffer(), GetGPRSize());
  dst += GetGPRSize();
  ::memcpy(dst, GetFPRBuffer(), GetFPRSize());
  dst += GetFPRSize();
  ::memcpy(dst, &m_lsx, sizeof(m_lsx));
  dst += sizeof(m_lsx);
  ::memcpy(dst, &m_lasx, sizeof(m_lasx));

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
  m_fpu_is_valid = true;
  error = WriteFPR();
  if (error.Fail())
    return error;

  // Currently, we assume that LoongArch always support LASX.
  // TODO: check whether LSX/LASX exists.
  src += GetFPRSize();
  ::memcpy(&m_lsx, src, sizeof(m_lsx));
  m_lsx_is_valid = true;
  error = WriteLSX();
  if (error.Fail())
    return error;

  src += sizeof(m_lsx);
  ::memcpy(&m_lasx, src, sizeof(m_lasx));
  m_lasx_is_valid = true;
  error = WriteLASX();
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

bool NativeRegisterContextLinux_loongarch64::IsLSX(unsigned reg) const {
  return GetRegisterInfo().GetRegisterSetFromRegisterIndex(reg) ==
         RegisterInfoPOSIX_loongarch64::LSXRegSet;
}

bool NativeRegisterContextLinux_loongarch64::IsLASX(unsigned reg) const {
  return GetRegisterInfo().GetRegisterSetFromRegisterIndex(reg) ==
         RegisterInfoPOSIX_loongarch64::LASXRegSet;
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
  m_lsx_is_valid = false;
  m_lasx_is_valid = false;

  return WriteRegisterSet(&ioVec, GetFPRSize(), NT_FPREGSET);
}

Status NativeRegisterContextLinux_loongarch64::ReadLSX() {
  Status error;

  if (m_lsx_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = &m_lsx;
  ioVec.iov_len = sizeof(m_lsx);

  error = ReadRegisterSet(&ioVec, sizeof(m_lsx), NT_LOONGARCH_LSX);

  if (error.Success())
    m_lsx_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_loongarch64::WriteLSX() {
  Status error = ReadLSX();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = &m_lsx;
  ioVec.iov_len = sizeof(m_lsx);

  m_fpu_is_valid = false;
  m_lsx_is_valid = false;
  m_lasx_is_valid = false;

  return WriteRegisterSet(&ioVec, sizeof(m_lsx), NT_LOONGARCH_LSX);
}

Status NativeRegisterContextLinux_loongarch64::ReadLASX() {
  Status error;

  if (m_lasx_is_valid)
    return error;

  struct iovec ioVec;
  ioVec.iov_base = &m_lasx;
  ioVec.iov_len = sizeof(m_lasx);

  error = ReadRegisterSet(&ioVec, sizeof(m_lasx), NT_LOONGARCH_LASX);

  if (error.Success())
    m_lasx_is_valid = true;

  return error;
}

Status NativeRegisterContextLinux_loongarch64::WriteLASX() {
  Status error = ReadLASX();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = &m_lasx;
  ioVec.iov_len = sizeof(m_lasx);

  m_fpu_is_valid = false;
  m_lsx_is_valid = false;
  m_lasx_is_valid = false;

  return WriteRegisterSet(&ioVec, sizeof(m_lasx), NT_LOONGARCH_LASX);
}

void NativeRegisterContextLinux_loongarch64::InvalidateAllRegisters() {
  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
  m_lsx_is_valid = false;
  m_lasx_is_valid = false;
}

uint32_t NativeRegisterContextLinux_loongarch64::CalculateFprOffset(
    const RegisterInfo *reg_info) const {
  return reg_info->byte_offset - GetGPRSize();
}

uint32_t NativeRegisterContextLinux_loongarch64::CalculateLsxOffset(
    const RegisterInfo *reg_info) const {
  return reg_info->byte_offset - GetGPRSize() - sizeof(m_fpr);
}

uint32_t NativeRegisterContextLinux_loongarch64::CalculateLasxOffset(
    const RegisterInfo *reg_info) const {
  return reg_info->byte_offset - GetGPRSize() - sizeof(m_fpr) - sizeof(m_lsx);
}

std::vector<uint32_t>
NativeRegisterContextLinux_loongarch64::GetExpeditedRegisters(
    ExpeditedRegs expType) const {
  std::vector<uint32_t> expedited_reg_nums =
      NativeRegisterContext::GetExpeditedRegisters(expType);

  return expedited_reg_nums;
}

llvm::Error NativeRegisterContextLinux_loongarch64::ReadHardwareDebugInfo() {
  if (!m_refresh_hwdebug_info)
    return llvm::Error::success();

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
    return error.ToError();

  m_max_hwp_supported = dreg_state.dbg_info & 0x3f;

  regset = NT_LOONGARCH_HW_BREAK;
  error = NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET, tid, &regset,
                                            &ioVec, ioVec.iov_len);
  if (error.Fail())
    return error.ToError();

  m_max_hbp_supported = dreg_state.dbg_info & 0x3f;

  m_refresh_hwdebug_info = false;

  return llvm::Error::success();
}

llvm::Error NativeRegisterContextLinux_loongarch64::WriteHardwareDebugRegs(
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
                                           &regset, &ioVec, ioVec.iov_len)
      .ToError();
}
#endif // defined(__loongarch__) && __loongarch_grlen == 64
