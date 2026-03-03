//===-- NativeRegisterContextLinux_riscv64.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__riscv) && __riscv_xlen == 64

#include "NativeRegisterContextLinux_riscv64.h"

#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/Procfs.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_riscv64.h"
#include "Plugins/Process/Utility/lldb-riscv-register-enums.h"

// System includes - They have to be included after framework includes because
// they define some macros which collide with variable names in other modules
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>
// NT_PRSTATUS and NT_FPREGSET definition
#include <elf.h>

#ifndef NT_RISCV_VECTOR
#define NT_RISCV_VECTOR 0x901
#endif
#ifndef __NR_riscv_hwprobe
#define __NR_riscv_hwprobe 258
#endif
#ifndef RISCV_HWPROBE_KEY_IMA_EXT_0
#define RISCV_HWPROBE_KEY_IMA_EXT_0 4
#endif
#ifndef RISCV_HWPROBE_IMA_V
#define RISCV_HWPROBE_IMA_V (1 << 2)
#endif

struct HWProbeRISCV {
  int64_t key;
  uint64_t value;
};

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

static uint64_t GetVLENB() {
  struct HWProbeRISCV query = {RISCV_HWPROBE_KEY_IMA_EXT_0, 0};
  if (syscall(__NR_riscv_hwprobe, &query, 1, 0, NULL, 0) != 0)
    return 0;

  if ((query.value & RISCV_HWPROBE_IMA_V) == 0)
    return 0;

  uint64_t vlenb = 0;
  asm volatile("csrr %[vlenb], vlenb" : [vlenb] "=r"(vlenb));
  return vlenb;
}

static RegisterInfoPOSIX_riscv64::VPR CreateVPRBuffer() {
  uint64_t vlenb = GetVLENB();
  if (vlenb > 0)
    return RegisterInfoPOSIX_riscv64::VPR(vlenb);
  return RegisterInfoPOSIX_riscv64::VPR();
}

std::unique_ptr<NativeRegisterContextLinux>
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(
    const ArchSpec &target_arch, NativeThreadLinux &native_thread) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::riscv64: {
    Flags opt_regsets(RegisterInfoPOSIX_riscv64::eRegsetMaskDefault);

    RegisterInfoPOSIX_riscv64::FPR fpr;
    struct iovec ioVec;
    ioVec.iov_base = &fpr;
    ioVec.iov_len = sizeof(fpr);
    unsigned int regset = NT_FPREGSET;

    if (NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET,
                                          native_thread.GetID(), &regset,
                                          &ioVec, sizeof(fpr))
            .Success()) {
      opt_regsets.Set(RegisterInfoPOSIX_riscv64::eRegsetMaskFP);
    }

    uint64_t vlenb = GetVLENB();

    auto register_info_up = std::make_unique<RegisterInfoPOSIX_riscv64>(
        target_arch, opt_regsets, vlenb);
    return std::make_unique<NativeRegisterContextLinux_riscv64>(
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

NativeRegisterContextLinux_riscv64::NativeRegisterContextLinux_riscv64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread,
    std::unique_ptr<RegisterInfoPOSIX_riscv64> register_info_up)
    : NativeRegisterContextRegisterInfo(native_thread,
                                        register_info_up.release()),
      NativeRegisterContextLinux(native_thread), m_vpr(CreateVPRBuffer()) {
  ::memset(&m_fpr, 0, sizeof(m_fpr));
  ::memset(&m_gpr, 0, sizeof(m_gpr));

  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
  m_vpr_is_valid = false;
}

const RegisterInfoPOSIX_riscv64 &
NativeRegisterContextLinux_riscv64::GetRegisterInfo() const {
  return static_cast<const RegisterInfoPOSIX_riscv64 &>(
      NativeRegisterContextRegisterInfo::GetRegisterInfoInterface());
}

uint32_t NativeRegisterContextLinux_riscv64::GetRegisterSetCount() const {
  return GetRegisterInfo().GetRegisterSetCount();
}

const RegisterSet *
NativeRegisterContextLinux_riscv64::GetRegisterSet(uint32_t set_index) const {
  return GetRegisterInfo().GetRegisterSet(set_index);
}

uint32_t NativeRegisterContextLinux_riscv64::GetUserRegisterCount() const {
  uint32_t count = 0;
  for (uint32_t set_index = 0; set_index < GetRegisterSetCount(); ++set_index)
    count += GetRegisterSet(set_index)->num_registers;
  return count;
}

Status
NativeRegisterContextLinux_riscv64::ReadRegister(const RegisterInfo *reg_info,
                                                 RegisterValue &reg_value) {
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

  if (reg == gpr_x0_riscv) {
    reg_value.SetUInt(0, reg_info->byte_size);
    return error;
  }

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
  } else if (IsVPR(reg)) {
    error = ReadVPR();
    if (error.Fail())
      return error;

    offset = reg_info->byte_offset;
    src = static_cast<uint8_t *>(GetVPRBuffer()) + offset;
  } else
    return Status::FromErrorString(
        "failed - register wasn't recognized to be a GPR or an FPR, "
        "write strategy unknown");

  reg_value.SetFromMemoryData(*reg_info, src, reg_info->byte_size,
                              eByteOrderLittle, error);

  return error;
}

Status NativeRegisterContextLinux_riscv64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  Status error;

  if (!reg_info)
    return Status::FromErrorString("reg_info NULL");

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (reg == LLDB_INVALID_REGNUM)
    return Status::FromErrorStringWithFormat(
        "no lldb regnum for %s",
        reg_info->name != nullptr ? reg_info->name : "<unknown register>");

  if (reg == gpr_x0_riscv) {
    // do nothing.
    return error;
  }

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
  } else if (IsVPR(reg)) {
    error = ReadVPR();
    if (error.Fail())
      return error;

    offset = reg_info->byte_offset;
    dst = static_cast<uint8_t *>(GetVPRBuffer()) + offset;
    ::memcpy(dst, reg_value.GetBytes(), reg_info->byte_size);

    return WriteVPR();
  }

  return Status::FromErrorString("Failed to write register value");
}

Status NativeRegisterContextLinux_riscv64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(GetRegContextSize(), 0));

  error = ReadGPR();
  if (error.Fail())
    return error;

  if (GetRegisterInfo().IsFPPresent()) {
    error = ReadFPR();
    if (error.Fail())
      return error;
  }

  if (GetRegisterInfo().IsVPPresent()) {
    error = ReadVPR();
    if (error.Fail())
      return error;
  }

  uint8_t *dst = const_cast<uint8_t *>(data_sp->GetBytes());
  ::memcpy(dst, GetGPRBuffer(), GetGPRSize());
  dst += GetGPRSize();
  if (GetRegisterInfo().IsFPPresent()) {
    ::memcpy(dst, GetFPRBuffer(), GetFPRSize());
    dst += GetFPRSize();
  }
  if (GetRegisterInfo().IsVPPresent())
    ::memcpy(dst, GetVPRBuffer(), GetVPRSize());

  return error;
}

Status NativeRegisterContextLinux_riscv64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_riscv64::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != GetRegContextSize()) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_riscv64::%s data_sp contained mismatched "
        "data size, expected %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, GetRegContextSize(), data_sp->GetByteSize());
    return error;
  }

  uint8_t *src = const_cast<uint8_t *>(data_sp->GetBytes());
  if (src == nullptr) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_riscv64::%s "
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

  if (GetRegisterInfo().IsFPPresent()) {
    ::memcpy(GetFPRBuffer(), src, GetFPRSize());

    error = WriteFPR();
    if (error.Fail())
      return error;

    src += GetFPRSize();
  }

  if (GetRegisterInfo().IsVPPresent()) {
    ::memcpy(GetVPRBuffer(), src, GetVPRSize());

    error = WriteVPR();
    if (error.Fail())
      return error;
  }

  return error;
}

size_t NativeRegisterContextLinux_riscv64::GetRegContextSize() {
  size_t size = GetGPRSize();
  if (GetRegisterInfo().IsFPPresent())
    size += GetFPRSize();
  if (GetRegisterInfo().IsVPPresent())
    size += GetVPRSize();
  return size;
}

bool NativeRegisterContextLinux_riscv64::IsGPR(unsigned reg) const {
  return GetRegisterInfo().GetRegisterSetFromRegisterIndex(reg) ==
         RegisterInfoPOSIX_riscv64::GPRegSet;
}

bool NativeRegisterContextLinux_riscv64::IsFPR(unsigned reg) const {
  return GetRegisterInfo().IsFPReg(reg);
}

bool NativeRegisterContextLinux_riscv64::IsVPR(unsigned reg) const {
  return GetRegisterInfo().IsVPReg(reg);
}

Status NativeRegisterContextLinux_riscv64::ReadGPR() {
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

Status NativeRegisterContextLinux_riscv64::WriteGPR() {
  Status error = ReadGPR();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetGPRBuffer();
  ioVec.iov_len = GetGPRSize();

  m_gpr_is_valid = false;

  return WriteRegisterSet(&ioVec, GetGPRSize(), NT_PRSTATUS);
}

Status NativeRegisterContextLinux_riscv64::ReadFPR() {
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

Status NativeRegisterContextLinux_riscv64::WriteFPR() {
  Status error = ReadFPR();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetFPRBuffer();
  ioVec.iov_len = GetFPRSize();

  m_fpu_is_valid = false;

  return WriteRegisterSet(&ioVec, GetFPRSize(), NT_FPREGSET);
}

Status NativeRegisterContextLinux_riscv64::ReadVPR() {
  if (m_vpr_is_valid)
    return Status();

  struct iovec ioVec;
  ioVec.iov_base = GetVPRBuffer();
  ioVec.iov_len = GetVPRSize();

  Status error = ReadRegisterSet(&ioVec, GetVPRSize(), NT_RISCV_VECTOR);
  if (error.Fail())
    return error;

  // Additionally check the vlenb value. Due to bugs in early versions of
  // RVV support in the Linux kernel, it was possible to obtain an invalid
  // vector register context even if the PTRACE_GETREGSET call succeeded.
  bool is_valid_ctx =
      GetVPRBuffer() &&
      static_cast<RegisterInfoPOSIX_riscv64::VPR::RawVPR *>(GetVPRBuffer())
              ->vlenb > 0;
  if (!is_valid_ctx)
    return Status::FromErrorString("Invalid vector register context");

  m_vpr_is_valid = true;
  return Status();
}

Status NativeRegisterContextLinux_riscv64::WriteVPR() {
  Status error = ReadVPR();
  if (error.Fail())
    return error;

  struct iovec ioVec;
  ioVec.iov_base = GetVPRBuffer();
  ioVec.iov_len = GetVPRSize();

  m_vpr_is_valid = false;

  return WriteRegisterSet(&ioVec, GetVPRSize(), NT_RISCV_VECTOR);
}

void NativeRegisterContextLinux_riscv64::InvalidateAllRegisters() {
  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
  m_vpr_is_valid = false;
}

uint32_t NativeRegisterContextLinux_riscv64::CalculateFprOffset(
    const RegisterInfo *reg_info) const {
  return reg_info->byte_offset - GetGPRSize();
}

std::vector<uint32_t> NativeRegisterContextLinux_riscv64::GetExpeditedRegisters(
    ExpeditedRegs expType) const {
  std::vector<uint32_t> expedited_reg_nums =
      NativeRegisterContext::GetExpeditedRegisters(expType);

  return expedited_reg_nums;
}

#endif // defined (__riscv) && __riscv_xlen == 64
