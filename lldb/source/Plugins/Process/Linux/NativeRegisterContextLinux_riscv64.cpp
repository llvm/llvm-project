//===-- NativeRegisterContextLinux_riscv64.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__riscv) && __riscv_xlen == 64

#include "NativeRegisterContextLinux_riscv64.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/Procfs.h"
#include "lldb/Register/Arch/lldb-riscv-register-enums.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Register/Common/RegisterInfo_riscv64.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"

// System includes - They have to be included after framework includes because
// they define some macros which collide with variable names in other modules
#include <elf.h>
#include <sys/ptrace.h>
#include <sys/uio.h>

#ifndef PTRACE_GETREGSET
#define PTRACE_GETREGSET 0x4204
#endif


namespace reg_ids = lldb_private::registers::riscv::id;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

std::unique_ptr<NativeRegisterContextLinux>
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(
    const ArchSpec &target_arch, NativeThreadLinux &native_thread) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::riscv64: {
    Flags opt_regsets(RegisterInfo_riscv64::eRegsetMaskDefault);

    linux_riscv64::FPR fpr;
    struct iovec ioVec;
    ioVec.iov_base = &fpr;
    ioVec.iov_len = sizeof(fpr);
    unsigned int regset = NT_FPREGSET;

    if (NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET,
                                          native_thread.GetID(), &regset,
                                          &ioVec, sizeof(fpr))
            .Success()) {
      opt_regsets.Set(RegisterInfo_riscv64::eRegsetMaskFP);
    }

    auto register_model =
        CreateLinuxRISCV64RegisterLayout(target_arch, opt_regsets);
    return std::make_unique<NativeRegisterContextLinux_riscv64>(
        target_arch, native_thread, std::move(register_model));
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
    RegisterLayout register_model)
    : NativeRegisterContext(native_thread, target_arch,
                            std::move(register_model)),
      NativeRegisterContextLinux(native_thread) {
  ::memset(&m_fpr, 0, sizeof(m_fpr));
  ::memset(&m_gpr, 0, sizeof(m_gpr));

  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
}

llvm::Expected<RegisterValue>
NativeRegisterContextLinux_riscv64::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg = description ? description->Find(id) : nullptr;
  if (!reg)
    return Status::FromErrorString("unknown register").ToError();
  std::vector<RegisterData> banks;
  if (reg->id != reg_ids::gpr_x0) {
    bool general = IsGPR(reg->numbers[lldb::eRegisterKindLLDB]);
    Status error = general ? ReadGPR() : ReadFPR();
    if (error.Fail())
      return error.ToError();
    banks.push_back({general ? 0u : 1u,
                     {static_cast<const uint8_t *>(general ? GetGPRBuffer()
                                                           : GetFPRBuffer()),
                      general ? GetGPRSize() : GetFPRSize()}});
  }
  auto result = GetLayout().Read(reg->id, banks);
  return result;
}

llvm::Error NativeRegisterContextLinux_riscv64::WriteRegister(
    RegisterID id, const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *reg = description ? description->Find(id) : nullptr;
  if (!reg)
    return Status::FromErrorString("unknown register").ToError();
  if (reg->id == reg_ids::gpr_x0)
    return llvm::Error::success(); // Writes to architectural x0 have no effect.
  bool general = IsGPR(reg->numbers[lldb::eRegisterKindLLDB]);
  Status error = general ? ReadGPR() : ReadFPR();
  if (error.Fail())
    return error.ToError();
  MutableRegisterData bank{
      general ? 0u : 1u,
      {static_cast<uint8_t *>(general ? GetGPRBuffer() : GetFPRBuffer()),
       general ? GetGPRSize() : GetFPRSize()}};
  if (llvm::Error error = GetLayout().Write(reg->id, reg_value, bank))
    return error;
  return (general ? WriteGPR() : WriteFPR()).ToError();
}

Status NativeRegisterContextLinux_riscv64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(GetRegContextSize(), 0));

  error = ReadGPR();
  if (error.Fail())
    return error;

  if (GetDescription()->FindGroup("fpr") != nullptr) {
    error = ReadFPR();
    if (error.Fail())
      return error;
  }

  uint8_t *dst = const_cast<uint8_t *>(data_sp->GetBytes());
  ::memcpy(dst, GetGPRBuffer(), GetGPRSize());
  dst += GetGPRSize();
  if (GetDescription()->FindGroup("fpr") != nullptr)
    ::memcpy(dst, GetFPRBuffer(), GetFPRSize());

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
  ::memcpy(GetGPRBuffer(), src, GetBankSize(0));

  error = WriteGPR();
  if (error.Fail())
    return error;

  src += GetBankSize(0);

  if (GetDescription()->FindGroup("fpr") != nullptr) {
    ::memcpy(GetFPRBuffer(), src, GetFPRSize());

    error = WriteFPR();
    if (error.Fail())
      return error;
  }

  return error;
}

size_t NativeRegisterContextLinux_riscv64::GetRegContextSize() {
  size_t size = GetGPRSize();
  if (GetDescription()->FindGroup("fpr") != nullptr)
    size += GetFPRSize();
  return size;
}

bool NativeRegisterContextLinux_riscv64::IsGPR(unsigned reg) const {
  const auto *value =
      GetDescription()->FindByNumber(lldb::eRegisterKindLLDB, reg);
  return value && GetDescription()->IsInGroup(value->id, "gpr");
}

bool NativeRegisterContextLinux_riscv64::IsFPR(unsigned reg) const {
  const auto *value =
      GetDescription()->FindByNumber(lldb::eRegisterKindLLDB, reg);
  return value && GetDescription()->IsInGroup(value->id, "fpr");
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

void NativeRegisterContextLinux_riscv64::InvalidateAllRegisters() {
  m_gpr_is_valid = false;
  m_fpu_is_valid = false;
}

#endif // defined (__riscv) && __riscv_xlen == 64
