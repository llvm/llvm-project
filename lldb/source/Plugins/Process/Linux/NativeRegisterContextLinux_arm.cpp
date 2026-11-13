//===-- NativeRegisterContextLinux_arm.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)

#include "NativeRegisterContextLinux_arm.h"
#include "lldb/Register/Arch/lldb-arm-register-enums.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_arm.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/Procfs.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Register/Common/RegisterInfo_arm.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"

#if defined(__arm64__) || defined(__aarch64__)
#include "Plugins/Process/Linux/NativeRegisterContextLinux_arm64dbreg.h"
#endif

// System includes - They have to be included after framework includes because
// they define some macros which collide with variable names in other modules.
#include <elf.h>
#include <sys/ptrace.h>
#include <sys/uio.h>

#define REG_CONTEXT_SIZE (GetGPRSize() + sizeof(m_fpr) + sizeof(m_tls))

#ifndef PTRACE_GETVFPREGS
#define PTRACE_GETVFPREGS 27
#define PTRACE_SETVFPREGS 28
#endif
#if defined(__arm__) && !defined(PTRACE_GETHBPREGS)
#define PTRACE_GETHBPREGS 29
#define PTRACE_SETHBPREGS 30
#endif
#if !defined(PTRACE_TYPE_ARG3)
#define PTRACE_TYPE_ARG3 void *
#endif
#if !defined(PTRACE_TYPE_ARG4)
#define PTRACE_TYPE_ARG4 void *
#endif

#ifndef PTRACE_GET_THREAD_AREA
#define PTRACE_GET_THREAD_AREA 22
#endif

namespace reg_ids = lldb_private::registers::arm::id;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

#if defined(__arm__)

std::unique_ptr<NativeRegisterContextLinux>
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(
    const ArchSpec &target_arch, NativeThreadLinux &native_thread) {
  return std::make_unique<NativeRegisterContextLinux_arm>(target_arch,
                                                           native_thread);
}

llvm::Expected<ArchSpec>
NativeRegisterContextLinux::DetermineArchitecture(lldb::tid_t tid) {
  return HostInfo::GetArchitecture();
}

#endif // defined(__arm__)

NativeRegisterContextLinux_arm::NativeRegisterContextLinux_arm(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContext(native_thread, target_arch,
                            CreateLinuxArmRegisterLayout(target_arch, true)),
      NativeRegisterContextLinux(native_thread) {
  assert(target_arch.GetMachine() == llvm::Triple::arm);

  ::memset(&m_fpr, 0, sizeof(m_fpr));
  ::memset(&m_tls, 0, sizeof(m_tls));
  ::memset(&m_gpr_arm, 0, sizeof(m_gpr_arm));

  // 16 is just a maximum value, query hardware for actual watchpoint count
  m_max_hwp_supported = 16;
  m_max_hbp_supported = 16;
  m_refresh_hwdebug_info = true;
}

llvm::Expected<RegisterValue>
NativeRegisterContextLinux_arm::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  auto required = GetLayout().GetRequiredBanks(descriptor->id, false);
  if (!required)
    return required.takeError();
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case 0:
      status = ReadGPR();
      break;
    case 1:
      status = ReadFPR();
      break;
    case 2:
      status = ReadTLS();
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  auto value = ReadBankRegister(
      descriptor->id,
      {{0, {reinterpret_cast<const uint8_t *>(&m_gpr_arm), sizeof(m_gpr_arm)}},
       {1, {reinterpret_cast<const uint8_t *>(&m_fpr), sizeof(m_fpr)}},
       {2, {reinterpret_cast<const uint8_t *>(&m_tls), sizeof(m_tls)}}});
  return value;
}

llvm::Error
NativeRegisterContextLinux_arm::WriteRegister(RegisterID id,
                                              const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  if (IsTLS(descriptor->numbers[lldb::eRegisterKindLLDB]))
    return Status::FromErrorString(
               "writing to a thread pointer register is not implemented")
        .ToError();
  auto required = GetLayout().GetRequiredBanks(descriptor->id, true);
  if (!required)
    return required.takeError();
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case 0:
      status = ReadGPR();
      break;
    case 1:
      status = ReadFPR();
      break;
    case 2:
      status = ReadTLS();
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  RegisterValue write_value = reg_value;
  // Preserve the Linux workaround for an odd PC while already in Thumb mode.
  if (descriptor->numbers[lldb::eRegisterKindLLDB] == reg_ids::gpr_pc &&
      (m_gpr_arm[reg_ids::gpr_cpsr] & 0x20) && !(m_gpr_arm[reg_ids::gpr_pc] & 1) &&
      (reg_value.GetAsUInt32() & 1))
    write_value.SetUInt32(reg_value.GetAsUInt32() & ~1u);
  if (auto error = WriteBankRegister(
          descriptor->id, write_value,
          {{0, {reinterpret_cast<uint8_t *>(&m_gpr_arm), sizeof(m_gpr_arm)}},
           {1, {reinterpret_cast<uint8_t *>(&m_fpr), sizeof(m_fpr)}},
           {2, {reinterpret_cast<uint8_t *>(&m_tls), sizeof(m_tls)}}}))
    return error;
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case 0:
      status = WriteGPR();
      break;
    case 1:
      status = WriteFPR();
      break;
    case 2:
      status = Status::FromErrorString(
          "writing to a thread pointer register is not implemented");
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  return llvm::Error::success();
}

Status NativeRegisterContextLinux_arm::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(REG_CONTEXT_SIZE, 0));
  error = ReadGPR();
  if (error.Fail())
    return error;

  error = ReadFPR();
  if (error.Fail())
    return error;

  error = ReadTLS();
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, &m_gpr_arm, GetGPRSize());
  dst += GetGPRSize();
  ::memcpy(dst, &m_fpr, sizeof(m_fpr));
  dst += sizeof(m_fpr);
  ::memcpy(dst, &m_tls, sizeof(m_tls));

  return error;
}

Status NativeRegisterContextLinux_arm::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_arm::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != REG_CONTEXT_SIZE) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_arm::%s data_sp contained mismatched "
        "data size, expected %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, (uint64_t)REG_CONTEXT_SIZE, data_sp->GetByteSize());
    return error;
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_arm::%s "
        "DataBuffer::GetBytes() returned a null "
        "pointer",
        __FUNCTION__);
    return error;
  }
  ::memcpy(&m_gpr_arm, src, GetBankSize(0));

  error = WriteGPR();
  if (error.Fail())
    return error;

  src += GetBankSize(0);
  ::memcpy(&m_fpr, src, sizeof(m_fpr));

  error = WriteFPR();
  if (error.Fail())
    return error;

  // Note: writing to a thread pointer register is not implemented.

  return error;
}

bool NativeRegisterContextLinux_arm::IsGPR(unsigned reg) const {
  auto registers = GetDescription()->GetRegisters();
  const auto *value = reg < registers.size() ? &registers[reg] : nullptr;
  return value && GetDescription()->IsInGroup(value->id, "gpr");
}

bool NativeRegisterContextLinux_arm::IsFPR(unsigned reg) const {
  auto registers = GetDescription()->GetRegisters();
  const auto *value = reg < registers.size() ? &registers[reg] : nullptr;
  return value && GetDescription()->IsInGroup(value->id, "fpu");
}

bool NativeRegisterContextLinux_arm::IsTLS(unsigned reg) const {
  auto registers = GetDescription()->GetRegisters();
  const auto *value = reg < registers.size() ? &registers[reg] : nullptr;
  return value && GetDescription()->IsInGroup(value->id, "tls");
}

llvm::Error NativeRegisterContextLinux_arm::ReadHardwareDebugInfo() {
  if (!m_refresh_hwdebug_info)
    return llvm::Error::success();

#ifdef __arm__
  unsigned int cap_val;
  Status error = NativeProcessLinux::PtraceWrapper(
      PTRACE_GETHBPREGS, m_thread.GetID(), nullptr, &cap_val,
      sizeof(unsigned int));

  if (error.Fail())
    return error.ToError();

  m_max_hwp_supported = (cap_val >> 8) & 0xff;
  m_max_hbp_supported = cap_val & 0xff;
  m_refresh_hwdebug_info = false;

  return error.ToError();
#else  // __aarch64__
  return arm64::ReadHardwareDebugInfo(m_thread.GetID(), m_max_hwp_supported,
                                      m_max_hbp_supported)
      .ToError();
#endif // ifdef __arm__
}

llvm::Error
NativeRegisterContextLinux_arm::WriteHardwareDebugRegs(DREGType hwbType) {
#ifdef __arm__
  uint32_t max_index = m_max_hbp_supported;
  if (hwbType == eDREGTypeWATCH)
    max_index = m_max_hwp_supported;

  for (uint32_t idx = 0; idx < max_index; ++idx)
    if (auto error = WriteHardwareDebugReg(hwbType, idx))
      return error;

  return llvm::Error::success();
#else  // __aarch64__
  uint32_t max_supported =
      (hwbType == NativeRegisterContextDBReg::eDREGTypeWATCH)
          ? m_max_hwp_supported
          : m_max_hbp_supported;
  auto &regs = (hwbType == NativeRegisterContextDBReg::eDREGTypeWATCH)
                   ? m_hwp_regs
                   : m_hbp_regs;
  return arm64::WriteHardwareDebugRegs(hwbType, m_thread.GetID(), max_supported,
                                       regs)
      .ToError();
#endif // ifdef __arm__
}

#ifdef __arm__
llvm::Error
NativeRegisterContextLinux_arm::WriteHardwareDebugReg(DREGType hwbType,
                                                      int hwb_index) {
  Status error;
  lldb::addr_t *addr_buf;
  uint32_t *ctrl_buf;
  int addr_idx = (hwb_index << 1) + 1;
  int ctrl_idx = addr_idx + 1;

  if (hwbType == NativeRegisterContextDBReg::eDREGTypeWATCH) {
    addr_idx *= -1;
    addr_buf = &m_hwp_regs[hwb_index].address;
    ctrl_idx *= -1;
    ctrl_buf = &m_hwp_regs[hwb_index].control;
  } else {
    addr_buf = &m_hbp_regs[hwb_index].address;
    ctrl_buf = &m_hbp_regs[hwb_index].control;
  }

  error = NativeProcessLinux::PtraceWrapper(
      PTRACE_SETHBPREGS, m_thread.GetID(), (PTRACE_TYPE_ARG3)(intptr_t)addr_idx,
      addr_buf, sizeof(unsigned int));

  if (error.Fail())
    return error.ToError();

  error = NativeProcessLinux::PtraceWrapper(
      PTRACE_SETHBPREGS, m_thread.GetID(), (PTRACE_TYPE_ARG3)(intptr_t)ctrl_idx,
      ctrl_buf, sizeof(unsigned int));

  return error.ToError();
}
#endif // ifdef __arm__

Status NativeRegisterContextLinux_arm::ReadGPR() {
#ifdef __arm__
  return NativeRegisterContextLinux::ReadGPR();
#else  // __aarch64__
  struct iovec ioVec;
  ioVec.iov_base = GetGPRBuffer();
  ioVec.iov_len = GetGPRSize();

  return ReadRegisterSet(&ioVec, GetGPRSize(), NT_PRSTATUS);
#endif // __arm__
}

Status NativeRegisterContextLinux_arm::WriteGPR() {
#ifdef __arm__
  return NativeRegisterContextLinux::WriteGPR();
#else  // __aarch64__
  struct iovec ioVec;
  ioVec.iov_base = GetGPRBuffer();
  ioVec.iov_len = GetGPRSize();

  return WriteRegisterSet(&ioVec, GetGPRSize(), NT_PRSTATUS);
#endif // __arm__
}

Status NativeRegisterContextLinux_arm::ReadFPR() {
#ifdef __arm__
  return NativeProcessLinux::PtraceWrapper(PTRACE_GETVFPREGS, m_thread.GetID(),
                                           nullptr, GetFPRBuffer(),
                                           GetFPRSize());
#else  // __aarch64__
  struct iovec ioVec;
  ioVec.iov_base = GetFPRBuffer();
  ioVec.iov_len = GetFPRSize();

  return ReadRegisterSet(&ioVec, GetFPRSize(), NT_ARM_VFP);
#endif // __arm__
}

Status NativeRegisterContextLinux_arm::WriteFPR() {
#ifdef __arm__
  return NativeProcessLinux::PtraceWrapper(PTRACE_SETVFPREGS, m_thread.GetID(),
                                           nullptr, GetFPRBuffer(),
                                           GetFPRSize());
#else  // __aarch64__
  struct iovec ioVec;
  ioVec.iov_base = GetFPRBuffer();
  ioVec.iov_len = GetFPRSize();

  return WriteRegisterSet(&ioVec, GetFPRSize(), NT_ARM_VFP);
#endif // __arm__
}

Status NativeRegisterContextLinux_arm::ReadTLS() {
#ifdef __arm__
  return NativeProcessLinux::PtraceWrapper(PTRACE_GET_THREAD_AREA,
                                           m_thread.GetID(), nullptr,
                                           GetTLSBuffer(), GetTLSSize());
#else  // __aarch64__
  Status error;

  struct iovec ioVec;
  ioVec.iov_base = GetTLSBuffer();
  ioVec.iov_len = GetTLSSize();

  return ReadRegisterSet(&ioVec, GetTLSSize(), NT_ARM_TLS);
#endif // __arm__
}

#endif // defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
