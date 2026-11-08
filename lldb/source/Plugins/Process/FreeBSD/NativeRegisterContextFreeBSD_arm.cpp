//===-- NativeRegisterContextFreeBSD_arm.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__arm__)

#include "NativeRegisterContextFreeBSD_arm.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_arm.h"

#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "Plugins/Process/FreeBSD/NativeProcessFreeBSD.h"
#include "lldb/Register/Common/RegisterInfo_arm.h"

// clang-format off
#include <sys/param.h>
#include <sys/ptrace.h>
#include <sys/types.h>
// clang-format on

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_freebsd;

NativeRegisterContextFreeBSD *
NativeRegisterContextFreeBSD::CreateHostNativeRegisterContextFreeBSD(
    const ArchSpec &target_arch, NativeThreadFreeBSD &native_thread) {
  return new NativeRegisterContextFreeBSD_arm(target_arch, native_thread);
}

NativeRegisterContextFreeBSD_arm::NativeRegisterContextFreeBSD_arm(
    const ArchSpec &target_arch, NativeThreadFreeBSD &native_thread)
    : NativeRegisterContext(native_thread, target_arch,
                            CreateFreeBSDArmRegisterLayout(target_arch)) {}

Status NativeRegisterContextFreeBSD_arm::ReadRegisterSet(uint32_t set) {
  switch (set) {
  case RegisterInfo_arm::GPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETREGS, m_thread.GetID(),
                                               &m_gpr);
  case RegisterInfo_arm::FPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETVFPREGS, m_thread.GetID(),
                                               &m_fpr);
  }
  llvm_unreachable("NativeRegisterContextFreeBSD_arm::ReadRegisterSet");
}

Status NativeRegisterContextFreeBSD_arm::WriteRegisterSet(uint32_t set) {
  switch (set) {
  case RegisterInfo_arm::GPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETREGS, m_thread.GetID(),
                                               &m_gpr);
  case RegisterInfo_arm::FPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETVFPREGS, m_thread.GetID(),
                                               &m_fpr);
  }
  llvm_unreachable("NativeRegisterContextFreeBSD_arm::WriteRegisterSet");
}

llvm::Expected<RegisterValue>
NativeRegisterContextFreeBSD_arm::ReadRegister(RegisterID id) {
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
      status = ReadRegisterSet(RegisterInfo_arm::GPRegSet);
      break;
    case 1:
      status = ReadRegisterSet(RegisterInfo_arm::FPRegSet);
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  auto value = ReadBankRegister(
      descriptor->id,
      {{0, {reinterpret_cast<const uint8_t *>(&m_gpr), sizeof(m_gpr)}},
       {1, {reinterpret_cast<const uint8_t *>(&m_fpr), sizeof(m_fpr)}}});
  return value;
}

llvm::Error NativeRegisterContextFreeBSD_arm::WriteRegister(
    RegisterID id, const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  auto required = GetLayout().GetRequiredBanks(descriptor->id, true);
  if (!required)
    return required.takeError();
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case 0:
      status = ReadRegisterSet(RegisterInfo_arm::GPRegSet);
      break;
    case 1:
      status = ReadRegisterSet(RegisterInfo_arm::FPRegSet);
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  if (auto error = WriteBankRegister(
          descriptor->id, reg_value,
          {{0, {reinterpret_cast<uint8_t *>(&m_gpr), sizeof(m_gpr)}},
           {1, {reinterpret_cast<uint8_t *>(&m_fpr), sizeof(m_fpr)}}}))
    return error;
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case 0:
      status = WriteRegisterSet(RegisterInfo_arm::GPRegSet);
      break;
    case 1:
      status = WriteRegisterSet(RegisterInfo_arm::FPRegSet);
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  return llvm::Error::success();
}

Status NativeRegisterContextFreeBSD_arm::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error = ReadRegisterSet(RegisterInfo_arm::GPRegSet);
  if (error.Fail())
    return error;
  error = ReadRegisterSet(RegisterInfo_arm::FPRegSet);
  if (error.Fail())
    return error;
  data_sp = std::make_shared<DataBufferHeap>(sizeof(m_gpr) + sizeof(m_fpr), 0);
  std::memcpy(data_sp->GetBytes(), &m_gpr, sizeof(m_gpr));
  std::memcpy(data_sp->GetBytes() + sizeof(m_gpr), &m_fpr, sizeof(m_fpr));
  return Status();
}

Status NativeRegisterContextFreeBSD_arm::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (!data_sp || data_sp->GetByteSize() != sizeof(m_gpr) + sizeof(m_fpr))
    return Status::FromErrorString("invalid private register checkpoint");
  std::memcpy(&m_gpr, data_sp->GetBytes(), sizeof(m_gpr));
  std::memcpy(&m_fpr, data_sp->GetBytes() + sizeof(m_gpr), sizeof(m_fpr));
  Status error = WriteRegisterSet(RegisterInfo_arm::GPRegSet);
  if (error.Fail())
    return error;
  return WriteRegisterSet(RegisterInfo_arm::FPRegSet);
}

llvm::Error NativeRegisterContextFreeBSD_arm::CopyHardwareWatchpointsFrom(
    NativeRegisterContextFreeBSD &source) {
  return llvm::Error::success();
}

#endif // defined (__arm__)
