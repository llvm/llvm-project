//===-- NativeRegisterContextFreeBSD_powerpc.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__powerpc__)

#include <set>

#include "NativeRegisterContextFreeBSD_powerpc.h"

#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "Plugins/Process/FreeBSD/NativeProcessFreeBSD.h"
#include "lldb/Register/Arch/lldb-powerpc-register-enums.h"

// clang-format off
#include <sys/param.h>
#include <sys/ptrace.h>
#include <sys/types.h>
// clang-format on
#include <optional>


namespace reg_ids = lldb_private::registers::powerpc::id;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_freebsd;

static const uint32_t g_gpr_regnums[] = {
    reg_ids::gpr_r0,  reg_ids::gpr_r1,  reg_ids::gpr_r2,  reg_ids::gpr_r3,
    reg_ids::gpr_r4,  reg_ids::gpr_r5,  reg_ids::gpr_r6,  reg_ids::gpr_r7,
    reg_ids::gpr_r8,  reg_ids::gpr_r9,  reg_ids::gpr_r10, reg_ids::gpr_r11,
    reg_ids::gpr_r12, reg_ids::gpr_r13, reg_ids::gpr_r14, reg_ids::gpr_r15,
    reg_ids::gpr_r16, reg_ids::gpr_r17, reg_ids::gpr_r18, reg_ids::gpr_r19,
    reg_ids::gpr_r20, reg_ids::gpr_r21, reg_ids::gpr_r22, reg_ids::gpr_r23,
    reg_ids::gpr_r24, reg_ids::gpr_r25, reg_ids::gpr_r26, reg_ids::gpr_r27,
    reg_ids::gpr_r28, reg_ids::gpr_r29, reg_ids::gpr_r30, reg_ids::gpr_r31,
    reg_ids::gpr_lr,  reg_ids::gpr_cr,  reg_ids::gpr_xer, reg_ids::gpr_ctr,
    reg_ids::gpr_pc,
};

static const uint32_t g_fpr_regnums[] = {
    reg_ids::fpr_f0,    reg_ids::fpr_f1,  reg_ids::fpr_f2,  reg_ids::fpr_f3,
    reg_ids::fpr_f4,    reg_ids::fpr_f5,  reg_ids::fpr_f6,  reg_ids::fpr_f7,
    reg_ids::fpr_f8,    reg_ids::fpr_f9,  reg_ids::fpr_f10, reg_ids::fpr_f11,
    reg_ids::fpr_f12,   reg_ids::fpr_f13, reg_ids::fpr_f14, reg_ids::fpr_f15,
    reg_ids::fpr_f16,   reg_ids::fpr_f17, reg_ids::fpr_f18, reg_ids::fpr_f19,
    reg_ids::fpr_f20,   reg_ids::fpr_f21, reg_ids::fpr_f22, reg_ids::fpr_f23,
    reg_ids::fpr_f24,   reg_ids::fpr_f25, reg_ids::fpr_f26, reg_ids::fpr_f27,
    reg_ids::fpr_f28,   reg_ids::fpr_f29, reg_ids::fpr_f30, reg_ids::fpr_f31,
    reg_ids::fpr_fpscr,
};

// Number of register sets provided by this context.
enum { k_num_register_sets = 2 };

struct RegisterGroupNumbers {
  const char *name;
  const char *short_name;
  std::vector<uint32_t> registers;
};

static const RegisterGroupNumbers g_reg_sets_powerpc[k_num_register_sets] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_regnums,
                             g_gpr_regnums + reg_ids::k_num_gpr_registers)},
    {"Floating Point Registers", "fpr",
     std::vector<RegisterID>(g_fpr_regnums,
                             g_fpr_regnums + reg_ids::k_num_fpr_registers)}};

NativeRegisterContextFreeBSD *
NativeRegisterContextFreeBSD::CreateHostNativeRegisterContextFreeBSD(
    const ArchSpec &target_arch, NativeThreadFreeBSD &native_thread) {
  return new NativeRegisterContextFreeBSD_powerpc(target_arch, native_thread);
}

static RegisterLayout CreateRegisterLayout(const ArchSpec &target_arch) {
  if (HostInfo::GetArchitecture().GetAddressByteSize() == 4) {
    return CreateFreeBSDPowerPC32RegisterLayout(target_arch);
  } else {
    assert((HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
           "Register setting path assumes this is a 64-bit host");
    return CreateFreeBSDPowerPC64RegisterLayout(target_arch);
  }
}

NativeRegisterContextFreeBSD_powerpc::NativeRegisterContextFreeBSD_powerpc(
    const ArchSpec &target_arch, NativeThreadFreeBSD &native_thread)
    : NativeRegisterContext(native_thread, target_arch,
                            CreateRegisterLayout(target_arch)) {
  // Bind native presentation policy to this schema generation once. The
  // private definitions above use the backend's external numbering.
  auto select_group = [&](uint32_t set_index) -> const RegisterGroupNumbers * {
    switch (GetTargetArchitecture().GetMachine()) {
    case llvm::Triple::ppc:
      return &g_reg_sets_powerpc[set_index];
    default:
      llvm_unreachable("Unhandled target architecture.");
    }
  };
  auto previous = GetDescription();
  std::vector<RegisterGroup> groups;
  std::set<RegisterID> visible;
  for (uint32_t index = 0; index < k_num_register_sets; ++index) {
    const auto *input = select_group(index);
    if (!input)
      continue;
    RegisterGroup group{input->name, input->short_name, {}};
    for (uint32_t number : input->registers) {
      const auto *reg = previous->FindByNumber(lldb::eRegisterKindLLDB, number);
      assert(reg && "native presentation names an unknown register");
      if (reg) {
        group.registers.push_back(reg->id);
        visible.insert(reg->id);
      }
    }
    groups.push_back(std::move(group));
  }
  auto description = llvm::cantFail(RegisterDescription::Create(
      {previous->GetRegisters().begin(), previous->GetRegisters().end()},
      std::move(groups),
      {previous->GetOwnedTypes().begin(), previous->GetOwnedTypes().end()}));
  std::vector<RegisterID> order;
  for (RegisterID id : previous->GetUserRegisters())
    if (visible.count(id))
      order.push_back(id);
  auto presentation = llvm::cantFail(description.WithUserRegisters(order));
  SetRegisterLayout(llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(std::move(presentation)),
      GetLayout())));
}

std::optional<NativeRegisterContextFreeBSD_powerpc::RegSetKind>
NativeRegisterContextFreeBSD_powerpc::GetSetForNativeRegNum(
    uint32_t reg_num) const {
  switch (GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::ppc:
    if (reg_num >= reg_ids::k_first_gpr && reg_num <= reg_ids::k_last_gpr)
      return GPRegSet;
    if (reg_num >= reg_ids::k_first_fpr && reg_num <= reg_ids::k_last_fpr)
      return FPRegSet;
    break;
  default:
    llvm_unreachable("Unhandled target architecture.");
  }

  llvm_unreachable("Register does not belong to any register set");
}

Status NativeRegisterContextFreeBSD_powerpc::ReadRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETREGS, m_thread.GetID(),
                                               &m_gpr);
  case FPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETFPREGS, m_thread.GetID(),
                                               &m_fpr);
  }
  llvm_unreachable("NativeRegisterContextFreeBSD_powerpc::ReadRegisterSet");
}

Status NativeRegisterContextFreeBSD_powerpc::WriteRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETREGS, m_thread.GetID(),
                                               &m_gpr);
  case FPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETFPREGS, m_thread.GetID(),
                                               &m_fpr);
  }
  llvm_unreachable("NativeRegisterContextFreeBSD_powerpc::WriteRegisterSet");
}

llvm::Expected<RegisterValue>
NativeRegisterContextFreeBSD_powerpc::ReadRegister(RegisterID id) {
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
      status = ReadRegisterSet(GPRegSet);
      break;
    case 1:
      status = ReadRegisterSet(FPRegSet);
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

llvm::Error NativeRegisterContextFreeBSD_powerpc::WriteRegister(
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
      status = ReadRegisterSet(GPRegSet);
      break;
    case 1:
      status = ReadRegisterSet(FPRegSet);
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
      status = WriteRegisterSet(GPRegSet);
      break;
    case 1:
      status = WriteRegisterSet(FPRegSet);
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  return llvm::Error::success();
}

Status NativeRegisterContextFreeBSD_powerpc::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error = ReadRegisterSet(GPRegSet);
  if (error.Fail())
    return error;
  error = ReadRegisterSet(FPRegSet);
  if (error.Fail())
    return error;
  data_sp = std::make_shared<DataBufferHeap>(sizeof(m_gpr) + sizeof(m_fpr), 0);
  std::memcpy(data_sp->GetBytes(), &m_gpr, sizeof(m_gpr));
  std::memcpy(data_sp->GetBytes() + sizeof(m_gpr), &m_fpr, sizeof(m_fpr));
  return Status();
}

Status NativeRegisterContextFreeBSD_powerpc::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (!data_sp || data_sp->GetByteSize() != sizeof(m_gpr) + sizeof(m_fpr))
    return Status::FromErrorString("invalid private register checkpoint");
  std::memcpy(&m_gpr, data_sp->GetBytes(), sizeof(m_gpr));
  std::memcpy(&m_fpr, data_sp->GetBytes() + sizeof(m_gpr), sizeof(m_fpr));
  Status error = WriteRegisterSet(GPRegSet);
  if (error.Fail())
    return error;
  return WriteRegisterSet(FPRegSet);
}

llvm::Error NativeRegisterContextFreeBSD_powerpc::CopyHardwareWatchpointsFrom(
    NativeRegisterContextFreeBSD &source) {
  return llvm::Error::success();
}

#endif // defined (__powerpc__)
