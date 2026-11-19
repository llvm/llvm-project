//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__powerpc__)

#include "NativeRegisterContextFreeBSD_ppc.h"
#include "NativeProcessFreeBSD.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "llvm/ADT/STLExtras.h"
#include <array>
#include <cstring>
#include <set>

// clang-format off
#include <sys/param.h>
#include <sys/ptrace.h>
#include <sys/types.h>
// clang-format on

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_freebsd;

static_assert(sizeof(reg) == 37 * sizeof(reg::lr));
static_assert(offsetof(reg, pc) == 36 * sizeof(reg::lr));
static_assert(sizeof(fpreg) == 33 * sizeof(uint64_t));
static_assert(offsetof(fpreg, fpscr) == 32 * sizeof(uint64_t));
static_assert(sizeof(vmxreg) == 528);
static_assert(offsetof(vmxreg, vrsave) == 520);
static_assert(offsetof(vmxreg, vscr) == 524);

RegisterLayout
NativeRegisterContextFreeBSD_ppc::SelectLayout(RegisterLayout layout,
                                               bool has_vmx, bool has_vsx) {
  auto previous = layout.GetDescription();
  std::vector<RegisterGroup> groups;
  std::set<RegisterID> visible;

  for (const auto &input : previous->GetGroups()) {
    RegisterGroup group{input.name, input.short_name, {}};

    for (RegisterID id : input.registers) {
      const auto *location = layout.Find(id);
      if (!location || !llvm::all_of(location->pieces, [&](const auto &piece) {
            return piece.bank < 2 || (piece.bank == 2 && has_vmx) ||
                   (piece.bank == 3 && has_vsx);
          }))
        continue;

      group.registers.push_back(id);
      visible.insert(id);
    }

    if (!group.registers.empty())
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

  llvm::cantFail(layout.Finalize(
      std::make_shared<RegisterDescription>(std::move(presentation)),
      layout.GetByteOrder()));
  return layout;
}

bool NativeRegisterContextFreeBSD_ppc::HasRegisterSet(RegSetKind set) const {
  return set == GPRegSet || set == FPRegSet ||
         (set == VMXRegSet && m_has_vmx) || (set == VSXRegSet && m_has_vsx);
}

llvm::SmallVector<MutableRegisterData, 4>
NativeRegisterContextFreeBSD_ppc::GetRegisterBanks() {
  const std::array<MutableRegisterData, 4> storage = {
      {{GPRegSet, {reinterpret_cast<uint8_t *>(&m_gpr), sizeof(m_gpr)}},
       {FPRegSet, {reinterpret_cast<uint8_t *>(&m_fpr), sizeof(m_fpr)}},
       {VMXRegSet, {reinterpret_cast<uint8_t *>(&m_vmx), sizeof(m_vmx)}},
       {VSXRegSet, {reinterpret_cast<uint8_t *>(&m_vsx), sizeof(m_vsx)}}}};
  llvm::SmallVector<MutableRegisterData, 4> banks;

  // A PPC32 layout, or PPC64 without VSX, declares no VSX bank. Passing an
  // undeclared bank to RegisterLayout would reject even an ordinary GPR read.
  for (const auto &bank : storage)
    if (GetLayout().FindBank(bank.bank))
      banks.push_back(bank);

  return banks;
}

Status NativeRegisterContextFreeBSD_ppc::ReadRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETREGS, m_thread.GetID(),
                                               &m_gpr);
  case FPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETFPREGS, m_thread.GetID(),
                                               &m_fpr);
  case VMXRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETVRREGS, m_thread.GetID(),
                                               &m_vmx);
  case VSXRegSet:
#if defined(__powerpc64__)
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETVSRREGS, m_thread.GetID(),
                                               &m_vsx);
#else
    return Status::FromErrorString("VSX access requires a PPC64 host");
#endif
  }

  llvm_unreachable("NativeRegisterContextFreeBSD_ppc::ReadRegisterSet");
}

Status NativeRegisterContextFreeBSD_ppc::WriteRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETREGS, m_thread.GetID(),
                                               &m_gpr);
  case FPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETFPREGS, m_thread.GetID(),
                                               &m_fpr);
  case VMXRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETVRREGS, m_thread.GetID(),
                                               &m_vmx);
  case VSXRegSet:
#if defined(__powerpc64__)
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETVSRREGS, m_thread.GetID(),
                                               &m_vsx);
#else
    return Status::FromErrorString("VSX access requires a PPC64 host");
#endif
  }

  llvm_unreachable("NativeRegisterContextFreeBSD_ppc::WriteRegisterSet");
}

llvm::Expected<RegisterValue>
NativeRegisterContextFreeBSD_ppc::ReadRegister(RegisterID id) {
  auto required = GetLayout().GetRequiredBanks(id);
  if (!required)
    return required.takeError();

  for (RegisterBankID bank : *required) {
    auto set = static_cast<RegSetKind>(bank);
    if (!HasRegisterSet(set))
      return llvm::createStringError("unsupported register bank");

    if (Status error = ReadRegisterSet(set); error.Fail())
      return error.ToError();
  }

  auto banks = GetRegisterBanks();
  std::vector<RegisterData> data;

  for (const auto &bank : banks)
    data.push_back({bank.bank, bank.bytes});

  return ReadBankRegister(id, data);
}

llvm::Error
NativeRegisterContextFreeBSD_ppc::WriteRegister(RegisterID id,
                                                const RegisterValue &value) {
  auto required = GetLayout().GetRequiredBanks(id, true);
  if (!required)
    return required.takeError();

  for (RegisterBankID bank : *required) {
    auto set = static_cast<RegSetKind>(bank);
    if (!HasRegisterSet(set))
      return llvm::createStringError("unsupported register bank");

    if (Status error = ReadRegisterSet(set); error.Fail())
      return error.ToError();
  }

  if (auto error = WriteBankRegister(id, value, GetRegisterBanks()))
    return error;

  for (RegisterBankID bank : *required)
    if (Status error = WriteRegisterSet(static_cast<RegSetKind>(bank));
        error.Fail())
      return error.ToError();

  return llvm::Error::success();
}

Status NativeRegisterContextFreeBSD_ppc::ReadAllRegisterValues(
    WritableDataBufferSP &data) {
  auto banks = GetRegisterBanks();
  size_t size = 0;

  for (const auto &bank : banks) {
    auto set = static_cast<RegSetKind>(bank.bank);
    if (!HasRegisterSet(set))
      continue;

    if (Status error = ReadRegisterSet(set); error.Fail())
      return error;

    size += bank.bytes.size();
  }

  data = std::make_shared<DataBufferHeap>(size, 0);

  size_t offset = 0;

  for (const auto &bank : banks) {
    if (!HasRegisterSet(static_cast<RegSetKind>(bank.bank)))
      continue;

    std::memcpy(data->GetBytes() + offset, bank.bytes.data(),
                bank.bytes.size());
    offset += bank.bytes.size();
  }

  return Status();
}

Status NativeRegisterContextFreeBSD_ppc::WriteAllRegisterValues(
    const DataBufferSP &data) {
  auto banks = GetRegisterBanks();
  size_t size = 0;

  for (const auto &bank : banks)
    if (HasRegisterSet(static_cast<RegSetKind>(bank.bank)))
      size += bank.bytes.size();

  if (!data || data->GetByteSize() != size)
    return Status::FromErrorString("invalid private register checkpoint");

  size_t offset = 0;

  for (const auto &bank : banks) {
    auto set = static_cast<RegSetKind>(bank.bank);
    if (!HasRegisterSet(set))
      continue;

    std::memcpy(bank.bytes.data(), data->GetBytes() + offset,
                bank.bytes.size());
    offset += bank.bytes.size();

    if (Status error = WriteRegisterSet(set); error.Fail())
      return error;
  }

  return Status();
}

#endif // defined(__powerpc__)
