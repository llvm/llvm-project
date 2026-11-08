//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <set>

#include "RegisterContextELFCore_x86.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Register/RegisterValue.h"

using namespace lldb_private;

RegisterContextELFCore_x86::RegisterContextELFCore_x86(
    Thread &thread, RegisterLayout layout, const DataExtractor &gpregset,
    llvm::ArrayRef<CoreNote> notes)
    : ThreadRegisterContext(thread, 0, std::move(layout)) {
  size_t size, len;

  size = GetBankSize(0);
  m_gpregset.reset(new uint8_t[size]);
  len =
      gpregset.ExtractBytes(0, size, lldb::eByteOrderLittle, m_gpregset.get());
  if (len != size)
    m_gpregset.reset();

  DataExtractor fpregset =
      getRegset(notes, GetTargetArchitecture().GetTriple(), FPR_Desc);
  size = sizeof(FXSAVE);
  m_fpregset.reset(new uint8_t[size]);
  len =
      fpregset.ExtractBytes(0, size, lldb::eByteOrderLittle, m_fpregset.get());
  if (len != size)
    m_fpregset.reset();

  const size_t group_count = [&]() -> size_t {
    // Keep the existing AVX group visibility when a complete FXSAVE note
    // exists.
    return m_fpregset ? 3 : 2;
  }();
  auto previous = GetDescription();
  auto selected = previous->GetGroups().take_front(group_count);
  std::vector<RegisterGroup> groups(selected.begin(), selected.end());
  std::set<RegisterID> visible;
  for (const auto &group : groups)
    visible.insert(group.registers.begin(), group.registers.end());
  std::vector<RegisterID> order;
  for (RegisterID id : previous->GetUserRegisters())
    if (visible.count(id))
      order.push_back(id);
  auto description = llvm::cantFail(RegisterDescription::Create(
      {previous->GetRegisters().begin(), previous->GetRegisters().end()},
      std::move(groups),
      {previous->GetOwnedTypes().begin(), previous->GetOwnedTypes().end()}));
  auto presentation = llvm::cantFail(description.WithUserRegisters(order));
  SetRegisterLayout(llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(std::move(presentation)),
      GetLayout())));
}

llvm::Expected<RegisterValue>
RegisterContextELFCore_x86::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  auto result = ReadBankRegister(
      reg_info->id,
      {{x86_layout::GP, {m_gpregset.get(), m_gpregset ? GetBankSize(0) : 0}},
       {x86_layout::FP, {m_fpregset.get(), m_fpregset ? sizeof(FXSAVE) : 0}}});
  return result;
}

bool RegisterContextELFCore_x86::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  return false;
}

llvm::Error
RegisterContextELFCore_x86::WriteRegister(RegisterID id,
                                            const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}

bool RegisterContextELFCore_x86::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return false;
}

bool RegisterContextELFCore_x86::HardwareSingleStep(bool enable) {
  return false;
}
