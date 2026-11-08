//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextELFCore_riscv64.h"

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_riscv64.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/DataBufferHeap.h"
#include <optional>

using namespace lldb_private;

llvm::Expected<std::unique_ptr<RegisterContextELFCore_riscv64>>
RegisterContextELFCore_riscv64::Create(Thread &thread, const ArchSpec &arch,
                                         const DataExtractor &gpregset,
                                         llvm::ArrayRef<CoreNote> notes) {
  Flags opt_regsets = RegisterInfo_riscv64::eRegsetMaskDefault;

  DataExtractor fpregset = getRegset(notes, arch.GetTriple(), FPR_Desc);
  std::optional<RegisterLayout> layout;
  switch (arch.GetTriple().getOS()) {
  case llvm::Triple::Linux:
    if (fpregset.GetByteSize() >= sizeof(linux_riscv64::FPR))
      opt_regsets.Set(RegisterInfo_riscv64::eRegsetMaskFP);
    layout = CreateLinuxRISCV64RegisterLayout(arch, opt_regsets);
    break;
  default:
    return llvm::createStringError("unsupported OS for RISC-V64 ELF registers");
  }

  return std::unique_ptr<RegisterContextELFCore_riscv64>(
      new RegisterContextELFCore_riscv64(
          thread, std::move(*layout), gpregset, notes));
}

RegisterContextELFCore_riscv64::RegisterContextELFCore_riscv64(
    Thread &thread, RegisterLayout layout, const DataExtractor &gpregset,
    llvm::ArrayRef<CoreNote> notes)
    : ThreadRegisterContext(thread, 0, std::move(layout)) {

  m_gpr.SetData(std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                 gpregset.GetByteSize()));
  m_gpr.SetByteOrder(gpregset.GetByteOrder());

  if (GetDescription()->FindGroup("fpr") != nullptr) {
    ArchSpec arch = GetTargetArchitecture();
    m_fpr = getRegset(notes, arch.GetTriple(), FPR_Desc);
  }
}

RegisterContextELFCore_riscv64::~RegisterContextELFCore_riscv64() = default;

llvm::Expected<RegisterValue>
RegisterContextELFCore_riscv64::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  const auto *reg = description->Find(id);
  if (!reg)
    return llvm::createStringError("register value is unavailable");
  std::vector<RegisterData> banks;
  banks.push_back({0, {m_gpr.GetDataStart(), m_gpr.GetByteSize()}});
  if (GetDescription()->FindGroup("fpr") != nullptr)
    banks.push_back({1, {m_fpr.GetDataStart(), m_fpr.GetByteSize()}});
  auto result = ReadBankRegister(reg->id, banks);
  return result;
}

llvm::Error
RegisterContextELFCore_riscv64::WriteRegister(RegisterID id,
                                                const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}
