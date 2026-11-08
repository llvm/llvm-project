//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextELFCore_loongarch64.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_loongarch64.h"
#include "lldb/Register/RegisterLayout.h"

#include "lldb/Utility/DataBufferHeap.h"

using namespace lldb_private;

llvm::Expected<std::unique_ptr<RegisterContextELFCore_loongarch64>>
RegisterContextELFCore_loongarch64::Create(Thread &thread,
                                             const ArchSpec &arch,
                                             const DataExtractor &gpregset,
                                             llvm::ArrayRef<CoreNote> notes) {
  switch (arch.GetTriple().getOS()) {
  case llvm::Triple::Linux:
    return std::unique_ptr<RegisterContextELFCore_loongarch64>(
        new RegisterContextELFCore_loongarch64(
            thread, CreateLinuxLoongArch64RegisterLayout(arch), gpregset,
            notes));
  default:
    return llvm::createStringError(
        "unsupported OS for LoongArch64 ELF registers");
  }
}

RegisterContextELFCore_loongarch64::RegisterContextELFCore_loongarch64(
    Thread &thread, RegisterLayout layout, const DataExtractor &gpregset,
    llvm::ArrayRef<CoreNote> notes)
    : ThreadRegisterContext(thread, 0, std::move(layout)) {

  m_gpr.SetData(std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                 gpregset.GetByteSize()));
  m_gpr.SetByteOrder(gpregset.GetByteOrder());

  ArchSpec arch = GetTargetArchitecture();
  DataExtractor fpregset = getRegset(notes, arch.GetTriple(), FPR_Desc);
  m_fpr.SetData(std::make_shared<DataBufferHeap>(fpregset.GetDataStart(),
                                                 fpregset.GetByteSize()));
  m_fpr.SetByteOrder(fpregset.GetByteOrder());
}

RegisterContextELFCore_loongarch64::~RegisterContextELFCore_loongarch64() =
    default;

llvm::Expected<RegisterValue>
RegisterContextELFCore_loongarch64::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  const auto *reg = GetDescription()->Find(reg_info->id);
  if (!reg)
    return llvm::createStringError("register value is unavailable");
  const RegisterData banks[] = {
      {0, {m_gpr.GetDataStart(), m_gpr.GetByteSize()}},
      {1, {m_fpr.GetDataStart(), m_fpr.GetByteSize()}}};
  auto result = ReadBankRegister(reg->id, banks);
  return result;
}

llvm::Error RegisterContextELFCore_loongarch64::WriteRegister(
    RegisterID id, const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}
