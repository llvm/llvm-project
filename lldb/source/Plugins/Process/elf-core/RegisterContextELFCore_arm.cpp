//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextELFCore_arm.h"

#include "lldb/Target/Thread.h"
#include "lldb/Register/RegisterValue.h"

#include <memory>

using namespace lldb_private;

RegisterContextELFCore_arm::RegisterContextELFCore_arm(
    Thread &thread, RegisterLayout layout, const DataExtractor &gpregset,
    llvm::ArrayRef<CoreNote> notes)
    : ThreadRegisterContext(thread, 0, std::move(layout)) {
  m_gpr_buffer = std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                  gpregset.GetByteSize());
  m_gpr.SetData(m_gpr_buffer);
  m_gpr.SetByteOrder(gpregset.GetByteOrder());

  const llvm::Triple &target_triple = GetTargetArchitecture().GetTriple();
  m_fpr = getRegset(notes, target_triple, ARM_VFP_Desc);
}

RegisterContextELFCore_arm::~RegisterContextELFCore_arm() = default;

llvm::Expected<RegisterValue>
RegisterContextELFCore_arm::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  if (!reg_info)
    return llvm::createStringError("register value is unavailable");
  const auto *reg = GetDescription()->Find(reg_info->id);
  if (!reg)
    return llvm::createStringError("register value is unavailable");
  auto result = ReadBankRegister(
      reg->id, {{0, {m_gpr.GetDataStart(), m_gpr.GetByteSize()}},
                {1, {m_fpr.GetDataStart(), m_fpr.GetByteSize()}}});
  return result;
}

bool RegisterContextELFCore_arm::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  return false;
}

llvm::Error
RegisterContextELFCore_arm::WriteRegister(RegisterID id,
                                                 const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}

bool RegisterContextELFCore_arm::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return false;
}

bool RegisterContextELFCore_arm::HardwareSingleStep(bool enable) {
  return false;
}
