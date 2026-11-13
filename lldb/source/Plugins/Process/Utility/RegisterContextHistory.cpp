//===-- RegisterContextHistory.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/RegisterValue.h"
#include "RegisterDescriptionPC.h"
#include "RegisterContextHistory.h"

using namespace lldb;
using namespace lldb_private;

RegisterContextHistory::RegisterContextHistory(Thread &thread,
                                               uint32_t concrete_frame_idx,
                                               uint32_t address_byte_size,
                                               addr_t pc_value)
    : ThreadRegisterContext(thread, concrete_frame_idx,
                            CreatePCRegisterDescription(address_byte_size)),
      m_pc_value(pc_value) {}

RegisterContextHistory::~RegisterContextHistory() = default;

llvm::Expected<RegisterValue>
RegisterContextHistory::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  if (!reg_info)
    return llvm::createStringError("register value is unavailable");
  uint32_t reg_number = reg_info->numbers[eRegisterKindGeneric];
  if (reg_number == LLDB_REGNUM_GENERIC_PC) {
    value.SetUInt(m_pc_value, reg_info->byte_size);
    return value;
  }
  return llvm::createStringError("register value is unavailable");
}

llvm::Error RegisterContextHistory::WriteRegister(RegisterID id,
                                                  const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}

bool RegisterContextHistory::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  return false;
}

bool RegisterContextHistory::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return false;
}
