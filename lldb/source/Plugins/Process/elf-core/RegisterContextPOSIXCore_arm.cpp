//===-- RegisterContextPOSIXCore_arm.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextPOSIXCore_arm.h"

#include "lldb/Target/Thread.h"
#include "lldb/Utility/RegisterValue.h"

#include <memory>

using namespace lldb_private;

RegisterContextCorePOSIX_arm::RegisterContextCorePOSIX_arm(
    Thread &thread, std::unique_ptr<RegisterInfoPOSIX_arm> register_info,
    const DataExtractor &gpregset, llvm::ArrayRef<CoreNote> notes)
    : RegisterContextPOSIX_arm(thread, std::move(register_info)) {
  m_gpr_buffer = std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                  gpregset.GetByteSize());
  m_gpr.SetData(m_gpr_buffer);
  m_gpr.SetByteOrder(gpregset.GetByteOrder());

  const llvm::Triple &target_triple =
      m_register_info_up->GetTargetArchitecture().GetTriple();
  m_fpr = getRegset(notes, target_triple, ARM_VFP_Desc);
}

RegisterContextCorePOSIX_arm::~RegisterContextCorePOSIX_arm() = default;

bool RegisterContextCorePOSIX_arm::ReadGPR() { return true; }

bool RegisterContextCorePOSIX_arm::ReadFPR() { return false; }

bool RegisterContextCorePOSIX_arm::WriteGPR() {
  assert(0);
  return false;
}

bool RegisterContextCorePOSIX_arm::WriteFPR() {
  assert(0);
  return false;
}

bool RegisterContextCorePOSIX_arm::ReadRegister(const RegisterInfo *reg_info,
                                                RegisterValue &value) {
  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM)
    return false;

  if (IsGPR(reg)) {
    lldb::offset_t offset = reg_info->byte_offset;
    if (m_gpr.ValidOffsetForDataOfSize(offset, reg_info->byte_size)) {
      value = m_gpr.GetMaxU64(&offset, reg_info->byte_size);
      return offset == reg_info->byte_offset + reg_info->byte_size;
    }
  } else if (IsFPR(reg)) {
    assert(reg_info->byte_offset >= GetGPRSize());
    lldb::offset_t offset = reg_info->byte_offset - GetGPRSize();
    if (m_fpr.ValidOffsetForDataOfSize(offset, reg_info->byte_size))
      return value
          .SetValueFromData(*reg_info, m_fpr, offset, /*partial_data_ok=*/false)
          .Success();
  }

  return false;
}

bool RegisterContextCorePOSIX_arm::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  return false;
}

bool RegisterContextCorePOSIX_arm::WriteRegister(const RegisterInfo *reg_info,
                                                 const RegisterValue &value) {
  return false;
}

bool RegisterContextCorePOSIX_arm::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return false;
}

bool RegisterContextCorePOSIX_arm::HardwareSingleStep(bool enable) {
  return false;
}
