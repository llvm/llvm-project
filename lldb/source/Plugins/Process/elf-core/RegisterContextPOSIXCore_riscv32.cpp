//===-- RegisterContextPOSIXCore_riscv32.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextPOSIXCore_riscv32.h"
#include "lldb/Utility/DataBufferHeap.h"

using namespace lldb_private;

std::unique_ptr<RegisterContextCorePOSIX_riscv32>
RegisterContextCorePOSIX_riscv32::Create(Thread &thread, const ArchSpec &arch,
                                         const DataExtractor &gpregset,
                                         llvm::ArrayRef<CoreNote> notes) {
  Flags opt_regsets = RegisterInfoPOSIX_riscv32::eRegsetMaskDefault;

  return std::unique_ptr<RegisterContextCorePOSIX_riscv32>(
      new RegisterContextCorePOSIX_riscv32(
          thread,
          std::make_unique<RegisterInfoPOSIX_riscv32>(arch, opt_regsets),
          gpregset, notes));
}

RegisterContextCorePOSIX_riscv32::RegisterContextCorePOSIX_riscv32(
    Thread &thread, std::unique_ptr<RegisterInfoPOSIX_riscv32> register_info,
    const DataExtractor &gpregset, llvm::ArrayRef<CoreNote> notes)
    : RegisterContextPOSIX_riscv32(thread, std::move(register_info)) {

  m_gpr.SetData(std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                 gpregset.GetByteSize()));
  m_gpr.SetByteOrder(gpregset.GetByteOrder());

  if (m_register_info_up->IsFPPresent()) {
    ArchSpec arch = m_register_info_up->GetTargetArchitecture();
    m_fpr = getRegset(notes, arch.GetTriple(), FPR_Desc);
  }
}

RegisterContextCorePOSIX_riscv32::~RegisterContextCorePOSIX_riscv32() = default;

bool RegisterContextCorePOSIX_riscv32::ReadGPR() { return true; }

bool RegisterContextCorePOSIX_riscv32::ReadFPR() { return true; }

bool RegisterContextCorePOSIX_riscv32::WriteGPR() {
  assert(false && "Writing registers is not allowed for core dumps");
  return false;
}

bool RegisterContextCorePOSIX_riscv32::WriteFPR() {
  assert(false && "Writing registers is not allowed for core dumps");
  return false;
}

bool RegisterContextCorePOSIX_riscv32::ReadRegister(
    const RegisterInfo *reg_info, RegisterValue &value) {
  const uint8_t *src = nullptr;
  lldb::offset_t offset = reg_info->byte_offset;

  if (IsGPR(reg_info->kinds[lldb::eRegisterKindLLDB])) {
    src = m_gpr.GetDataStart();
  } else if (IsFPR(reg_info->kinds[lldb::eRegisterKindLLDB])) {
    src = m_fpr.GetDataStart();
    offset -= GetGPRSize();
  } else {
    return false;
  }

  Status error;
  value.SetFromMemoryData(*reg_info, src + offset, reg_info->byte_size,
                          lldb::eByteOrderLittle, error);
  return error.Success();
}

bool RegisterContextCorePOSIX_riscv32::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &value) {
  return false;
}
