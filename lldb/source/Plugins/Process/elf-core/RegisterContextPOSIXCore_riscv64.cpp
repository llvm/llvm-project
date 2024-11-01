//===-- RegisterContextPOSIXCore_riscv64.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextPOSIXCore_riscv64.h"

#include "lldb/Utility/DataBufferHeap.h"

using namespace lldb_private;

std::unique_ptr<RegisterContextCorePOSIX_riscv64>
RegisterContextCorePOSIX_riscv64::Create(Thread &thread, const ArchSpec &arch,
                                         const DataExtractor &gpregset,
                                         llvm::ArrayRef<CoreNote> notes) {
  Flags opt_regsets = RegisterInfoPOSIX_riscv64::eRegsetMaskDefault;

  DataExtractor fpregset = getRegset(notes, arch.GetTriple(), FPR_Desc);
  if (fpregset.GetByteSize() >= sizeof(uint64_t)) {
    opt_regsets.Set(RegisterInfoPOSIX_riscv64::eRegsetMaskFP);
  }

  return std::unique_ptr<RegisterContextCorePOSIX_riscv64>(
      new RegisterContextCorePOSIX_riscv64(
          thread,
          std::make_unique<RegisterInfoPOSIX_riscv64>(arch, opt_regsets),
          gpregset, notes));
}

RegisterContextCorePOSIX_riscv64::RegisterContextCorePOSIX_riscv64(
    Thread &thread, std::unique_ptr<RegisterInfoPOSIX_riscv64> register_info,
    const DataExtractor &gpregset, llvm::ArrayRef<CoreNote> notes)
    : RegisterContextPOSIX_riscv64(thread, std::move(register_info)) {

  m_gpr.SetData(std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                 gpregset.GetByteSize()));
  m_gpr.SetByteOrder(gpregset.GetByteOrder());

  if (m_register_info_up->IsFPPresent()) {
    ArchSpec arch = m_register_info_up->GetTargetArchitecture();
    m_fpr = getRegset(notes, arch.GetTriple(), FPR_Desc);
  }
}

RegisterContextCorePOSIX_riscv64::~RegisterContextCorePOSIX_riscv64() = default;

bool RegisterContextCorePOSIX_riscv64::ReadGPR() { return true; }

bool RegisterContextCorePOSIX_riscv64::ReadFPR() { return true; }

bool RegisterContextCorePOSIX_riscv64::WriteGPR() {
  assert(false && "Writing registers is not allowed for core dumps");
  return false;
}

bool RegisterContextCorePOSIX_riscv64::WriteFPR() {
  assert(false && "Writing registers is not allowed for core dumps");
  return false;
}

bool RegisterContextCorePOSIX_riscv64::ReadRegister(
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

bool RegisterContextCorePOSIX_riscv64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &value) {
  return false;
}
