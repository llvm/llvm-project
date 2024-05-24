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
RegisterContextCorePOSIX_riscv64::Create(
    lldb_private::Thread &thread, const lldb_private::ArchSpec &arch,
    const lldb_private::DataExtractor &gpregset,
    llvm::ArrayRef<lldb_private::CoreNote> notes) {
  Flags flags = 0;

  auto register_info_up =
      std::make_unique<RegisterInfoPOSIX_riscv64>(arch, flags);
  return std::unique_ptr<RegisterContextCorePOSIX_riscv64>(
      new RegisterContextCorePOSIX_riscv64(thread, std::move(register_info_up),
                                           gpregset, notes));
}

RegisterContextCorePOSIX_riscv64::RegisterContextCorePOSIX_riscv64(
    Thread &thread, std::unique_ptr<RegisterInfoPOSIX_riscv64> register_info,
    const DataExtractor &gpregset, llvm::ArrayRef<CoreNote> notes)
    : RegisterContextPOSIX_riscv64(thread, std::move(register_info)) {

  m_gpr_buffer = std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                  gpregset.GetByteSize());
  m_gpr.SetData(m_gpr_buffer);
  m_gpr.SetByteOrder(gpregset.GetByteOrder());

  ArchSpec arch = m_register_info_up->GetTargetArchitecture();
  DataExtractor fpregset = getRegset(notes, arch.GetTriple(), FPR_Desc);
  m_fpr_buffer = std::make_shared<DataBufferHeap>(fpregset.GetDataStart(),
                                                  fpregset.GetByteSize());
  m_fpr.SetData(m_fpr_buffer);
  m_fpr.SetByteOrder(fpregset.GetByteOrder());
}

RegisterContextCorePOSIX_riscv64::~RegisterContextCorePOSIX_riscv64() = default;

bool RegisterContextCorePOSIX_riscv64::ReadGPR() { return true; }

bool RegisterContextCorePOSIX_riscv64::ReadFPR() { return true; }

bool RegisterContextCorePOSIX_riscv64::WriteGPR() {
  assert(0);
  return false;
}

bool RegisterContextCorePOSIX_riscv64::WriteFPR() {
  assert(0);
  return false;
}

bool RegisterContextCorePOSIX_riscv64::ReadRegister(
    const RegisterInfo *reg_info, RegisterValue &value) {
  const uint8_t *src = nullptr;
  lldb::offset_t offset = reg_info->byte_offset;

  if (IsGPR(reg_info->kinds[lldb::eRegisterKindLLDB])) {
    src = m_gpr.GetDataStart();
  } else { // IsFPR
    src = m_fpr.GetDataStart();
    offset -= GetGPRSize();
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
