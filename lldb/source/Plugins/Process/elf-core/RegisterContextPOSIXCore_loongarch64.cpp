//===-- RegisterContextPOSIXCore_loongarch64.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextPOSIXCore_loongarch64.h"

#include "lldb/Utility/DataBufferHeap.h"

using namespace lldb_private;

std::unique_ptr<RegisterContextCorePOSIX_loongarch64>
RegisterContextCorePOSIX_loongarch64::Create(Thread &thread,
                                             const ArchSpec &arch,
                                             const DataExtractor &gpregset,
                                             llvm::ArrayRef<CoreNote> notes) {
  return std::unique_ptr<RegisterContextCorePOSIX_loongarch64>(
      new RegisterContextCorePOSIX_loongarch64(
          thread,
          std::make_unique<RegisterInfoPOSIX_loongarch64>(arch, Flags()),
          gpregset, notes));
}

RegisterContextCorePOSIX_loongarch64::RegisterContextCorePOSIX_loongarch64(
    Thread &thread,
    std::unique_ptr<RegisterInfoPOSIX_loongarch64> register_info,
    const DataExtractor &gpregset, llvm::ArrayRef<CoreNote> notes)
    : RegisterContextPOSIX_loongarch64(thread, std::move(register_info)) {

  m_gpr.SetData(std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                 gpregset.GetByteSize()));
  m_gpr.SetByteOrder(gpregset.GetByteOrder());

  ArchSpec arch = m_register_info_up->GetTargetArchitecture();
  DataExtractor fpregset = getRegset(notes, arch.GetTriple(), FPR_Desc);
  m_fpr.SetData(std::make_shared<DataBufferHeap>(fpregset.GetDataStart(),
                                                 fpregset.GetByteSize()));
  m_fpr.SetByteOrder(fpregset.GetByteOrder());
}

RegisterContextCorePOSIX_loongarch64::~RegisterContextCorePOSIX_loongarch64() =
    default;

bool RegisterContextCorePOSIX_loongarch64::ReadGPR() { return true; }

bool RegisterContextCorePOSIX_loongarch64::ReadFPR() { return true; }

bool RegisterContextCorePOSIX_loongarch64::WriteGPR() {
  assert(false && "Writing registers is not allowed for core dumps");
  return false;
}

bool RegisterContextCorePOSIX_loongarch64::WriteFPR() {
  assert(false && "Writing registers is not allowed for core dumps");
  return false;
}

bool RegisterContextCorePOSIX_loongarch64::ReadRegister(
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

bool RegisterContextCorePOSIX_loongarch64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &value) {
  return false;
}
