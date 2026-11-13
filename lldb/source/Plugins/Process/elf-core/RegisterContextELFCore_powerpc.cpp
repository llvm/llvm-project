//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextELFCore_powerpc.h"

#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Register/RegisterValue.h"

#include <memory>

using namespace lldb_private;

RegisterContextELFCore_powerpc::RegisterContextELFCore_powerpc(
    Thread &thread, RegisterLayout layout, const DataExtractor &gpregset,
    llvm::ArrayRef<CoreNote> notes)
    : ThreadRegisterContext(thread, 0, std::move(layout)) {
  m_gpr_buffer = std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                  gpregset.GetByteSize());
  m_gpr.SetData(m_gpr_buffer);
  m_gpr.SetByteOrder(gpregset.GetByteOrder());

  ArchSpec arch = GetTargetArchitecture();
  DataExtractor fpregset = getRegset(notes, arch.GetTriple(), FPR_Desc);
  m_fpr_buffer = std::make_shared<DataBufferHeap>(fpregset.GetDataStart(),
                                                  fpregset.GetByteSize());
  m_fpr.SetData(m_fpr_buffer);
  m_fpr.SetByteOrder(fpregset.GetByteOrder());

  DataExtractor vregset = getRegset(notes, arch.GetTriple(), PPC_VMX_Desc);
  m_vec_buffer = std::make_shared<DataBufferHeap>(vregset.GetDataStart(),
                                                  vregset.GetByteSize());
  m_vec.SetData(m_vec_buffer);
  m_vec.SetByteOrder(vregset.GetByteOrder());
}

RegisterContextELFCore_powerpc::~RegisterContextELFCore_powerpc() = default;

llvm::Expected<RegisterValue>
RegisterContextELFCore_powerpc::ReadRegister(RegisterID id) {
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
                {1, {m_fpr.GetDataStart(), m_fpr.GetByteSize()}},
                {2, {m_vec.GetDataStart(), m_vec.GetByteSize()}}});
  return result;
}

bool RegisterContextELFCore_powerpc::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  return false;
}

llvm::Error
RegisterContextELFCore_powerpc::WriteRegister(RegisterID id,
                                                const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}

bool RegisterContextELFCore_powerpc::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return false;
}

bool RegisterContextELFCore_powerpc::HardwareSingleStep(bool enable) {
  return false;
}
