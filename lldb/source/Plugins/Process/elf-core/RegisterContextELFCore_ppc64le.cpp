//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextELFCore_ppc64le.h"

#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Register/RegisterValue.h"

#include "lldb/Register/Arch/lldb-ppc64le-register-enums.h"
#include "Plugins/Process/elf-core/RegisterUtilities.h"

#include <memory>

using namespace lldb_private;

RegisterContextELFCore_ppc64le::RegisterContextELFCore_ppc64le(
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

  DataExtractor vmxregset = getRegset(notes, arch.GetTriple(), PPC_VMX_Desc);
  m_vmx_buffer = std::make_shared<DataBufferHeap>(vmxregset.GetDataStart(),
                                                  vmxregset.GetByteSize());
  m_vmx.SetData(m_vmx_buffer);
  m_vmx.SetByteOrder(vmxregset.GetByteOrder());

  DataExtractor vsxregset = getRegset(notes, arch.GetTriple(), PPC_VSX_Desc);
  m_vsx_buffer = std::make_shared<DataBufferHeap>(vsxregset.GetDataStart(),
                                                  vsxregset.GetByteSize());
  m_vsx.SetData(m_vsx_buffer);
  m_vsx.SetByteOrder(vsxregset.GetByteOrder());
}

llvm::Expected<RegisterValue>
RegisterContextELFCore_ppc64le::ReadRegister(RegisterID id) {
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
                {2, {m_vmx.GetDataStart(), m_vmx.GetByteSize()}},
                {3, {m_vsx.GetDataStart(), m_vsx.GetByteSize()}}});
  return result;
}

llvm::Error
RegisterContextELFCore_ppc64le::WriteRegister(RegisterID id,
                                                const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}
