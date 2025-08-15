//===-- RegisterContextCoreAIX_ppc64.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextCoreAIX_ppc64.h"

#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/RegisterValue.h"

#include "Plugins/Process/Utility/lldb-ppc64le-register-enums.h"
#include "Plugins/Process/elf-core/RegisterUtilities.h"

#include <memory>

using namespace lldb_private;

RegisterContextCoreAIX_ppc64::RegisterContextCoreAIX_ppc64(
    Thread &thread, RegisterInfoInterface *register_info,
    const DataExtractor &gpregset)
    : RegisterContextPOSIX_ppc64le(thread, 0, register_info) {
  m_gpr_buffer = std::make_shared<DataBufferHeap>(gpregset.GetDataStart(),
                                                  gpregset.GetByteSize());
  m_gpr.SetData(m_gpr_buffer);
  m_gpr.SetByteOrder(gpregset.GetByteOrder());

  // This Code is for Registers like FPR, VSR, VMX and is disabled right now.
  // It will be implemented as per need.
  
  /*  ArchSpec arch = register_info->GetTargetArchitecture();
  DataExtractor fpregset;// = getRegset(notes, arch.GetTriple(), FPR_Desc);
  m_fpr_buffer = std::make_shared<DataBufferHeap>(fpregset.GetDataStart(),
                                                  fpregset.GetByteSize());
  m_fpr.SetData(m_fpr_buffer);
  m_fpr.SetByteOrder(fpregset.GetByteOrder());

  DataExtractor vmxregset;// = getRegset(notes, arch.GetTriple(), PPC_VMX_Desc);
  m_vmx_buffer = std::make_shared<DataBufferHeap>(vmxregset.GetDataStart(),
                                                  vmxregset.GetByteSize());
  m_vmx.SetData(m_vmx_buffer);
  m_vmx.SetByteOrder(vmxregset.GetByteOrder());

  DataExtractor vsxregset;// = getRegset(notes, arch.GetTriple(), PPC_VSX_Desc);
  m_vsx_buffer = std::make_shared<DataBufferHeap>(vsxregset.GetDataStart(),
                                                  vsxregset.GetByteSize());
  m_vsx.SetData(m_vsx_buffer);
  m_vsx.SetByteOrder(vsxregset.GetByteOrder());*/
}

size_t RegisterContextCoreAIX_ppc64::GetFPRSize() const {
  return k_num_fpr_registers_ppc64le * sizeof(uint64_t);
}

size_t RegisterContextCoreAIX_ppc64::GetVMXSize() const {
  return (k_num_vmx_registers_ppc64le - 1) * sizeof(uint64_t) * 2 +
         sizeof(uint32_t);
}

size_t RegisterContextCoreAIX_ppc64::GetVSXSize() const {
  return k_num_vsx_registers_ppc64le * sizeof(uint64_t) * 2;
}

bool RegisterContextCoreAIX_ppc64::ReadRegister(
    const RegisterInfo *reg_info, RegisterValue &value) {
  lldb::offset_t offset = reg_info->byte_offset;

  if (IsFPR(reg_info->kinds[lldb::eRegisterKindLLDB])) {
    uint64_t v;
    offset -= GetGPRSize();
    offset = m_fpr.CopyData(offset, reg_info->byte_size, &v);

    if (offset == reg_info->byte_size) {
      value.SetBytes(&v, reg_info->byte_size, m_fpr.GetByteOrder());
      return true;
    }
  } else if (IsVMX(reg_info->kinds[lldb::eRegisterKindLLDB])) {
    uint32_t v[4];
    offset -= GetGPRSize() + GetFPRSize();
    offset = m_vmx.CopyData(offset, reg_info->byte_size, &v);

    if (offset == reg_info->byte_size) {
      value.SetBytes(v, reg_info->byte_size, m_vmx.GetByteOrder());
      return true;
    }
  } else if (IsVSX(reg_info->kinds[lldb::eRegisterKindLLDB])) {
    uint32_t v[4];
    lldb::offset_t tmp_offset;
    offset -= GetGPRSize() + GetFPRSize() + GetVMXSize();

    if (offset < GetVSXSize() / 2) {
      tmp_offset = m_vsx.CopyData(offset / 2, reg_info->byte_size / 2, &v);

      if (tmp_offset != reg_info->byte_size / 2) {
        return false;
      }

      uint8_t *dst = (uint8_t *)&v + sizeof(uint64_t);
      tmp_offset = m_fpr.CopyData(offset / 2, reg_info->byte_size / 2, dst);

      if (tmp_offset != reg_info->byte_size / 2) {
        return false;
      }

      value.SetBytes(&v, reg_info->byte_size, m_vsx.GetByteOrder());
      return true;
    } else {
      offset =
          m_vmx.CopyData(offset - GetVSXSize() / 2, reg_info->byte_size, &v);
      if (offset == reg_info->byte_size) {
        value.SetBytes(v, reg_info->byte_size, m_vmx.GetByteOrder());
        return true;
      }
    }
  } else {
    uint64_t v = m_gpr.GetMaxU64(&offset, reg_info->byte_size);

    if (offset == reg_info->byte_offset + reg_info->byte_size) {
      if (reg_info->byte_size < sizeof(v))
        value = (uint32_t)v;
      else
        value = v;
      return true;
    }
  }

  return false;
}

bool RegisterContextCoreAIX_ppc64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &value) {
  return false;
}
