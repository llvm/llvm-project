//===-- RegisterInfoPOSIXDynamic_riscv32.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include <lldb/Utility/Flags.h>
#include <stddef.h>

#include "lldb/lldb-defines.h"
#include "llvm/Support/Compiler.h"

#include "RegisterInfoPOSIXDynamic_riscv32.h"

RegisterInfoPOSIXDynamic_riscv32::RegisterInfoPOSIXDynamic_riscv32(
    const lldb_private::ArchSpec &target_arch)
    : lldb_private::RegisterInfoAndSetInterface(target_arch),
      m_target_arch(target_arch) {}

uint32_t RegisterInfoPOSIXDynamic_riscv32::GetRegisterCount() const {
  return m_dyn_reg_infos.GetNumRegisters();
}

size_t RegisterInfoPOSIXDynamic_riscv32::GetGPRSize() const {
  for (uint32_t set_idx = 0; set_idx < GetRegisterSetCount(); ++set_idx) {
    const lldb_private::RegisterSet *set =
        m_dyn_reg_infos.GetRegisterSet(set_idx);
    if (lldb_private::ConstString(set->name) == "GPR")
      return set->num_registers;
  }
  return 0;
}

size_t RegisterInfoPOSIXDynamic_riscv32::GetFPRSize() const {
  for (uint32_t set_idx = 0; set_idx < GetRegisterSetCount(); ++set_idx) {
    const lldb_private::RegisterSet *set =
        m_dyn_reg_infos.GetRegisterSet(set_idx);
    if (lldb_private::ConstString(set->name) == "FPR")
      return set->num_registers;
  }
  return 0;
}

const lldb_private::RegisterInfo *
RegisterInfoPOSIXDynamic_riscv32::GetRegisterInfo() const {
  return &*m_dyn_reg_infos
               .registers<lldb_private::DynamicRegisterInfo::
                              reg_collection_const_range>()
               .begin();
}

size_t RegisterInfoPOSIXDynamic_riscv32::GetRegisterSetCount() const {
  return m_dyn_reg_infos.GetNumRegisterSets();
}

size_t RegisterInfoPOSIXDynamic_riscv32::GetRegisterSetFromRegisterIndex(
    uint32_t reg_index) const {
  for (size_t set_index = 0; set_index < m_dyn_reg_infos.GetNumRegisterSets();
       ++set_index) {
    const lldb_private::RegisterSet *reg_set =
        m_dyn_reg_infos.GetRegisterSet(set_index);
    for (uint32_t idx = 0; idx < reg_set->num_registers; ++idx)
      if (reg_set->registers[idx] == reg_index)
        return set_index;
  }
  return LLDB_INVALID_REGNUM;
}

const lldb_private::RegisterSet *
RegisterInfoPOSIXDynamic_riscv32::GetRegisterSet(size_t set_index) const {
  if (set_index < GetRegisterSetCount())
    return m_dyn_reg_infos.GetRegisterSet(set_index);
  return nullptr;
}

size_t RegisterInfoPOSIXDynamic_riscv32::SetRegisterInfo(
    std::vector<lldb_private::DynamicRegisterInfo::Register> regs) {
  return m_dyn_reg_infos.SetRegisterInfo(std::move(regs), m_target_arch);
}

const lldb_private::RegisterInfo *
RegisterInfoPOSIXDynamic_riscv32::GetRegisterInfo(
    llvm::StringRef reg_name) const {
  return m_dyn_reg_infos.GetRegisterInfo(reg_name);
}
