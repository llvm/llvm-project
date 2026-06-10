//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterInfoPOSIXDynamic_riscv32.h"

#include "lldb-riscv-register-enums.h"
#include "lldb/lldb-defines.h"
#include "llvm/Support/Compiler.h"

#include <iomanip>
#include <sstream>
#include <stddef.h>

#define DECLARE_REGISTER_INFOS_RISCV32_STRUCT
#include "Plugins/Process/Utility/RegisterInfos_riscv32.h"
#undef DECLARE_REGISTER_INFOS_RISCV32_STRUCT

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
    if (strcmp(set->name, "GPR") == 0)
      return set->num_registers;
  }
  return 0;
}

size_t RegisterInfoPOSIXDynamic_riscv32::GetFPRSize() const {
  for (uint32_t set_idx = 0; set_idx < GetRegisterSetCount(); ++set_idx) {
    const lldb_private::RegisterSet *set =
        m_dyn_reg_infos.GetRegisterSet(set_idx);
    if (strcmp(set->name, "FPR") == 0)
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

std::vector<lldb_private::RegisterInfo>
RegisterInfoPOSIXDynamic_riscv32::GetCSRegisterInfos(
    const std::vector<std::string> &features) {
  // Sort and deduplicate the feature list to make the resulting CSR metadata
  // independent of caller ordering.
  llvm::SmallVector<llvm::StringRef, 32> normalized_features;
  normalized_features.reserve(features.size());
  for (const std::string &feature : features)
    normalized_features.push_back(feature);
  std::sort(normalized_features.begin(), normalized_features.end());
  normalized_features.erase(
      std::unique(normalized_features.begin(), normalized_features.end()),
      normalized_features.end());

  const uint32_t k_num_csr_registers = csr_last_riscv - csr_first_riscv + 1;
  std::vector<lldb_private::RegisterInfo> cs_reg_infos;
  cs_reg_infos.reserve(k_num_csr_registers);

  // Construct default CS register information.
  for (uint32_t reg = 0; reg < k_num_csr_registers; ++reg) {
    lldb_private::RegisterInfo csr{};
    for (auto &kind : csr.kinds)
      kind = LLDB_INVALID_REGNUM;

    std::stringstream reg_hex;
    reg_hex << "0x" << std::hex << reg;
    lldb_private::ConstString name(std::string("csr_") + reg_hex.str());

    csr.name = name.GetCString();
    csr.alt_name = csr.name;
    csr.byte_size = 4;
    csr.byte_offset = 0;
    csr.encoding = lldb::eEncodingUint;
    csr.format = lldb::eFormatHex;
    csr.kinds[lldb::eRegisterKindEHFrame] = riscv_dwarf::dwarf_first_csr + reg;
    csr.kinds[lldb::eRegisterKindDWARF] = riscv_dwarf::dwarf_first_csr + reg;
    csr.kinds[lldb::eRegisterKindGeneric] = LLDB_INVALID_REGNUM;
    csr.kinds[lldb::eRegisterKindProcessPlugin] = LLDB_INVALID_REGNUM;
    csr.kinds[lldb::eRegisterKindLLDB] = csr_first_riscv + reg;
    csr.value_regs = nullptr;
    csr.invalidate_regs = nullptr;
    csr.flags_type = nullptr;

    cs_reg_infos.push_back(csr);
  }

  // Patch application is order-dependent; later patches override earlier ones
  // for the same CSR address.
  ConfigureCSRegInfos(cs_reg_infos, "default");

  for (const auto &feature : normalized_features)
    ConfigureCSRegInfos(cs_reg_infos, feature);

  return cs_reg_infos;
}

void RegisterInfoPOSIXDynamic_riscv32::ConfigureCSRegInfos(
    std::vector<lldb_private::RegisterInfo> &cs_reg_infos,
    llvm::StringRef feature) {
  auto it = g_register_infos_riscv32_csr_patches.find(feature);
  if (it == g_register_infos_riscv32_csr_patches.end())
    return;

  for (const auto &csr : it->second) {
    const uint32_t lldb_reg = csr.kinds[lldb::eRegisterKindLLDB];
    if (lldb_reg < csr_first_riscv || lldb_reg > csr_last_riscv)
      continue;
    uint32_t idx = lldb_reg - csr_first_riscv;
    if (idx < cs_reg_infos.size())
      cs_reg_infos[idx] = csr;
  }
}
