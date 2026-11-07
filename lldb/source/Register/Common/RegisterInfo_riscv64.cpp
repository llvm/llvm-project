//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterInfo_riscv64.h"
#include "lldb/Register/Arch/lldb-riscv-ehframe-register-enums.h"

#include "lldb/Register/Arch/lldb-riscv-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv-dwarf-register-enums.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-types.h"
#include "lldb/lldb-private.h"
#include <cstddef>
#include <cstdint>

#include "lldb/Utility/Flags.h"
#include "lldb/lldb-defines.h"
#include "llvm/Support/Compiler.h"

#include <cassert>
#include <stddef.h>

#define GPR_OFFSET(idx) ((idx)*8 + 0)
#define FPR_OFFSET(idx) ((idx)*8 + sizeof(RegisterInfo_riscv64::GPR))


namespace reg_ids = lldb_private::registers::riscv::id;
using namespace lldb_private;

#define DECLARE_REGISTER_INFOS_RISCV64_STRUCT
#include "Register/Arch/RegisterInfos_riscv64.h"
#undef DECLARE_REGISTER_INFOS_RISCV64_STRUCT

// Number of register sets provided by this context.
enum {
  k_num_gpr_registers = reg_ids::gpr_last - reg_ids::gpr_first + 1,
  k_num_fpr_registers = reg_ids::fpr_last - reg_ids::fpr_first + 1,
  k_num_register_sets_default = 1
};

// RISC-V64 general purpose registers.
static const uint32_t g_gpr_regnums_riscv64[] = {
    reg_ids::gpr_pc,  reg_ids::gpr_ra,    reg_ids::gpr_sp,  reg_ids::gpr_x3,
    reg_ids::gpr_x4,  reg_ids::gpr_x5,    reg_ids::gpr_x6,  reg_ids::gpr_x7,
    reg_ids::gpr_fp,  reg_ids::gpr_x9,    reg_ids::gpr_x10, reg_ids::gpr_x11,
    reg_ids::gpr_x12, reg_ids::gpr_x13,   reg_ids::gpr_x14, reg_ids::gpr_x15,
    reg_ids::gpr_x16, reg_ids::gpr_x17,   reg_ids::gpr_x18, reg_ids::gpr_x19,
    reg_ids::gpr_x20, reg_ids::gpr_x21,   reg_ids::gpr_x22, reg_ids::gpr_x23,
    reg_ids::gpr_x24, reg_ids::gpr_x25,   reg_ids::gpr_x26, reg_ids::gpr_x27,
    reg_ids::gpr_x28, reg_ids::gpr_x29,   reg_ids::gpr_x30, reg_ids::gpr_x31,
    reg_ids::gpr_x0,  LLDB_INVALID_REGNUM};

static_assert(((sizeof g_gpr_regnums_riscv64 /
                sizeof g_gpr_regnums_riscv64[0]) -
               1) == k_num_gpr_registers,
              "g_gpr_regnums_riscv64 has wrong number of register infos");

// Register sets for RISC-V64.
static const lldb_private::RegisterSet g_reg_set_gpr_riscv64 = {
    "General Purpose Registers", "gpr", k_num_gpr_registers,
    g_gpr_regnums_riscv64};
static const lldb_private::RegisterSet g_reg_set_fpr_riscv64 = {
    "Floating Point Registers", "fpr", k_num_fpr_registers, nullptr};

RegisterInfo_riscv64::RegisterInfo_riscv64(
    const lldb_private::ArchSpec &target_arch, lldb_private::Flags opt_regsets)
    : lldb_private::RegisterInfoAndSetInterface(target_arch),
      m_opt_regsets(opt_regsets) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::riscv64: {
    // By-default considering RISC-V has only GPR.
    // Other register sets could be enabled optionally by opt_regsets.
    AddRegSetGP();

    if (m_opt_regsets.AnySet(eRegsetMaskFP))
      AddRegSetFP();

    break;
  }
  default:
    assert(false && "Unhandled target architecture.");
  }
}

void RegisterInfo_riscv64::AddRegSetGP() {
  m_register_infos.resize(k_num_gpr_registers);
  memcpy(&m_register_infos[0], g_register_infos_riscv64_gpr,
         sizeof(g_register_infos_riscv64_gpr));
  m_register_sets.push_back(g_reg_set_gpr_riscv64);

  m_per_regset_regnum_range[GPRegSet] =
      std::make_pair(reg_ids::gpr_first, m_register_infos.size());
}

void RegisterInfo_riscv64::AddRegSetFP() {
  const uint32_t register_info_count = m_register_infos.size();
  const uint32_t register_set_count = m_register_sets.size();

  // Filling m_register_infos.
  // For FPR case we do not need to correct register offsets and kinds
  // while for other further cases (like VPR), register offset/kind
  // should be started counting from the last one in previously added
  // regset. This is needed for the case e.g. when architecture has GPR + VPR
  // sets only.
  m_register_infos.resize(register_info_count + k_num_fpr_registers);
  memcpy(&m_register_infos[register_info_count], g_register_infos_riscv64_fpr,
         sizeof(g_register_infos_riscv64_fpr));

  // Filling m_register_sets with enabled register set
  for (uint32_t i = 0; i < k_num_fpr_registers; i++)
    m_fp_regnum_collection.push_back(register_info_count + i);
  m_register_sets.push_back(g_reg_set_fpr_riscv64);
  m_register_sets.back().registers = m_fp_regnum_collection.data();

  m_per_regset_regnum_range[register_set_count] =
      std::make_pair(register_info_count, m_register_infos.size());
}

uint32_t RegisterInfo_riscv64::GetRegisterCount() const {
  return m_register_infos.size();
}

size_t RegisterInfo_riscv64::GetGPRSize() const {
  return sizeof(struct RegisterInfo_riscv64::GPR);
}

size_t RegisterInfo_riscv64::GetFPRSize() const {
  return sizeof(struct RegisterInfo_riscv64::FPR);
}

const lldb_private::RegisterInfo *
RegisterInfo_riscv64::GetRegisterInfo() const {
  return m_register_infos.data();
}

size_t RegisterInfo_riscv64::GetRegisterSetCount() const {
  return m_register_sets.size();
}

size_t RegisterInfo_riscv64::GetRegisterSetFromRegisterIndex(
    uint32_t reg_index) const {
  for (const auto &regset_range : m_per_regset_regnum_range) {
    if (reg_index >= regset_range.second.first &&
        reg_index < regset_range.second.second)
      return regset_range.first;
  }
  return LLDB_INVALID_REGNUM;
}

bool RegisterInfo_riscv64::IsFPReg(unsigned reg) const {
  return llvm::is_contained(m_fp_regnum_collection, reg);
}

const lldb_private::RegisterSet *
RegisterInfo_riscv64::GetRegisterSet(size_t set_index) const {
  if (set_index < GetRegisterSetCount())
    return &m_register_sets[set_index];
  return nullptr;
}
