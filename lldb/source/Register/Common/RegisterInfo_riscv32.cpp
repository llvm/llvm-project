//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterInfo_riscv32.h"
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

#define GPR_OFFSET(idx) ((idx) * 4 + 0)
#define FPR_OFFSET(idx) ((idx) * 4 + sizeof(RegisterInfo_riscv32::GPR))

#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterInfo_riscv32::GPR) +                                    \
   sizeof(RegisterInfo_riscv32::FPR))


namespace reg_ids = lldb_private::registers::riscv::id;
using namespace lldb_private;

#define DECLARE_REGISTER_INFOS_RISCV32_STRUCT
#include "Register/Arch/RegisterInfos_riscv32.h"
#undef DECLARE_REGISTER_INFOS_RISCV32_STRUCT

const lldb_private::RegisterInfo *RegisterInfo_riscv32::GetRegisterInfoPtr(
    const lldb_private::ArchSpec &target_arch) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::riscv32:
    return g_register_infos_riscv32_le;
  default:
    assert(false && "Unhandled target architecture.");
    return nullptr;
  }
}

uint32_t RegisterInfo_riscv32::GetRegisterInfoCount(
    const lldb_private::ArchSpec &target_arch) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::riscv32:
    return static_cast<uint32_t>(sizeof(g_register_infos_riscv32_le) /
                                 sizeof(g_register_infos_riscv32_le[0]));
  default:
    assert(false && "Unhandled target architecture.");
    return 0;
  }
}

// Number of register sets provided by this context.
enum {
  k_num_gpr_registers = reg_ids::gpr_last - reg_ids::gpr_first + 1,
  k_num_fpr_registers = reg_ids::fpr_last - reg_ids::fpr_first + 1,
  k_num_register_sets = 2
};

// RISC-V32 general purpose registers.
static const uint32_t g_gpr_regnums_riscv32[] = {
    reg_ids::gpr_pc,  reg_ids::gpr_ra,    reg_ids::gpr_sp,  reg_ids::gpr_x3,
    reg_ids::gpr_x4,  reg_ids::gpr_x5,    reg_ids::gpr_x6,  reg_ids::gpr_x7,
    reg_ids::gpr_fp,  reg_ids::gpr_x9,    reg_ids::gpr_x10, reg_ids::gpr_x11,
    reg_ids::gpr_x12, reg_ids::gpr_x13,   reg_ids::gpr_x14, reg_ids::gpr_x15,
    reg_ids::gpr_x16, reg_ids::gpr_x17,   reg_ids::gpr_x18, reg_ids::gpr_x19,
    reg_ids::gpr_x20, reg_ids::gpr_x21,   reg_ids::gpr_x22, reg_ids::gpr_x23,
    reg_ids::gpr_x24, reg_ids::gpr_x25,   reg_ids::gpr_x26, reg_ids::gpr_x27,
    reg_ids::gpr_x28, reg_ids::gpr_x29,   reg_ids::gpr_x30, reg_ids::gpr_x31,
    reg_ids::gpr_x0,  LLDB_INVALID_REGNUM};

static_assert(((sizeof g_gpr_regnums_riscv32 /
                sizeof g_gpr_regnums_riscv32[0]) -
               1) == k_num_gpr_registers,
              "g_gpr_regnums_riscv32 has wrong number of register infos");

// RISC-V32 floating point registers.
static const uint32_t g_fpr_regnums_riscv32[] = {
    reg_ids::fpr_f0,   reg_ids::fpr_f1,    reg_ids::fpr_f2,  reg_ids::fpr_f3,
    reg_ids::fpr_f4,   reg_ids::fpr_f5,    reg_ids::fpr_f6,  reg_ids::fpr_f7,
    reg_ids::fpr_f8,   reg_ids::fpr_f9,    reg_ids::fpr_f10, reg_ids::fpr_f11,
    reg_ids::fpr_f12,  reg_ids::fpr_f13,   reg_ids::fpr_f14, reg_ids::fpr_f15,
    reg_ids::fpr_f16,  reg_ids::fpr_f17,   reg_ids::fpr_f18, reg_ids::fpr_f19,
    reg_ids::fpr_f20,  reg_ids::fpr_f21,   reg_ids::fpr_f22, reg_ids::fpr_f23,
    reg_ids::fpr_f24,  reg_ids::fpr_f25,   reg_ids::fpr_f26, reg_ids::fpr_f27,
    reg_ids::fpr_f28,  reg_ids::fpr_f29,   reg_ids::fpr_f30, reg_ids::fpr_f31,
    reg_ids::fpr_fcsr, LLDB_INVALID_REGNUM};

static_assert(((sizeof g_fpr_regnums_riscv32 /
                sizeof g_fpr_regnums_riscv32[0]) -
               1) == k_num_fpr_registers,
              "g_fpr_regnums_riscv32 has wrong number of register infos");

// Register sets for RISC-V32.
static const lldb_private::RegisterSet g_reg_sets_riscv32[k_num_register_sets] =
    {{"General Purpose Registers", "gpr", k_num_gpr_registers,
      g_gpr_regnums_riscv32},
     {"Floating Point Registers", "fpr", k_num_fpr_registers,
      g_fpr_regnums_riscv32}};

RegisterInfo_riscv32::RegisterInfo_riscv32(
    const lldb_private::ArchSpec &target_arch, lldb_private::Flags opt_regsets)
    : lldb_private::RegisterInfoAndSetInterface(target_arch),
      m_register_info_p(GetRegisterInfoPtr(target_arch)),
      m_register_info_count(GetRegisterInfoCount(target_arch)),
      m_opt_regsets(opt_regsets) {}

uint32_t RegisterInfo_riscv32::GetRegisterCount() const {
  return m_register_info_count;
}

size_t RegisterInfo_riscv32::GetGPRSize() const {
  return sizeof(struct RegisterInfo_riscv32::GPR);
}

size_t RegisterInfo_riscv32::GetFPRSize() const {
  return sizeof(struct RegisterInfo_riscv32::FPR);
}

const lldb_private::RegisterInfo *
RegisterInfo_riscv32::GetRegisterInfo() const {
  return m_register_info_p;
}

size_t RegisterInfo_riscv32::GetRegisterSetCount() const {
  return k_num_register_sets;
}

size_t RegisterInfo_riscv32::GetRegisterSetFromRegisterIndex(
    uint32_t reg_index) const {
  // coverity[unsigned_compare]
  if (reg_index >= reg_ids::gpr_first && reg_index <= reg_ids::gpr_last)
    return eRegsetMaskDefault;
  if (reg_index >= reg_ids::fpr_first && reg_index <= reg_ids::fpr_last)
    return eRegsetMaskFP;
  return LLDB_INVALID_REGNUM;
}

const lldb_private::RegisterSet *
RegisterInfo_riscv32::GetRegisterSet(size_t set_index) const {
  if (set_index < GetRegisterSetCount())
    return &g_reg_sets_riscv32[set_index];
  return nullptr;
}
