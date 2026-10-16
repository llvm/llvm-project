//===-- RegisterInfoPOSIX_riscv32.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "RegisterInfoPOSIX_riscv32.h"
#include "lldb/Utility/Flags.h"
#include "lldb/lldb-defines.h"
#include "llvm/Support/Compiler.h"

#include <cassert>
#include <stddef.h>

#define GPR_OFFSET(idx) ((idx) * 4 + 0)
#define FPR_OFFSET(idx) ((idx) * 4 + sizeof(RegisterInfoPOSIX_riscv32::GPR))

#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterInfoPOSIX_riscv32::GPR) +                                    \
   sizeof(RegisterInfoPOSIX_riscv32::FPR))

#define DECLARE_REGISTER_INFOS_RISCV32_STRUCT
#include "RegisterInfos_riscv32.h"
#undef DECLARE_REGISTER_INFOS_RISCV32_STRUCT

const lldb_private::RegisterInfo *RegisterInfoPOSIX_riscv32::GetRegisterInfoPtr(
    const lldb_private::ArchSpec &target_arch) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::riscv32:
    return g_register_infos_riscv32_le;
  default:
    assert(false && "Unhandled target architecture.");
    return nullptr;
  }
}

uint32_t RegisterInfoPOSIX_riscv32::GetRegisterInfoCount(
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
  k_num_gpr_registers = gpr_last_riscv - gpr_first_riscv + 1,
  k_num_fpr_registers = fpr_last_riscv - fpr_first_riscv + 1,
  k_num_register_sets = 2
};

// RISC-V32 general purpose registers.
static const uint32_t g_gpr_regnums_riscv32[] = {
    gpr_pc_riscv,  gpr_ra_riscv,       gpr_sp_riscv,  gpr_x3_riscv,
    gpr_x4_riscv,  gpr_x5_riscv,       gpr_x6_riscv,  gpr_x7_riscv,
    gpr_fp_riscv,  gpr_x9_riscv,       gpr_x10_riscv, gpr_x11_riscv,
    gpr_x12_riscv, gpr_x13_riscv,      gpr_x14_riscv, gpr_x15_riscv,
    gpr_x16_riscv, gpr_x17_riscv,      gpr_x18_riscv, gpr_x19_riscv,
    gpr_x20_riscv, gpr_x21_riscv,      gpr_x22_riscv, gpr_x23_riscv,
    gpr_x24_riscv, gpr_x25_riscv,      gpr_x26_riscv, gpr_x27_riscv,
    gpr_x28_riscv, gpr_x29_riscv,      gpr_x30_riscv, gpr_x31_riscv,
    gpr_x0_riscv,  LLDB_INVALID_REGNUM};

static_assert(((sizeof g_gpr_regnums_riscv32 /
                sizeof g_gpr_regnums_riscv32[0]) -
               1) == k_num_gpr_registers,
              "g_gpr_regnums_riscv32 has wrong number of register infos");

// RISC-V32 floating point registers.
static const uint32_t g_fpr_regnums_riscv32[] = {
    fpr_f0_riscv,   fpr_f1_riscv,       fpr_f2_riscv,  fpr_f3_riscv,
    fpr_f4_riscv,   fpr_f5_riscv,       fpr_f6_riscv,  fpr_f7_riscv,
    fpr_f8_riscv,   fpr_f9_riscv,       fpr_f10_riscv, fpr_f11_riscv,
    fpr_f12_riscv,  fpr_f13_riscv,      fpr_f14_riscv, fpr_f15_riscv,
    fpr_f16_riscv,  fpr_f17_riscv,      fpr_f18_riscv, fpr_f19_riscv,
    fpr_f20_riscv,  fpr_f21_riscv,      fpr_f22_riscv, fpr_f23_riscv,
    fpr_f24_riscv,  fpr_f25_riscv,      fpr_f26_riscv, fpr_f27_riscv,
    fpr_f28_riscv,  fpr_f29_riscv,      fpr_f30_riscv, fpr_f31_riscv,
    fpr_fcsr_riscv, LLDB_INVALID_REGNUM};

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

RegisterInfoPOSIX_riscv32::RegisterInfoPOSIX_riscv32(
    const lldb_private::ArchSpec &target_arch, lldb_private::Flags opt_regsets)
    : lldb_private::RegisterInfoAndSetInterface(target_arch),
      m_register_info_p(GetRegisterInfoPtr(target_arch)),
      m_register_info_count(GetRegisterInfoCount(target_arch)),
      m_opt_regsets(opt_regsets) {}

uint32_t RegisterInfoPOSIX_riscv32::GetRegisterCount() const {
  return m_register_info_count;
}

size_t RegisterInfoPOSIX_riscv32::GetGPRSize() const {
  return sizeof(struct RegisterInfoPOSIX_riscv32::GPR);
}

size_t RegisterInfoPOSIX_riscv32::GetFPRSize() const {
  return sizeof(struct RegisterInfoPOSIX_riscv32::FPR);
}

const lldb_private::RegisterInfo *
RegisterInfoPOSIX_riscv32::GetRegisterInfo() const {
  return m_register_info_p;
}

size_t RegisterInfoPOSIX_riscv32::GetRegisterSetCount() const {
  return k_num_register_sets;
}

size_t RegisterInfoPOSIX_riscv32::GetRegisterSetFromRegisterIndex(
    uint32_t reg_index) const {
  // coverity[unsigned_compare]
  if (reg_index >= gpr_first_riscv && reg_index <= gpr_last_riscv)
    return eRegsetMaskDefault;
  if (reg_index >= fpr_first_riscv && reg_index <= fpr_last_riscv)
    return eRegsetMaskFP;
  return LLDB_INVALID_REGNUM;
}

const lldb_private::RegisterSet *
RegisterInfoPOSIX_riscv32::GetRegisterSet(size_t set_index) const {
  if (set_index < GetRegisterSetCount())
    return &g_reg_sets_riscv32[set_index];
  return nullptr;
}
