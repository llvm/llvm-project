//===-- RegisterInfoPOSIX_loongarch64.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include <cassert>
#include <lldb/Utility/Flags.h>
#include <stddef.h>

#include "lldb/lldb-defines.h"
#include "llvm/Support/Compiler.h"

#include "RegisterInfoPOSIX_loongarch64.h"

#define GPR_OFFSET(idx) ((idx)*8 + 0)
#define FPR_OFFSET(idx) ((idx)*8 + sizeof(RegisterInfoPOSIX_loongarch64::GPR))
#define FCC_OFFSET(idx) ((idx)*1 + 32 * 8 + sizeof(RegisterInfoPOSIX_loongarch64::GPR))
#define FCSR_OFFSET (8 * 1 + 32 * 8 + sizeof(RegisterInfoPOSIX_loongarch64::GPR))
#define LSX_OFFSET(idx)                                                        \
  ((idx) * 16 + sizeof(RegisterInfoPOSIX_loongarch64::GPR) +                   \
   sizeof(RegisterInfoPOSIX_loongarch64::FPR))
#define LASX_OFFSET(idx)                                                       \
  ((idx) * 32 + sizeof(RegisterInfoPOSIX_loongarch64::GPR) +                   \
   sizeof(RegisterInfoPOSIX_loongarch64::FPR) +                                \
   sizeof(RegisterInfoPOSIX_loongarch64::LSX))

#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterInfoPOSIX_loongarch64::GPR) +                                \
   sizeof(RegisterInfoPOSIX_loongarch64::FPR) +                                \
   sizeof(RegisterInfoPOSIX_loongarch64::LSX) +                                \
   sizeof(RegisterInfoPOSIX_loongarch64::LASX))

#define DECLARE_REGISTER_INFOS_LOONGARCH64_STRUCT
#include "RegisterInfos_loongarch64.h"
#undef DECLARE_REGISTER_INFOS_LOONGARCH64_STRUCT

const lldb_private::RegisterInfo *
RegisterInfoPOSIX_loongarch64::GetRegisterInfoPtr(
    const lldb_private::ArchSpec &target_arch) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::loongarch64:
    return g_register_infos_loongarch64;
  default:
    assert(false && "Unhandled target architecture.");
    return nullptr;
  }
}

uint32_t RegisterInfoPOSIX_loongarch64::GetRegisterInfoCount(
    const lldb_private::ArchSpec &target_arch) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::loongarch64:
    return static_cast<uint32_t>(sizeof(g_register_infos_loongarch64) /
                                 sizeof(g_register_infos_loongarch64[0]));
  default:
    assert(false && "Unhandled target architecture.");
    return 0;
  }
}

// Number of register sets provided by this context.
enum {
  k_num_gpr_registers = gpr_last_loongarch - gpr_first_loongarch + 1,
  k_num_fpr_registers = fpr_last_loongarch - fpr_first_loongarch + 1,
  k_num_lsx_registers = lsx_last_loongarch - lsx_first_loongarch + 1,
  k_num_lasx_registers = lasx_last_loongarch - lasx_first_loongarch + 1,
  k_num_register_sets = 4
};

// LoongArch64 general purpose registers.
static const uint32_t g_gpr_regnums_loongarch64[] = {
    gpr_r0_loongarch,        gpr_r1_loongarch,        gpr_r2_loongarch,
    gpr_r3_loongarch,        gpr_r4_loongarch,        gpr_r5_loongarch,
    gpr_r6_loongarch,        gpr_r7_loongarch,        gpr_r8_loongarch,
    gpr_r9_loongarch,        gpr_r10_loongarch,       gpr_r11_loongarch,
    gpr_r12_loongarch,       gpr_r13_loongarch,       gpr_r14_loongarch,
    gpr_r15_loongarch,       gpr_r16_loongarch,       gpr_r17_loongarch,
    gpr_r18_loongarch,       gpr_r19_loongarch,       gpr_r20_loongarch,
    gpr_r21_loongarch,       gpr_r22_loongarch,       gpr_r23_loongarch,
    gpr_r24_loongarch,       gpr_r25_loongarch,       gpr_r26_loongarch,
    gpr_r27_loongarch,       gpr_r28_loongarch,       gpr_r29_loongarch,
    gpr_r30_loongarch,       gpr_r31_loongarch,       gpr_orig_a0_loongarch,
    gpr_pc_loongarch,        gpr_badv_loongarch,      gpr_reserved0_loongarch,
    gpr_reserved1_loongarch, gpr_reserved2_loongarch, gpr_reserved3_loongarch,
    gpr_reserved4_loongarch, gpr_reserved5_loongarch, gpr_reserved6_loongarch,
    gpr_reserved7_loongarch, gpr_reserved8_loongarch, gpr_reserved9_loongarch,
    LLDB_INVALID_REGNUM};

static_assert(((sizeof g_gpr_regnums_loongarch64 /
                sizeof g_gpr_regnums_loongarch64[0]) -
               1) == k_num_gpr_registers,
              "g_gpr_regnums_loongarch64 has wrong number of register infos");

// LoongArch64 floating point registers.
static const uint32_t g_fpr_regnums_loongarch64[] = {
    fpr_f0_loongarch,   fpr_f1_loongarch,   fpr_f2_loongarch,
    fpr_f3_loongarch,   fpr_f4_loongarch,   fpr_f5_loongarch,
    fpr_f6_loongarch,   fpr_f7_loongarch,   fpr_f8_loongarch,
    fpr_f9_loongarch,   fpr_f10_loongarch,  fpr_f11_loongarch,
    fpr_f12_loongarch,  fpr_f13_loongarch,  fpr_f14_loongarch,
    fpr_f15_loongarch,  fpr_f16_loongarch,  fpr_f17_loongarch,
    fpr_f18_loongarch,  fpr_f19_loongarch,  fpr_f20_loongarch,
    fpr_f21_loongarch,  fpr_f22_loongarch,  fpr_f23_loongarch,
    fpr_f24_loongarch,  fpr_f25_loongarch,  fpr_f26_loongarch,
    fpr_f27_loongarch,  fpr_f28_loongarch,  fpr_f29_loongarch,
    fpr_f30_loongarch,  fpr_f31_loongarch,  fpr_fcc0_loongarch,
    fpr_fcc1_loongarch, fpr_fcc2_loongarch, fpr_fcc3_loongarch,
    fpr_fcc4_loongarch, fpr_fcc5_loongarch, fpr_fcc6_loongarch,
    fpr_fcc7_loongarch, fpr_fcsr_loongarch, LLDB_INVALID_REGNUM};

static_assert(((sizeof g_fpr_regnums_loongarch64 /
                sizeof g_fpr_regnums_loongarch64[0]) -
               1) == k_num_fpr_registers,
              "g_fpr_regnums_loongarch64 has wrong number of register infos");

// LoongArch64 lsx vector registers.
static const uint32_t g_lsx_regnums_loongarch64[] = {
    lsx_vr0_loongarch,  lsx_vr1_loongarch,  lsx_vr2_loongarch,
    lsx_vr3_loongarch,  lsx_vr4_loongarch,  lsx_vr5_loongarch,
    lsx_vr6_loongarch,  lsx_vr7_loongarch,  lsx_vr8_loongarch,
    lsx_vr9_loongarch,  lsx_vr10_loongarch, lsx_vr11_loongarch,
    lsx_vr12_loongarch, lsx_vr13_loongarch, lsx_vr14_loongarch,
    lsx_vr15_loongarch, lsx_vr16_loongarch, lsx_vr17_loongarch,
    lsx_vr18_loongarch, lsx_vr19_loongarch, lsx_vr20_loongarch,
    lsx_vr21_loongarch, lsx_vr22_loongarch, lsx_vr23_loongarch,
    lsx_vr24_loongarch, lsx_vr25_loongarch, lsx_vr26_loongarch,
    lsx_vr27_loongarch, lsx_vr28_loongarch, lsx_vr29_loongarch,
    lsx_vr30_loongarch, lsx_vr31_loongarch, LLDB_INVALID_REGNUM};

static_assert(((sizeof g_lsx_regnums_loongarch64 /
                sizeof g_lsx_regnums_loongarch64[0]) -
               1) == k_num_lsx_registers,
              "g_lsx_regnums_loongarch64 has wrong number of register infos");

// LoongArch64 lasx vector registers.
static const uint32_t g_lasx_regnums_loongarch64[] = {
    lasx_xr0_loongarch,  lasx_xr1_loongarch,  lasx_xr2_loongarch,
    lasx_xr3_loongarch,  lasx_xr4_loongarch,  lasx_xr5_loongarch,
    lasx_xr6_loongarch,  lasx_xr7_loongarch,  lasx_xr8_loongarch,
    lasx_xr9_loongarch,  lasx_xr10_loongarch, lasx_xr11_loongarch,
    lasx_xr12_loongarch, lasx_xr13_loongarch, lasx_xr14_loongarch,
    lasx_xr15_loongarch, lasx_xr16_loongarch, lasx_xr17_loongarch,
    lasx_xr18_loongarch, lasx_xr19_loongarch, lasx_xr20_loongarch,
    lasx_xr21_loongarch, lasx_xr22_loongarch, lasx_xr23_loongarch,
    lasx_xr24_loongarch, lasx_xr25_loongarch, lasx_xr26_loongarch,
    lasx_xr27_loongarch, lasx_xr28_loongarch, lasx_xr29_loongarch,
    lasx_xr30_loongarch, lasx_xr31_loongarch, LLDB_INVALID_REGNUM};

static_assert(((sizeof g_lasx_regnums_loongarch64 /
                sizeof g_lasx_regnums_loongarch64[0]) -
               1) == k_num_lasx_registers,
              "g_lasx_regnums_loongarch64 has wrong number of register infos");

// Register sets for LoongArch64.
static const lldb_private::RegisterSet
    g_reg_sets_loongarch64[k_num_register_sets] = {
        {"General Purpose Registers", "gpr", k_num_gpr_registers,
         g_gpr_regnums_loongarch64},
        {"Floating Point Registers", "fpr", k_num_fpr_registers,
         g_fpr_regnums_loongarch64},
        {"LSX Vector Registers", "lsx", k_num_lsx_registers,
         g_lsx_regnums_loongarch64},
        {"LASX Vector Registers", "lasx", k_num_lasx_registers,
         g_lasx_regnums_loongarch64}};

RegisterInfoPOSIX_loongarch64::RegisterInfoPOSIX_loongarch64(
    const lldb_private::ArchSpec &target_arch, lldb_private::Flags flags)
    : lldb_private::RegisterInfoAndSetInterface(target_arch),
      m_register_info_p(GetRegisterInfoPtr(target_arch)),
      m_register_info_count(GetRegisterInfoCount(target_arch)) {}

uint32_t RegisterInfoPOSIX_loongarch64::GetRegisterCount() const {
  return m_register_info_count;
}

size_t RegisterInfoPOSIX_loongarch64::GetGPRSize() const {
  return sizeof(struct RegisterInfoPOSIX_loongarch64::GPR);
}

size_t RegisterInfoPOSIX_loongarch64::GetFPRSize() const {
  return sizeof(struct RegisterInfoPOSIX_loongarch64::FPR);
}

const lldb_private::RegisterInfo *
RegisterInfoPOSIX_loongarch64::GetRegisterInfo() const {
  return m_register_info_p;
}

size_t RegisterInfoPOSIX_loongarch64::GetRegisterSetCount() const {
  return k_num_register_sets;
}

size_t RegisterInfoPOSIX_loongarch64::GetRegisterSetFromRegisterIndex(
    uint32_t reg_index) const {
  // coverity[unsigned_compare]
  if (reg_index >= gpr_first_loongarch && reg_index <= gpr_last_loongarch)
    return GPRegSet;
  if (reg_index >= fpr_first_loongarch && reg_index <= fpr_last_loongarch)
    return FPRegSet;
  if (reg_index >= lsx_first_loongarch && reg_index <= lsx_last_loongarch)
    return LSXRegSet;
  if (reg_index >= lasx_first_loongarch && reg_index <= lasx_last_loongarch)
    return LASXRegSet;
  return LLDB_INVALID_REGNUM;
}

const lldb_private::RegisterSet *
RegisterInfoPOSIX_loongarch64::GetRegisterSet(size_t set_index) const {
  if (set_index < GetRegisterSetCount())
    return &g_reg_sets_loongarch64[set_index];
  return nullptr;
}
