//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include "lldb/Register/Arch/lldb-loongarch-ehframe-register-enums.h"

#include <lldb/Utility/Flags.h>
#include <stddef.h>

#include "lldb/Register/Arch/lldb-loongarch-register-enums.h"
#include "lldb/Register/Arch/lldb-loongarch-dwarf-register-enums.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-types.h"
#include "lldb/lldb-private.h"
#include "llvm/Support/Compiler.h"
#include <cstddef>
#include <cstdint>

#include "RegisterInfo_loongarch64.h"

#define GPR_OFFSET(idx) ((idx)*8 + 0)
#define FPR_OFFSET(idx) ((idx)*8 + sizeof(RegisterInfo_loongarch64::GPR))
#define FCC_OFFSET(idx) ((idx)*1 + 32 * 8 + sizeof(RegisterInfo_loongarch64::GPR))
#define FCSR_OFFSET (8 * 1 + 32 * 8 + sizeof(RegisterInfo_loongarch64::GPR))
#define LSX_OFFSET(idx)                                                        \
  ((idx) * 16 + sizeof(RegisterInfo_loongarch64::GPR) +                   \
   sizeof(RegisterInfo_loongarch64::FPR))
#define LASX_OFFSET(idx)                                                       \
  ((idx) * 32 + sizeof(RegisterInfo_loongarch64::GPR) +                   \
   sizeof(RegisterInfo_loongarch64::FPR) +                                \
   sizeof(RegisterInfo_loongarch64::LSX))

#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterInfo_loongarch64::GPR) +                                \
   sizeof(RegisterInfo_loongarch64::FPR) +                                \
   sizeof(RegisterInfo_loongarch64::LSX) +                                \
   sizeof(RegisterInfo_loongarch64::LASX))


namespace reg_ids = lldb_private::registers::loongarch::id;
using namespace lldb_private;

#define DECLARE_REGISTER_INFOS_LOONGARCH64_STRUCT
#include "Register/Arch/RegisterInfos_loongarch64.h"
#undef DECLARE_REGISTER_INFOS_LOONGARCH64_STRUCT

const lldb_private::RegisterInfo *
RegisterInfo_loongarch64::GetRegisterInfoPtr(
    const lldb_private::ArchSpec &target_arch) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::loongarch64:
    return g_register_infos_loongarch64;
  default:
    assert(false && "Unhandled target architecture.");
    return nullptr;
  }
}

uint32_t RegisterInfo_loongarch64::GetRegisterInfoCount(
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
  k_num_gpr_registers = reg_ids::gpr_last - reg_ids::gpr_first + 1,
  k_num_fpr_registers = reg_ids::fpr_last - reg_ids::fpr_first + 1,
  k_num_lsx_registers = reg_ids::lsx_last - reg_ids::lsx_first + 1,
  k_num_lasx_registers = reg_ids::lasx_last - reg_ids::lasx_first + 1,
  k_num_register_sets = 4
};

// LoongArch64 general purpose registers.
static const uint32_t g_gpr_regnums_loongarch64[] = {
    reg_ids::gpr_r0,        reg_ids::gpr_r1,        reg_ids::gpr_r2,
    reg_ids::gpr_r3,        reg_ids::gpr_r4,        reg_ids::gpr_r5,
    reg_ids::gpr_r6,        reg_ids::gpr_r7,        reg_ids::gpr_r8,
    reg_ids::gpr_r9,        reg_ids::gpr_r10,       reg_ids::gpr_r11,
    reg_ids::gpr_r12,       reg_ids::gpr_r13,       reg_ids::gpr_r14,
    reg_ids::gpr_r15,       reg_ids::gpr_r16,       reg_ids::gpr_r17,
    reg_ids::gpr_r18,       reg_ids::gpr_r19,       reg_ids::gpr_r20,
    reg_ids::gpr_r21,       reg_ids::gpr_r22,       reg_ids::gpr_r23,
    reg_ids::gpr_r24,       reg_ids::gpr_r25,       reg_ids::gpr_r26,
    reg_ids::gpr_r27,       reg_ids::gpr_r28,       reg_ids::gpr_r29,
    reg_ids::gpr_r30,       reg_ids::gpr_r31,       reg_ids::gpr_orig_a0,
    reg_ids::gpr_pc,        reg_ids::gpr_badv,      reg_ids::gpr_reserved0,
    reg_ids::gpr_reserved1, reg_ids::gpr_reserved2, reg_ids::gpr_reserved3,
    reg_ids::gpr_reserved4, reg_ids::gpr_reserved5, reg_ids::gpr_reserved6,
    reg_ids::gpr_reserved7, reg_ids::gpr_reserved8, reg_ids::gpr_reserved9,
    LLDB_INVALID_REGNUM};

static_assert(((sizeof g_gpr_regnums_loongarch64 /
                sizeof g_gpr_regnums_loongarch64[0]) -
               1) == k_num_gpr_registers,
              "g_gpr_regnums_loongarch64 has wrong number of register infos");

// LoongArch64 floating point registers.
static const uint32_t g_fpr_regnums_loongarch64[] = {
    reg_ids::fpr_f0,   reg_ids::fpr_f1,    reg_ids::fpr_f2,   reg_ids::fpr_f3,
    reg_ids::fpr_f4,   reg_ids::fpr_f5,    reg_ids::fpr_f6,   reg_ids::fpr_f7,
    reg_ids::fpr_f8,   reg_ids::fpr_f9,    reg_ids::fpr_f10,  reg_ids::fpr_f11,
    reg_ids::fpr_f12,  reg_ids::fpr_f13,   reg_ids::fpr_f14,  reg_ids::fpr_f15,
    reg_ids::fpr_f16,  reg_ids::fpr_f17,   reg_ids::fpr_f18,  reg_ids::fpr_f19,
    reg_ids::fpr_f20,  reg_ids::fpr_f21,   reg_ids::fpr_f22,  reg_ids::fpr_f23,
    reg_ids::fpr_f24,  reg_ids::fpr_f25,   reg_ids::fpr_f26,  reg_ids::fpr_f27,
    reg_ids::fpr_f28,  reg_ids::fpr_f29,   reg_ids::fpr_f30,  reg_ids::fpr_f31,
    reg_ids::fpr_fcc0, reg_ids::fpr_fcc1,  reg_ids::fpr_fcc2, reg_ids::fpr_fcc3,
    reg_ids::fpr_fcc4, reg_ids::fpr_fcc5,  reg_ids::fpr_fcc6, reg_ids::fpr_fcc7,
    reg_ids::fpr_fcsr, LLDB_INVALID_REGNUM};

static_assert(((sizeof g_fpr_regnums_loongarch64 /
                sizeof g_fpr_regnums_loongarch64[0]) -
               1) == k_num_fpr_registers,
              "g_fpr_regnums_loongarch64 has wrong number of register infos");

// LoongArch64 lsx vector registers.
static const uint32_t g_lsx_regnums_loongarch64[] = {
    reg_ids::lsx_vr0,   reg_ids::lsx_vr1,  reg_ids::lsx_vr2,  reg_ids::lsx_vr3,
    reg_ids::lsx_vr4,   reg_ids::lsx_vr5,  reg_ids::lsx_vr6,  reg_ids::lsx_vr7,
    reg_ids::lsx_vr8,   reg_ids::lsx_vr9,  reg_ids::lsx_vr10, reg_ids::lsx_vr11,
    reg_ids::lsx_vr12,  reg_ids::lsx_vr13, reg_ids::lsx_vr14, reg_ids::lsx_vr15,
    reg_ids::lsx_vr16,  reg_ids::lsx_vr17, reg_ids::lsx_vr18, reg_ids::lsx_vr19,
    reg_ids::lsx_vr20,  reg_ids::lsx_vr21, reg_ids::lsx_vr22, reg_ids::lsx_vr23,
    reg_ids::lsx_vr24,  reg_ids::lsx_vr25, reg_ids::lsx_vr26, reg_ids::lsx_vr27,
    reg_ids::lsx_vr28,  reg_ids::lsx_vr29, reg_ids::lsx_vr30, reg_ids::lsx_vr31,
    LLDB_INVALID_REGNUM};

static_assert(((sizeof g_lsx_regnums_loongarch64 /
                sizeof g_lsx_regnums_loongarch64[0]) -
               1) == k_num_lsx_registers,
              "g_lsx_regnums_loongarch64 has wrong number of register infos");

// LoongArch64 lasx vector registers.
static const uint32_t g_lasx_regnums_loongarch64[] = {
    reg_ids::lasx_xr0,  reg_ids::lasx_xr1,  reg_ids::lasx_xr2,
    reg_ids::lasx_xr3,  reg_ids::lasx_xr4,  reg_ids::lasx_xr5,
    reg_ids::lasx_xr6,  reg_ids::lasx_xr7,  reg_ids::lasx_xr8,
    reg_ids::lasx_xr9,  reg_ids::lasx_xr10, reg_ids::lasx_xr11,
    reg_ids::lasx_xr12, reg_ids::lasx_xr13, reg_ids::lasx_xr14,
    reg_ids::lasx_xr15, reg_ids::lasx_xr16, reg_ids::lasx_xr17,
    reg_ids::lasx_xr18, reg_ids::lasx_xr19, reg_ids::lasx_xr20,
    reg_ids::lasx_xr21, reg_ids::lasx_xr22, reg_ids::lasx_xr23,
    reg_ids::lasx_xr24, reg_ids::lasx_xr25, reg_ids::lasx_xr26,
    reg_ids::lasx_xr27, reg_ids::lasx_xr28, reg_ids::lasx_xr29,
    reg_ids::lasx_xr30, reg_ids::lasx_xr31, LLDB_INVALID_REGNUM};

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

RegisterInfo_loongarch64::RegisterInfo_loongarch64(
    const lldb_private::ArchSpec &target_arch, lldb_private::Flags flags)
    : lldb_private::RegisterInfoAndSetInterface(target_arch),
      m_register_info_p(GetRegisterInfoPtr(target_arch)),
      m_register_info_count(GetRegisterInfoCount(target_arch)) {}

uint32_t RegisterInfo_loongarch64::GetRegisterCount() const {
  return m_register_info_count;
}

size_t RegisterInfo_loongarch64::GetGPRSize() const {
  return sizeof(struct RegisterInfo_loongarch64::GPR);
}

size_t RegisterInfo_loongarch64::GetFPRSize() const {
  return sizeof(struct RegisterInfo_loongarch64::FPR);
}

const lldb_private::RegisterInfo *
RegisterInfo_loongarch64::GetRegisterInfo() const {
  return m_register_info_p;
}

size_t RegisterInfo_loongarch64::GetRegisterSetCount() const {
  return k_num_register_sets;
}

size_t RegisterInfo_loongarch64::GetRegisterSetFromRegisterIndex(
    uint32_t reg_index) const {
  // coverity[unsigned_compare]
  if (reg_index >= reg_ids::gpr_first && reg_index <= reg_ids::gpr_last)
    return GPRegSet;
  if (reg_index >= reg_ids::fpr_first && reg_index <= reg_ids::fpr_last)
    return FPRegSet;
  if (reg_index >= reg_ids::lsx_first && reg_index <= reg_ids::lsx_last)
    return LSXRegSet;
  if (reg_index >= reg_ids::lasx_first && reg_index <= reg_ids::lasx_last)
    return LASXRegSet;
  return LLDB_INVALID_REGNUM;
}

const lldb_private::RegisterSet *
RegisterInfo_loongarch64::GetRegisterSet(size_t set_index) const {
  if (set_index < GetRegisterSetCount())
    return &g_reg_sets_loongarch64[set_index];
  return nullptr;
}
