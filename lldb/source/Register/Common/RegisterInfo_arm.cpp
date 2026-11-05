//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>

#include "Register/Arch/lldb-arm-register-enums.h"
#include <cstddef>
#include <vector>

#include "lldb/lldb-defines.h"
#include "llvm/Support/Compiler.h"

#include "RegisterInfo_arm.h"

using namespace lldb;
using namespace lldb_private;

// Based on RegisterContextDarwin_arm.cpp
#define GPR_OFFSET(idx) ((idx)*4)
#define FPU_OFFSET(idx) ((idx)*4 + sizeof(RegisterInfo_arm::GPR))
#define FPSCR_OFFSET                                                           \
  (LLVM_EXTENSION offsetof(RegisterInfo_arm::FPU, fpscr) +                \
   sizeof(RegisterInfo_arm::GPR))
#define TLS_OFFSET                                                             \
  (sizeof(RegisterInfo_arm::GPR) + sizeof(RegisterInfo_arm::FPU))
#define EXC_OFFSET(idx)                                                        \
  ((idx) * 4 + sizeof(RegisterInfo_arm::GPR) +                            \
   sizeof(RegisterInfo_arm::FPU) + sizeof(RegisterInfo_arm::TLS))
#define DBG_OFFSET(reg)                                                        \
  ((LLVM_EXTENSION offsetof(RegisterInfo_arm::DBG, reg) +                 \
    sizeof(RegisterInfo_arm::GPR) + sizeof(RegisterInfo_arm::FPU) +  \
    sizeof(RegisterInfo_arm::TLS) + sizeof(RegisterInfo_arm::EXC)))

#define DEFINE_DBG(reg, i)                                                     \
  #reg, NULL, sizeof(((RegisterInfo_arm::DBG *) NULL)->reg[i]),           \
                      DBG_OFFSET(reg[i]), eEncodingUint, eFormatHex,           \
                                 {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    \
                                  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    \
                                  dbg_##reg##i##_arm_with_tls },                              \
                                  NULL, NULL, NULL,
#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterInfo_arm::GPR) + sizeof(RegisterInfo_arm::FPU) +   \
   sizeof(RegisterInfo_arm::EXC))

// Include RegisterInfos_arm to declare our g_register_infos_arm structure.
#define DECLARE_REGISTER_INFOS_ARM_STRUCT
#include "Register/Arch/RegisterInfos_arm.h"
#undef DECLARE_REGISTER_INFOS_ARM_STRUCT

static const lldb_private::RegisterInfo *
GetRegisterInfoPtr(const lldb_private::ArchSpec &target_arch) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::arm:
    return g_register_infos_arm;
  default:
    assert(false && "Unhandled target architecture.");
    return nullptr;
  }
}

static uint32_t
GetRegisterInfoCount(const lldb_private::ArchSpec &target_arch) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::arm:
    return static_cast<uint32_t>(sizeof(g_register_infos_arm) /
                                 sizeof(g_register_infos_arm[0]));
  default:
    assert(false && "Unhandled target architecture.");
    return 0;
  }
}

// Number of register sets provided by this context.
enum {
  k_num_gpr_registers = gpr_cpsr_arm - gpr_r0_arm + 1,
  k_num_fpr_registers = fpu_q15_arm - fpu_s0_arm + 1,
  k_num_tls_registers = 1,
  k_num_register_sets_without_tls = 2,
  k_num_register_sets_with_tls = 3
};

// arm general purpose registers.
static const uint32_t g_gpr_regnums_arm[] = {
    gpr_r0_arm,   gpr_r1_arm,
    gpr_r2_arm,   gpr_r3_arm,
    gpr_r4_arm,   gpr_r5_arm,
    gpr_r6_arm,   gpr_r7_arm,
    gpr_r8_arm,   gpr_r9_arm,
    gpr_r10_arm,  gpr_r11_arm,
    gpr_r12_arm,  gpr_sp_arm,
    gpr_lr_arm,   gpr_pc_arm,
    gpr_cpsr_arm, LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert(((sizeof g_gpr_regnums_arm / sizeof g_gpr_regnums_arm[0]) - 1) ==
                  k_num_gpr_registers,
              "g_gpr_regnums_arm has wrong number of register infos");

// arm floating point registers.
static const uint32_t g_fpu_regnums_arm[] = {
    fpu_s0_arm,    fpu_s1_arm,
    fpu_s2_arm,    fpu_s3_arm,
    fpu_s4_arm,    fpu_s5_arm,
    fpu_s6_arm,    fpu_s7_arm,
    fpu_s8_arm,    fpu_s9_arm,
    fpu_s10_arm,   fpu_s11_arm,
    fpu_s12_arm,   fpu_s13_arm,
    fpu_s14_arm,   fpu_s15_arm,
    fpu_s16_arm,   fpu_s17_arm,
    fpu_s18_arm,   fpu_s19_arm,
    fpu_s20_arm,   fpu_s21_arm,
    fpu_s22_arm,   fpu_s23_arm,
    fpu_s24_arm,   fpu_s25_arm,
    fpu_s26_arm,   fpu_s27_arm,
    fpu_s28_arm,   fpu_s29_arm,
    fpu_s30_arm,   fpu_s31_arm,
    fpu_fpscr_arm, fpu_d0_arm,
    fpu_d1_arm,    fpu_d2_arm,
    fpu_d3_arm,    fpu_d4_arm,
    fpu_d5_arm,    fpu_d6_arm,
    fpu_d7_arm,    fpu_d8_arm,
    fpu_d9_arm,    fpu_d10_arm,
    fpu_d11_arm,   fpu_d12_arm,
    fpu_d13_arm,   fpu_d14_arm,
    fpu_d15_arm,   fpu_d16_arm,
    fpu_d17_arm,   fpu_d18_arm,
    fpu_d19_arm,   fpu_d20_arm,
    fpu_d21_arm,   fpu_d22_arm,
    fpu_d23_arm,   fpu_d24_arm,
    fpu_d25_arm,   fpu_d26_arm,
    fpu_d27_arm,   fpu_d28_arm,
    fpu_d29_arm,   fpu_d30_arm,
    fpu_d31_arm,   fpu_q0_arm,
    fpu_q1_arm,    fpu_q2_arm,
    fpu_q3_arm,    fpu_q4_arm,
    fpu_q5_arm,    fpu_q6_arm,
    fpu_q7_arm,    fpu_q8_arm,
    fpu_q9_arm,    fpu_q10_arm,
    fpu_q11_arm,   fpu_q12_arm,
    fpu_q13_arm,   fpu_q14_arm,
    fpu_q15_arm,   LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert(((sizeof g_fpu_regnums_arm / sizeof g_fpu_regnums_arm[0]) - 1) ==
                  k_num_fpr_registers,
              "g_fpu_regnums_arm has wrong number of register infos");

// arm thread local storage registers.
static const uint32_t g_tls_regnums_arm[] = {
    tls_tpidruro_arm,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert(((sizeof g_tls_regnums_arm / sizeof g_tls_regnums_arm[0]) - 1) ==
                  k_num_tls_registers,
              "g_tls_regnums_arm has wrong number of register infos");

// Register sets for arm.
static const RegisterSet g_reg_sets_arm[k_num_register_sets_with_tls] = {
    {"General Purpose Registers", "gpr", k_num_gpr_registers,
     g_gpr_regnums_arm},
    {"Floating Point Registers", "fpu", k_num_fpr_registers, g_fpu_regnums_arm},
    {"Thread Local Storage Registers", "tls", k_num_tls_registers,
     g_tls_regnums_arm}};

RegisterInfo_arm::RegisterInfo_arm(
    const lldb_private::ArchSpec &target_arch, bool has_tls_reg)
    : lldb_private::RegisterInfoAndSetInterface(target_arch),
      m_register_info_p(GetRegisterInfoPtr(target_arch)),
      m_register_info_count(GetRegisterInfoCount(target_arch)),
      m_has_tls_reg(has_tls_reg) {}

size_t RegisterInfo_arm::GetGPRSize() const {
  return sizeof(struct RegisterInfo_arm::GPR);
}

size_t RegisterInfo_arm::GetFPRSize() const {
  return sizeof(struct RegisterInfo_arm::FPU);
}

const lldb_private::RegisterInfo *
RegisterInfo_arm::GetRegisterInfo() const {
  return m_register_info_p;
}

size_t RegisterInfo_arm::GetRegisterSetCount() const {
  return m_has_tls_reg ? k_num_register_sets_with_tls
                       : k_num_register_sets_without_tls;
}

size_t RegisterInfo_arm::GetRegisterSetFromRegisterIndex(
    uint32_t reg_index) const {
  if (reg_index <= gpr_cpsr_arm)
    return GPRegSet;
  if (reg_index <= fpu_q15_arm)
    return FPRegSet;
  if (reg_index == tls_tpidruro_arm && m_has_tls_reg)
    return TLSRegSet;
  return LLDB_INVALID_REGNUM;
}

const lldb_private::RegisterSet *
RegisterInfo_arm::GetRegisterSet(size_t set_index) const {
  if (set_index < GetRegisterSetCount())
    return &g_reg_sets_arm[set_index];
  return nullptr;
}

uint32_t RegisterInfo_arm::GetRegisterCount() const {
  return m_register_info_count;
}
