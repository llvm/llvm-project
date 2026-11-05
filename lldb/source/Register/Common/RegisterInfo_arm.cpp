//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include "lldb/Register/Arch/lldb-arm-ehframe-register-enums.h"
#include <cstddef>
#include <vector>

#include "lldb/Register/Arch/lldb-arm-register-enums.h"
#include "lldb/Register/Arch/lldb-arm-dwarf-register-enums.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-types.h"
#include "lldb/lldb-private.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>

#include "RegisterInfo_arm.h"

namespace reg_ids = lldb_private::registers::arm::id;
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
                                  reg_ids::dbg_##reg##i##_with_tls },                              \
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
  k_num_gpr_registers = reg_ids::gpr_cpsr - reg_ids::gpr_r0 + 1,
  k_num_fpr_registers = reg_ids::fpu_q15 - reg_ids::fpu_s0 + 1,
  k_num_tls_registers = 1,
  k_num_register_sets_without_tls = 2,
  k_num_register_sets_with_tls = 3
};

// arm general purpose registers.
static const uint32_t g_gpr_regnums_arm[] = {
    reg_ids::gpr_r0,   reg_ids::gpr_r1,    reg_ids::gpr_r2,  reg_ids::gpr_r3,
    reg_ids::gpr_r4,   reg_ids::gpr_r5,    reg_ids::gpr_r6,  reg_ids::gpr_r7,
    reg_ids::gpr_r8,   reg_ids::gpr_r9,    reg_ids::gpr_r10, reg_ids::gpr_r11,
    reg_ids::gpr_r12,  reg_ids::gpr_sp,    reg_ids::gpr_lr,  reg_ids::gpr_pc,
    reg_ids::gpr_cpsr, LLDB_INVALID_REGNUM // register sets need to end with
                                           // this flag
};
static_assert(((sizeof g_gpr_regnums_arm / sizeof g_gpr_regnums_arm[0]) - 1) ==
                  k_num_gpr_registers,
              "g_gpr_regnums_arm has wrong number of register infos");

// arm floating point registers.
static const uint32_t g_fpu_regnums_arm[] = {
    reg_ids::fpu_s0,    reg_ids::fpu_s1,    reg_ids::fpu_s2,  reg_ids::fpu_s3,
    reg_ids::fpu_s4,    reg_ids::fpu_s5,    reg_ids::fpu_s6,  reg_ids::fpu_s7,
    reg_ids::fpu_s8,    reg_ids::fpu_s9,    reg_ids::fpu_s10, reg_ids::fpu_s11,
    reg_ids::fpu_s12,   reg_ids::fpu_s13,   reg_ids::fpu_s14, reg_ids::fpu_s15,
    reg_ids::fpu_s16,   reg_ids::fpu_s17,   reg_ids::fpu_s18, reg_ids::fpu_s19,
    reg_ids::fpu_s20,   reg_ids::fpu_s21,   reg_ids::fpu_s22, reg_ids::fpu_s23,
    reg_ids::fpu_s24,   reg_ids::fpu_s25,   reg_ids::fpu_s26, reg_ids::fpu_s27,
    reg_ids::fpu_s28,   reg_ids::fpu_s29,   reg_ids::fpu_s30, reg_ids::fpu_s31,
    reg_ids::fpu_fpscr, reg_ids::fpu_d0,    reg_ids::fpu_d1,  reg_ids::fpu_d2,
    reg_ids::fpu_d3,    reg_ids::fpu_d4,    reg_ids::fpu_d5,  reg_ids::fpu_d6,
    reg_ids::fpu_d7,    reg_ids::fpu_d8,    reg_ids::fpu_d9,  reg_ids::fpu_d10,
    reg_ids::fpu_d11,   reg_ids::fpu_d12,   reg_ids::fpu_d13, reg_ids::fpu_d14,
    reg_ids::fpu_d15,   reg_ids::fpu_d16,   reg_ids::fpu_d17, reg_ids::fpu_d18,
    reg_ids::fpu_d19,   reg_ids::fpu_d20,   reg_ids::fpu_d21, reg_ids::fpu_d22,
    reg_ids::fpu_d23,   reg_ids::fpu_d24,   reg_ids::fpu_d25, reg_ids::fpu_d26,
    reg_ids::fpu_d27,   reg_ids::fpu_d28,   reg_ids::fpu_d29, reg_ids::fpu_d30,
    reg_ids::fpu_d31,   reg_ids::fpu_q0,    reg_ids::fpu_q1,  reg_ids::fpu_q2,
    reg_ids::fpu_q3,    reg_ids::fpu_q4,    reg_ids::fpu_q5,  reg_ids::fpu_q6,
    reg_ids::fpu_q7,    reg_ids::fpu_q8,    reg_ids::fpu_q9,  reg_ids::fpu_q10,
    reg_ids::fpu_q11,   reg_ids::fpu_q12,   reg_ids::fpu_q13, reg_ids::fpu_q14,
    reg_ids::fpu_q15,   LLDB_INVALID_REGNUM // register sets need to end with
                                            // this flag
};
static_assert(((sizeof g_fpu_regnums_arm / sizeof g_fpu_regnums_arm[0]) - 1) ==
                  k_num_fpr_registers,
              "g_fpu_regnums_arm has wrong number of register infos");

// arm thread local storage registers.
static const uint32_t g_tls_regnums_arm[] = {
    reg_ids::tls_tpidruro,
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
  if (reg_index <= reg_ids::gpr_cpsr)
    return GPRegSet;
  if (reg_index <= reg_ids::fpu_q15)
    return FPRegSet;
  if (reg_index == reg_ids::tls_tpidruro && m_has_tls_reg)
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
