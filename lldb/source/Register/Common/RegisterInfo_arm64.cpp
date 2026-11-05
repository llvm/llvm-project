//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include "lldb/Register/Arch/lldb-arm64-ehframe-register-enums.h"
#include <cstddef>
#include <vector>

#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/Arch/lldb-arm64-dwarf-register-enums.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-types.h"
#include "lldb/lldb-private.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>

#include "RegisterInfo_arm64.h"

namespace reg_ids = lldb_private::registers::arm64::id;
using namespace lldb_private;

// Based on RegisterContextDarwin_arm64.cpp
#define GPR_OFFSET(idx) ((idx)*8)
#define GPR_OFFSET_NAME(reg)                                                   \
  (LLVM_EXTENSION offsetof(RegisterInfo_arm64::GPR, reg))

#define FPU_OFFSET(idx) ((idx)*16 + sizeof(RegisterInfo_arm64::GPR))
#define FPU_OFFSET_NAME(reg)                                                   \
  (LLVM_EXTENSION offsetof(RegisterInfo_arm64::FPU, reg) +                \
   sizeof(RegisterInfo_arm64::GPR))

// This information is based on AArch64 with SVE architecture reference manual.
// AArch64 with SVE has 32 Z and 16 P vector registers. There is also an FFR
// (First Fault) register and a VG (Vector Granule) pseudo register.

// SVE 16-byte quad word is the basic unit of expansion in vector length.
#define SVE_QUAD_WORD_BYTES 16

// Vector length is the multiplier which decides the no of quad words,
// (multiples of 128-bits or 16-bytes) present in a Z register. Vector length
// is decided during execution and can change at runtime. SVE AArch64 register
// infos have modes one for each valid value of vector length. A change in
// vector length requires register context to update sizes of SVE Z, P and FFR.
// Also register context needs to update byte offsets of all registers affected
// by the change in vector length.
#define SVE_REGS_DEFAULT_OFFSET_LINUX sizeof(RegisterInfo_arm64::GPR)

#define SVE_OFFSET_VG SVE_REGS_DEFAULT_OFFSET_LINUX

#define EXC_OFFSET_NAME(reg)                                                   \
  (LLVM_EXTENSION offsetof(RegisterInfo_arm64::EXC, reg) +                \
   sizeof(RegisterInfo_arm64::GPR) +                                      \
   sizeof(RegisterInfo_arm64::FPU))
#define DBG_OFFSET_NAME(reg)                                                   \
  (LLVM_EXTENSION offsetof(RegisterInfo_arm64::DBG, reg) +                \
   sizeof(RegisterInfo_arm64::GPR) +                                      \
   sizeof(RegisterInfo_arm64::FPU) +                                      \
   sizeof(RegisterInfo_arm64::EXC))

#define DEFINE_DBG(reg, i)                                                     \
  #reg, NULL,                                                                  \
      sizeof(((RegisterInfo_arm64::DBG *) NULL)->reg[i]),                 \
              DBG_OFFSET_NAME(reg[i]), lldb::eEncodingUint, lldb::eFormatHex,  \
                              {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,       \
                               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,       \
                               reg_ids::dbg_##reg##i },                                 \
                               NULL, NULL, NULL,
#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterInfo_arm64::GPR) +                                      \
   sizeof(RegisterInfo_arm64::FPU) +                                      \
   sizeof(RegisterInfo_arm64::EXC))

// Include RegisterInfos_arm64 to declare our g_register_infos_arm64 structure.
#define LLDB_REGISTER_INFOS_ARM64_KEEP_MACROS
#define DECLARE_REGISTER_INFOS_ARM64_STRUCT
#include "Register/Arch/RegisterInfos_arm64.h"
#include "Register/Arch/RegisterInfos_arm64_sve.h"
#undef DECLARE_REGISTER_INFOS_ARM64_STRUCT

static lldb_private::RegisterInfo g_register_infos_pauth[] = {
    DEFINE_EXTENSION_REG(data_mask), DEFINE_EXTENSION_REG(code_mask)};

static lldb_private::RegisterInfo g_register_infos_mte[] = {
    DEFINE_EXTENSION_REG(mte_ctrl)};

static lldb_private::RegisterInfo g_register_infos_tls[] = {
    DEFINE_EXTENSION_REG_GENERIC(tpidr, LLDB_REGNUM_GENERIC_TP),
    // Only present when SME is present
    DEFINE_EXTENSION_REG(tpidr2)};

static lldb_private::RegisterInfo g_register_infos_sme[] = {
    DEFINE_EXTENSION_REG(svcr),
    DEFINE_EXTENSION_REG(svg),
    // 16 is a default size we will change later.
    {"za", nullptr, 16, 0, lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,
     KIND_ALL_INVALID, nullptr, nullptr, nullptr}};

static lldb_private::RegisterInfo g_register_infos_sme2[] = {
    {"zt0", nullptr, 64, 0, lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,
     KIND_ALL_INVALID, nullptr, nullptr, nullptr}};

static lldb_private::RegisterInfo g_register_infos_fpmr[] = {
    DEFINE_EXTENSION_REG(fpmr)};

static lldb_private::RegisterInfo g_register_infos_gcs[] = {
    DEFINE_EXTENSION_REG(gcs_features_enabled),
    DEFINE_EXTENSION_REG(gcs_features_locked), DEFINE_EXTENSION_REG(gcspr_el0)};

static lldb_private::RegisterInfo g_register_infos_poe[] = {
    DEFINE_EXTENSION_REG(por_el0)};

#undef GPR_W_PSEUDO_REG_ENDIAN_OFFSET
#undef FPU_S_PSEUDO_REG_ENDIAN_OFFSET
#undef FPU_D_PSEUDO_REG_ENDIAN_OFFSET
#undef MISC_KIND
#undef LLDB_KIND
#undef GENERIC_KIND
#undef KIND_ALL_INVALID
#undef GPR64_KIND
#undef VREG_KIND
#undef DEFINE_GPR64
#undef DEFINE_GPR64_ALT
#undef DEFINE_GPR32
#undef DEFINE_VREG
#undef DEFINE_FPU_PSEUDO
#undef DEFINE_GPR_CPSR
#undef DEFINE_MISC_REGS
#undef DEFINE_MISC_LLDB_REGS
#undef DEFINE_EXTENSION_REG
#undef DEFINE_EXTENSION_REG_GENERIC
#undef LLDB_REGISTER_INFOS_ARM64_KEEP_MACROS

// Number of register sets provided by this context.
enum {
  k_num_gpr_registers = reg_ids::gpr_w28 - reg_ids::gpr_x0 + 1,
  k_num_fpr_registers = reg_ids::fpu_fpcr - reg_ids::fpu_v0 + 1,
  k_num_sve_registers = reg_ids::sve_ffr - reg_ids::sve_vg + 1,
  k_num_mte_register = 1,
  // Number of TLS registers is dynamic so it is not listed here.
  k_num_pauth_register = 2,
  // SME2's ZT0 will also be added to this set if present. So this number is
  // only for SME1 registers.
  k_num_sme_register = 3,
  k_num_fpmr_register = 1,
  k_num_gcs_register = 3,
  k_num_poe_register = 1,
  k_num_register_sets_default = 2,
  k_num_register_sets = 3
};

// ARM64 general purpose registers.
static const uint32_t g_gpr_regnums_arm64[] = {
    reg_ids::gpr_x0,  reg_ids::gpr_x1,   reg_ids::gpr_x2,  reg_ids::gpr_x3,
    reg_ids::gpr_x4,  reg_ids::gpr_x5,   reg_ids::gpr_x6,  reg_ids::gpr_x7,
    reg_ids::gpr_x8,  reg_ids::gpr_x9,   reg_ids::gpr_x10, reg_ids::gpr_x11,
    reg_ids::gpr_x12, reg_ids::gpr_x13,  reg_ids::gpr_x14, reg_ids::gpr_x15,
    reg_ids::gpr_x16, reg_ids::gpr_x17,  reg_ids::gpr_x18, reg_ids::gpr_x19,
    reg_ids::gpr_x20, reg_ids::gpr_x21,  reg_ids::gpr_x22, reg_ids::gpr_x23,
    reg_ids::gpr_x24, reg_ids::gpr_x25,  reg_ids::gpr_x26, reg_ids::gpr_x27,
    reg_ids::gpr_x28, reg_ids::gpr_fp,   reg_ids::gpr_lr,  reg_ids::gpr_sp,
    reg_ids::gpr_pc,  reg_ids::gpr_cpsr, reg_ids::gpr_w0,  reg_ids::gpr_w1,
    reg_ids::gpr_w2,  reg_ids::gpr_w3,   reg_ids::gpr_w4,  reg_ids::gpr_w5,
    reg_ids::gpr_w6,  reg_ids::gpr_w7,   reg_ids::gpr_w8,  reg_ids::gpr_w9,
    reg_ids::gpr_w10, reg_ids::gpr_w11,  reg_ids::gpr_w12, reg_ids::gpr_w13,
    reg_ids::gpr_w14, reg_ids::gpr_w15,  reg_ids::gpr_w16, reg_ids::gpr_w17,
    reg_ids::gpr_w18, reg_ids::gpr_w19,  reg_ids::gpr_w20, reg_ids::gpr_w21,
    reg_ids::gpr_w22, reg_ids::gpr_w23,  reg_ids::gpr_w24, reg_ids::gpr_w25,
    reg_ids::gpr_w26, reg_ids::gpr_w27,  reg_ids::gpr_w28, LLDB_INVALID_REGNUM};

static_assert(((sizeof g_gpr_regnums_arm64 / sizeof g_gpr_regnums_arm64[0]) -
               1) == k_num_gpr_registers,
              "g_gpr_regnums_arm64 has wrong number of register infos");

// ARM64 floating point registers.
static const uint32_t g_fpu_regnums_arm64[] = {
    reg_ids::fpu_v0,   reg_ids::fpu_v1,   reg_ids::fpu_v2,    reg_ids::fpu_v3,
    reg_ids::fpu_v4,   reg_ids::fpu_v5,   reg_ids::fpu_v6,    reg_ids::fpu_v7,
    reg_ids::fpu_v8,   reg_ids::fpu_v9,   reg_ids::fpu_v10,   reg_ids::fpu_v11,
    reg_ids::fpu_v12,  reg_ids::fpu_v13,  reg_ids::fpu_v14,   reg_ids::fpu_v15,
    reg_ids::fpu_v16,  reg_ids::fpu_v17,  reg_ids::fpu_v18,   reg_ids::fpu_v19,
    reg_ids::fpu_v20,  reg_ids::fpu_v21,  reg_ids::fpu_v22,   reg_ids::fpu_v23,
    reg_ids::fpu_v24,  reg_ids::fpu_v25,  reg_ids::fpu_v26,   reg_ids::fpu_v27,
    reg_ids::fpu_v28,  reg_ids::fpu_v29,  reg_ids::fpu_v30,   reg_ids::fpu_v31,
    reg_ids::fpu_s0,   reg_ids::fpu_s1,   reg_ids::fpu_s2,    reg_ids::fpu_s3,
    reg_ids::fpu_s4,   reg_ids::fpu_s5,   reg_ids::fpu_s6,    reg_ids::fpu_s7,
    reg_ids::fpu_s8,   reg_ids::fpu_s9,   reg_ids::fpu_s10,   reg_ids::fpu_s11,
    reg_ids::fpu_s12,  reg_ids::fpu_s13,  reg_ids::fpu_s14,   reg_ids::fpu_s15,
    reg_ids::fpu_s16,  reg_ids::fpu_s17,  reg_ids::fpu_s18,   reg_ids::fpu_s19,
    reg_ids::fpu_s20,  reg_ids::fpu_s21,  reg_ids::fpu_s22,   reg_ids::fpu_s23,
    reg_ids::fpu_s24,  reg_ids::fpu_s25,  reg_ids::fpu_s26,   reg_ids::fpu_s27,
    reg_ids::fpu_s28,  reg_ids::fpu_s29,  reg_ids::fpu_s30,   reg_ids::fpu_s31,
    reg_ids::fpu_d0,   reg_ids::fpu_d1,   reg_ids::fpu_d2,    reg_ids::fpu_d3,
    reg_ids::fpu_d4,   reg_ids::fpu_d5,   reg_ids::fpu_d6,    reg_ids::fpu_d7,
    reg_ids::fpu_d8,   reg_ids::fpu_d9,   reg_ids::fpu_d10,   reg_ids::fpu_d11,
    reg_ids::fpu_d12,  reg_ids::fpu_d13,  reg_ids::fpu_d14,   reg_ids::fpu_d15,
    reg_ids::fpu_d16,  reg_ids::fpu_d17,  reg_ids::fpu_d18,   reg_ids::fpu_d19,
    reg_ids::fpu_d20,  reg_ids::fpu_d21,  reg_ids::fpu_d22,   reg_ids::fpu_d23,
    reg_ids::fpu_d24,  reg_ids::fpu_d25,  reg_ids::fpu_d26,   reg_ids::fpu_d27,
    reg_ids::fpu_d28,  reg_ids::fpu_d29,  reg_ids::fpu_d30,   reg_ids::fpu_d31,
    reg_ids::fpu_fpsr, reg_ids::fpu_fpcr, LLDB_INVALID_REGNUM};
static_assert(((sizeof g_fpu_regnums_arm64 / sizeof g_fpu_regnums_arm64[0]) -
               1) == k_num_fpr_registers,
              "g_fpu_regnums_arm64 has wrong number of register infos");

// ARM64 SVE registers.
static const uint32_t g_sve_regnums_arm64[] = {
    reg_ids::sve_vg,  reg_ids::sve_z0,  reg_ids::sve_z1,    reg_ids::sve_z2,
    reg_ids::sve_z3,  reg_ids::sve_z4,  reg_ids::sve_z5,    reg_ids::sve_z6,
    reg_ids::sve_z7,  reg_ids::sve_z8,  reg_ids::sve_z9,    reg_ids::sve_z10,
    reg_ids::sve_z11, reg_ids::sve_z12, reg_ids::sve_z13,   reg_ids::sve_z14,
    reg_ids::sve_z15, reg_ids::sve_z16, reg_ids::sve_z17,   reg_ids::sve_z18,
    reg_ids::sve_z19, reg_ids::sve_z20, reg_ids::sve_z21,   reg_ids::sve_z22,
    reg_ids::sve_z23, reg_ids::sve_z24, reg_ids::sve_z25,   reg_ids::sve_z26,
    reg_ids::sve_z27, reg_ids::sve_z28, reg_ids::sve_z29,   reg_ids::sve_z30,
    reg_ids::sve_z31, reg_ids::sve_p0,  reg_ids::sve_p1,    reg_ids::sve_p2,
    reg_ids::sve_p3,  reg_ids::sve_p4,  reg_ids::sve_p5,    reg_ids::sve_p6,
    reg_ids::sve_p7,  reg_ids::sve_p8,  reg_ids::sve_p9,    reg_ids::sve_p10,
    reg_ids::sve_p11, reg_ids::sve_p12, reg_ids::sve_p13,   reg_ids::sve_p14,
    reg_ids::sve_p15, reg_ids::sve_ffr, LLDB_INVALID_REGNUM};
static_assert(((sizeof g_sve_regnums_arm64 / sizeof g_sve_regnums_arm64[0]) -
               1) == k_num_sve_registers,
              "g_sve_regnums_arm64 has wrong number of register infos");

// Register sets for ARM64.
static const lldb_private::RegisterSet g_reg_sets_arm64[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", k_num_gpr_registers,
     g_gpr_regnums_arm64},
    {"Floating Point Registers", "fpu", k_num_fpr_registers,
     g_fpu_regnums_arm64},
    {"Scalable Vector Extension Registers", "sve", k_num_sve_registers,
     g_sve_regnums_arm64}};

static const lldb_private::RegisterSet g_reg_set_pauth_arm64 = {
    "Pointer Authentication Registers", "pauth", k_num_pauth_register, nullptr};

static const lldb_private::RegisterSet g_reg_set_mte_arm64 = {
    "MTE Control Register", "mte", k_num_mte_register, nullptr};

// The size of the TLS set is dynamic, so not listed here.

static const lldb_private::RegisterSet g_reg_set_sme_arm64 = {
    "Scalable Matrix Extension Registers", "sme", k_num_sme_register, nullptr};

static const lldb_private::RegisterSet g_reg_set_fpmr_arm64 = {
    "Floating Point Mode Register", "fpmr", k_num_fpmr_register, nullptr};

static const lldb_private::RegisterSet g_reg_set_gcs_arm64 = {
    "Guarded Control Stack Registers", "gcs", k_num_gcs_register, nullptr};

static const lldb_private::RegisterSet g_reg_set_poe_arm64 = {
    "Permission Overlay Registers", "poe", k_num_poe_register, nullptr};

RegisterInfo_arm64::RegisterInfo_arm64(
    const lldb_private::ArchSpec &target_arch, lldb_private::Flags opt_regsets)
    : lldb_private::RegisterInfoAndSetInterface(target_arch),
      m_opt_regsets(opt_regsets) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_32: {
    m_register_set_p = g_reg_sets_arm64;
    m_register_set_count = k_num_register_sets_default;
    m_per_regset_regnum_range[GPRegSet] =
        std::make_pair(reg_ids::gpr_x0, reg_ids::gpr_w28 + 1);
    m_per_regset_regnum_range[FPRegSet] =
        std::make_pair(reg_ids::fpu_v0, reg_ids::fpu_fpcr + 1);

    // Now configure register sets supported by current target. If we have a
    // dynamic register set like MTE, Pointer Authentication regset then we need
    // to create dynamic register infos and regset array. Push back all optional
    // register infos and regset and calculate register offsets accordingly.
    if (m_opt_regsets.AnySet(eRegsetMaskSVE | eRegsetMaskSSVE)) {
      m_register_info_p = g_register_infos_arm64_sve_le;
      m_register_info_count = reg_ids::sve_ffr + 1;
      m_per_regset_regnum_range[m_register_set_count++] =
          std::make_pair(reg_ids::sve_vg, reg_ids::sve_ffr + 1);
    } else {
      m_register_info_p = g_register_infos_arm64_le;
      m_register_info_count = reg_ids::fpu_fpcr + 1;
    }

    if (m_opt_regsets.AnySet(eRegsetMaskDynamic)) {
      llvm::ArrayRef<lldb_private::RegisterInfo> reg_infos_ref =
          llvm::ArrayRef(m_register_info_p, m_register_info_count);
      llvm::ArrayRef<lldb_private::RegisterSet> reg_sets_ref =
          llvm::ArrayRef(m_register_set_p, m_register_set_count);
      llvm::copy(reg_infos_ref, std::back_inserter(m_dynamic_reg_infos));
      llvm::copy(reg_sets_ref, std::back_inserter(m_dynamic_reg_sets));

      if (m_opt_regsets.AllSet(eRegsetMaskPAuth))
        AddRegSetPAuth();

      if (m_opt_regsets.AllSet(eRegsetMaskMTE))
        AddRegSetMTE();

      if (m_opt_regsets.AllSet(eRegsetMaskTLS)) {
        // The TLS set always contains tpidr but only has tpidr2 when SME is
        // present.
        AddRegSetTLS(m_opt_regsets.AllSet(eRegsetMaskSSVE));
      }

      if (m_opt_regsets.AnySet(eRegsetMaskSSVE))
        AddRegSetSME(m_opt_regsets.AnySet(eRegsetMaskZT));

      if (m_opt_regsets.AllSet(eRegsetMaskFPMR))
        AddRegSetFPMR();

      if (m_opt_regsets.AllSet(eRegsetMaskGCS))
        AddRegSetGCS();

      if (m_opt_regsets.AllSet(eRegsetMaskPOE))
        AddRegSetPOE();

      m_register_info_count = m_dynamic_reg_infos.size();
      m_register_info_p = m_dynamic_reg_infos.data();
      m_register_set_p = m_dynamic_reg_sets.data();
      m_register_set_count = m_dynamic_reg_sets.size();
    }
    break;
  }
  default:
    assert(false && "Unhandled target architecture.");
  }
}

uint32_t RegisterInfo_arm64::GetRegisterCount() const {
  return m_register_info_count;
}

size_t RegisterInfo_arm64::GetGPRSizeStatic() {
  return sizeof(struct RegisterInfo_arm64::GPR);
}

size_t RegisterInfo_arm64::GetFPRSize() const {
  return sizeof(struct RegisterInfo_arm64::FPU);
}

const lldb_private::RegisterInfo *
RegisterInfo_arm64::GetRegisterInfo() const {
  return m_register_info_p;
}

size_t RegisterInfo_arm64::GetRegisterSetCount() const {
  return m_register_set_count;
}

size_t RegisterInfo_arm64::GetRegisterSetFromRegisterIndex(
    uint32_t reg_index) const {
  for (const auto &regset_range : m_per_regset_regnum_range) {
    if (reg_index >= regset_range.second.first &&
        reg_index < regset_range.second.second)
      return regset_range.first;
  }
  return LLDB_INVALID_REGNUM;
}

const lldb_private::RegisterSet *
RegisterInfo_arm64::GetRegisterSet(size_t set_index) const {
  if (set_index < GetRegisterSetCount())
    return &m_register_set_p[set_index];
  return nullptr;
}

void RegisterInfo_arm64::AddRegSetPAuth() {
  uint32_t pa_regnum = m_dynamic_reg_infos.size();
  for (uint32_t i = 0; i < k_num_pauth_register; i++) {
    pauth_regnum_collection.push_back(pa_regnum + i);
    m_dynamic_reg_infos.push_back(g_register_infos_pauth[i]);
    m_dynamic_reg_infos[pa_regnum + i].byte_offset =
        m_dynamic_reg_infos[pa_regnum + i - 1].byte_offset +
        m_dynamic_reg_infos[pa_regnum + i - 1].byte_size;
    m_dynamic_reg_infos[pa_regnum + i].kinds[lldb::eRegisterKindLLDB] =
        pa_regnum + i;
  }

  m_per_regset_regnum_range[m_register_set_count] =
      std::make_pair(pa_regnum, m_dynamic_reg_infos.size());
  m_dynamic_reg_sets.push_back(g_reg_set_pauth_arm64);
  m_dynamic_reg_sets.back().registers = pauth_regnum_collection.data();
}

void RegisterInfo_arm64::AddRegSetMTE() {
  uint32_t mte_regnum = m_dynamic_reg_infos.size();
  m_mte_regnum_collection.push_back(mte_regnum);
  m_dynamic_reg_infos.push_back(g_register_infos_mte[0]);
  m_dynamic_reg_infos[mte_regnum].byte_offset =
      m_dynamic_reg_infos[mte_regnum - 1].byte_offset +
      m_dynamic_reg_infos[mte_regnum - 1].byte_size;
  m_dynamic_reg_infos[mte_regnum].kinds[lldb::eRegisterKindLLDB] = mte_regnum;

  m_per_regset_regnum_range[m_register_set_count] =
      std::make_pair(mte_regnum, mte_regnum + 1);
  m_dynamic_reg_sets.push_back(g_reg_set_mte_arm64);
  m_dynamic_reg_sets.back().registers = m_mte_regnum_collection.data();
}

void RegisterInfo_arm64::AddRegSetTLS(bool has_tpidr2) {
  uint32_t tls_regnum = m_dynamic_reg_infos.size();
  uint32_t num_regs = has_tpidr2 ? 2 : 1;
  for (uint32_t i = 0; i < num_regs; i++) {
    m_tls_regnum_collection.push_back(tls_regnum + i);
    m_dynamic_reg_infos.push_back(g_register_infos_tls[i]);
    m_dynamic_reg_infos[tls_regnum + i].byte_offset =
        m_dynamic_reg_infos[tls_regnum + i - 1].byte_offset +
        m_dynamic_reg_infos[tls_regnum + i - 1].byte_size;
    m_dynamic_reg_infos[tls_regnum + i].kinds[lldb::eRegisterKindLLDB] =
        tls_regnum + i;
  }

  m_per_regset_regnum_range[m_register_set_count] =
      std::make_pair(tls_regnum, m_dynamic_reg_infos.size());
  m_dynamic_reg_sets.push_back(
      {"Thread Local Storage Registers", "tls", num_regs, nullptr});
  m_dynamic_reg_sets.back().registers = m_tls_regnum_collection.data();
}

void RegisterInfo_arm64::AddRegSetSME(bool has_zt) {
  const uint32_t first_sme_regnum = m_dynamic_reg_infos.size();
  uint32_t sme_regnum = first_sme_regnum;

  for (uint32_t i = 0; i < k_num_sme_register; ++i, ++sme_regnum) {
    m_sme_regnum_collection.push_back(sme_regnum);
    m_dynamic_reg_infos.push_back(g_register_infos_sme[i]);
    m_dynamic_reg_infos[sme_regnum].byte_offset =
        m_dynamic_reg_infos[sme_regnum - 1].byte_offset +
        m_dynamic_reg_infos[sme_regnum - 1].byte_size;
    m_dynamic_reg_infos[sme_regnum].kinds[lldb::eRegisterKindLLDB] = sme_regnum;
  }

  lldb_private::RegisterSet sme_regset = g_reg_set_sme_arm64;

  if (has_zt) {
    m_sme_regnum_collection.push_back(sme_regnum);
    m_dynamic_reg_infos.push_back(g_register_infos_sme2[0]);
    m_dynamic_reg_infos[sme_regnum].byte_offset =
        m_dynamic_reg_infos[sme_regnum - 1].byte_offset +
        m_dynamic_reg_infos[sme_regnum - 1].byte_size;
    m_dynamic_reg_infos[sme_regnum].kinds[lldb::eRegisterKindLLDB] = sme_regnum;

    sme_regset.num_registers += 1;
  }

  m_per_regset_regnum_range[m_register_set_count] =
      std::make_pair(first_sme_regnum, m_dynamic_reg_infos.size());
  m_dynamic_reg_sets.push_back(sme_regset);
  m_dynamic_reg_sets.back().registers = m_sme_regnum_collection.data();

  // When vg is written during streaming mode, svg will also change, as vg and
  // svg in this state are both showing the streaming vector length.
  // We model this as vg invalidating svg. In non-streaming mode this doesn't
  // happen but to keep things simple we will invalidate svg anyway.
  //
  // This must be added now, rather than when vg is defined because SME is a
  // dynamic set that may or may not be present.
  static uint32_t vg_invalidates[] = {GetRegNumSMESVG(), LLDB_INVALID_REGNUM};
  m_dynamic_reg_infos[GetRegNumSVEVG()].invalidate_regs = vg_invalidates;
}

void RegisterInfo_arm64::AddRegSetFPMR() {
  uint32_t fpmr_regnum = m_dynamic_reg_infos.size();
  m_fpmr_regnum_collection.push_back(fpmr_regnum);
  m_dynamic_reg_infos.push_back(g_register_infos_fpmr[0]);
  m_dynamic_reg_infos[fpmr_regnum].byte_offset =
      m_dynamic_reg_infos[fpmr_regnum - 1].byte_offset +
      m_dynamic_reg_infos[fpmr_regnum - 1].byte_size;
  m_dynamic_reg_infos[fpmr_regnum].kinds[lldb::eRegisterKindLLDB] = fpmr_regnum;

  m_per_regset_regnum_range[m_register_set_count] =
      std::make_pair(fpmr_regnum, fpmr_regnum + 1);
  m_dynamic_reg_sets.push_back(g_reg_set_fpmr_arm64);
  m_dynamic_reg_sets.back().registers = m_fpmr_regnum_collection.data();
}

void RegisterInfo_arm64::AddRegSetGCS() {
  uint32_t gcs_regnum = m_dynamic_reg_infos.size();
  for (uint32_t i = 0; i < k_num_gcs_register; i++) {
    m_gcs_regnum_collection.push_back(gcs_regnum + i);
    m_dynamic_reg_infos.push_back(g_register_infos_gcs[i]);
    m_dynamic_reg_infos[gcs_regnum + i].byte_offset =
        m_dynamic_reg_infos[gcs_regnum + i - 1].byte_offset +
        m_dynamic_reg_infos[gcs_regnum + i - 1].byte_size;
    m_dynamic_reg_infos[gcs_regnum + i].kinds[lldb::eRegisterKindLLDB] =
        gcs_regnum + i;
  }

  m_per_regset_regnum_range[m_register_set_count] =
      std::make_pair(gcs_regnum, m_dynamic_reg_infos.size());
  m_dynamic_reg_sets.push_back(g_reg_set_gcs_arm64);
  m_dynamic_reg_sets.back().registers = m_gcs_regnum_collection.data();
}

void RegisterInfo_arm64::AddRegSetPOE() {
  uint32_t poe_regnum = m_dynamic_reg_infos.size();
  m_poe_regnum_collection.push_back(poe_regnum);
  m_dynamic_reg_infos.push_back(g_register_infos_poe[0]);
  m_dynamic_reg_infos[poe_regnum].byte_offset =
      m_dynamic_reg_infos[poe_regnum - 1].byte_offset +
      m_dynamic_reg_infos[poe_regnum - 1].byte_size;
  m_dynamic_reg_infos[poe_regnum].kinds[lldb::eRegisterKindLLDB] = poe_regnum;

  m_per_regset_regnum_range[m_register_set_count] =
      std::make_pair(poe_regnum, poe_regnum + 1);
  m_dynamic_reg_sets.push_back(g_reg_set_poe_arm64);
  m_dynamic_reg_sets.back().registers = m_poe_regnum_collection.data();
}

uint32_t RegisterInfo_arm64::ConfigureVectorLengthSVE(uint32_t sve_vq) {
  // sve_vq contains SVE Quad vector length in context of AArch64 SVE.
  // SVE register infos if enabled cannot be disabled by selecting sve_vq = 0.
  // Also if an invalid or previously set vector length is passed to this
  // function then it will exit immediately with previously set vector length.
  if (!VectorSizeIsValid(sve_vq) || m_vector_reg_vq == sve_vq)
    return m_vector_reg_vq;

  // We cannot enable AArch64 only mode if SVE was enabled.
  if (sve_vq == eVectorQuadwordAArch64 &&
      m_vector_reg_vq > eVectorQuadwordAArch64)
    sve_vq = eVectorQuadwordAArch64SVE;

  m_vector_reg_vq = sve_vq;

  if (sve_vq == eVectorQuadwordAArch64)
    return m_vector_reg_vq;
  std::vector<lldb_private::RegisterInfo> &reg_info_ref =
      m_per_vq_reg_infos[sve_vq];

  if (reg_info_ref.empty()) {
    reg_info_ref = llvm::ArrayRef(m_register_info_p, m_register_info_count);

    uint32_t offset = SVE_REGS_DEFAULT_OFFSET_LINUX;
    reg_info_ref[reg_ids::fpu_fpsr].byte_offset = offset;
    reg_info_ref[reg_ids::fpu_fpcr].byte_offset = offset + 4;
    reg_info_ref[reg_ids::sve_vg].byte_offset = offset + 8;
    offset += 16;

    // Update Z registers size and offset
    uint32_t s_reg_base = reg_ids::fpu_s0;
    uint32_t d_reg_base = reg_ids::fpu_d0;
    uint32_t v_reg_base = reg_ids::fpu_v0;
    uint32_t z_reg_base = reg_ids::sve_z0;

    for (uint32_t index = 0; index < 32; index++) {
      reg_info_ref[s_reg_base + index].byte_offset = offset;
      reg_info_ref[d_reg_base + index].byte_offset = offset;
      reg_info_ref[v_reg_base + index].byte_offset = offset;
      reg_info_ref[z_reg_base + index].byte_offset = offset;

      reg_info_ref[z_reg_base + index].byte_size = sve_vq * SVE_QUAD_WORD_BYTES;
      offset += reg_info_ref[z_reg_base + index].byte_size;
    }

    // Update P registers and FFR size and offset
    for (uint32_t it = reg_ids::sve_p0; it <= reg_ids::sve_ffr; it++) {
      reg_info_ref[it].byte_offset = offset;
      reg_info_ref[it].byte_size = sve_vq * SVE_QUAD_WORD_BYTES / 8;
      offset += reg_info_ref[it].byte_size;
    }

    for (uint32_t it = reg_ids::sve_ffr + 1; it < m_register_info_count; it++) {
      reg_info_ref[it].byte_offset = offset;
      offset += reg_info_ref[it].byte_size;
    }

    m_per_vq_reg_infos[sve_vq] = reg_info_ref;
  }

  m_register_info_p = m_per_vq_reg_infos[sve_vq].data();
  return m_vector_reg_vq;
}

void RegisterInfo_arm64::ConfigureVectorLengthZA(uint32_t za_vq) {
  if (!VectorSizeIsValid(za_vq) || m_za_reg_vq == za_vq)
    return;

  m_za_reg_vq = za_vq;

  // For SVE changes, we replace m_register_info_p completely. ZA is in a
  // dynamic set and is just 1 register so we make an exception to const here.
  lldb_private::RegisterInfo *non_const_reginfo =
      const_cast<lldb_private::RegisterInfo *>(m_register_info_p);
  non_const_reginfo[m_sme_regnum_collection[2]].byte_size =
      (za_vq * 16) * (za_vq * 16);
}

bool RegisterInfo_arm64::IsSVEReg(unsigned reg) const {
  if (m_vector_reg_vq > eVectorQuadwordAArch64)
    return (reg_ids::sve_vg <= reg && reg <= reg_ids::sve_ffr);
  else
    return false;
}

bool RegisterInfo_arm64::IsSVEZReg(unsigned reg) const {
  return (reg_ids::sve_z0 <= reg && reg <= reg_ids::sve_z31);
}

bool RegisterInfo_arm64::IsSVEPReg(unsigned reg) const {
  return (reg_ids::sve_p0 <= reg && reg <= reg_ids::sve_p15);
}

bool RegisterInfo_arm64::IsSVERegVG(unsigned reg) const {
  return reg_ids::sve_vg == reg;
}

bool RegisterInfo_arm64::IsSVERegFFR(unsigned reg) const {
  return reg_ids::sve_ffr == reg;
}

bool RegisterInfo_arm64::IsSMERegZA(unsigned reg) const {
  return reg == m_sme_regnum_collection[2];
}

bool RegisterInfo_arm64::IsSMERegZT(unsigned reg) const {
  // ZT0 is part of the SME register set only if SME2 is present.
  return m_sme_regnum_collection.size() >= 4 &&
         reg == m_sme_regnum_collection[3];
}

bool RegisterInfo_arm64::IsGPR(unsigned reg) const {
  return GetRegisterSetFromRegisterIndex(reg) ==
         RegisterInfo_arm64::GPRegSet;
}

bool RegisterInfo_arm64::IsFPR(unsigned reg) const {
  return GetRegisterSetFromRegisterIndex(reg) ==
         RegisterInfo_arm64::FPRegSet;
}

bool RegisterInfo_arm64::IsPAuthReg(unsigned reg) const {
  return llvm::is_contained(pauth_regnum_collection, reg);
}

bool RegisterInfo_arm64::IsMTEReg(unsigned reg) const {
  return llvm::is_contained(m_mte_regnum_collection, reg);
}

bool RegisterInfo_arm64::IsTLSReg(unsigned reg) const {
  return llvm::is_contained(m_tls_regnum_collection, reg);
}

bool RegisterInfo_arm64::IsSMEReg(unsigned reg) const {
  return llvm::is_contained(m_sme_regnum_collection, reg);
}

bool RegisterInfo_arm64::IsFPMRReg(unsigned reg) const {
  return llvm::is_contained(m_fpmr_regnum_collection, reg);
}

bool RegisterInfo_arm64::IsGCSReg(unsigned reg) const {
  return llvm::is_contained(m_gcs_regnum_collection, reg);
}

bool RegisterInfo_arm64::IsPOEReg(unsigned reg) const {
  return llvm::is_contained(m_poe_regnum_collection, reg);
}

uint32_t RegisterInfo_arm64::GetRegNumSVEZ0() const { return reg_ids::sve_z0; }

uint32_t RegisterInfo_arm64::GetRegNumSVEFFR() const {
  return reg_ids::sve_ffr;
}

uint32_t RegisterInfo_arm64::GetRegNumFPCR() const { return reg_ids::fpu_fpcr; }

uint32_t RegisterInfo_arm64::GetRegNumFPSR() const { return reg_ids::fpu_fpsr; }

uint32_t RegisterInfo_arm64::GetRegNumFPV0() const { return reg_ids::fpu_v0; }

uint32_t RegisterInfo_arm64::GetRegNumSVEVG() const { return reg_ids::sve_vg; }

uint32_t RegisterInfo_arm64::GetRegNumSMESVG() const {
  return m_sme_regnum_collection[1];
}

uint32_t RegisterInfo_arm64::GetPAuthOffset() const {
  return m_register_info_p[pauth_regnum_collection[0]].byte_offset;
}

uint32_t RegisterInfo_arm64::GetMTEOffset() const {
  return m_register_info_p[m_mte_regnum_collection[0]].byte_offset;
}

uint32_t RegisterInfo_arm64::GetTLSOffset() const {
  return m_register_info_p[m_tls_regnum_collection[0]].byte_offset;
}

uint32_t RegisterInfo_arm64::GetSMEOffset() const {
  return m_register_info_p[m_sme_regnum_collection[0]].byte_offset;
}

uint32_t RegisterInfo_arm64::GetFPMROffset() const {
  return m_register_info_p[m_fpmr_regnum_collection[0]].byte_offset;
}

uint32_t RegisterInfo_arm64::GetGCSOffset() const {
  return m_register_info_p[m_gcs_regnum_collection[0]].byte_offset;
}

uint32_t RegisterInfo_arm64::GetPOEOffset() const {
  return m_register_info_p[m_poe_regnum_collection[0]].byte_offset;
}
