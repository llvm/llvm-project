//===-- RegisterContextMinidump_ARM.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextMinidump_ARM.h"
#include "lldb/Register/Arch/lldb-arm-ehframe-register-enums.h"

#include "lldb/Register/Arch/lldb-arm-dwarf-register-enums.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/lldb-enumerations.h"

// C includes
#include <cassert>

// C++ includes

namespace dwarf_regs = lldb_private::registers::arm::dwarf;
namespace ehframe_regs = lldb_private::registers::arm::ehframe;
using namespace lldb;
using namespace lldb_private;
using namespace minidump;

#define INV LLDB_INVALID_REGNUM

#define DEF_R(reg_id, i)                                                       \
  RegisterDescriptor(reg_id, "r" #i, nullptr, 4, eEncodingUint, eFormatHex,    \
                     {ehframe_regs::r##i, dwarf_regs::r##i, INV, INV, reg_r##i}, {}, {},     \
                     nullptr)

#define DEF_R_ARG(reg_id, i, n)                                                \
  RegisterDescriptor(                                                          \
      reg_id, "r" #i, "arg" #n, 4, eEncodingUint, eFormatHex,                  \
      {ehframe_regs::r##i, dwarf_regs::r##i, LLDB_REGNUM_GENERIC_ARG1 + i, INV, reg_r##i},   \
      {}, {}, nullptr)

#define DEF_D(reg_id, i)                                                       \
  RegisterDescriptor(                                                          \
      reg_id, "d" #i, nullptr, 8, eEncodingVector, eFormatVectorOfUInt8,       \
      {ehframe_regs::d##i, dwarf_regs::d##i, INV, INV, reg_d##i}, {}, {},      \
      nullptr)

#define DEF_S(reg_id, i)                                                       \
  RegisterDescriptor(                                                          \
      reg_id, "s" #i, nullptr, 4, eEncodingIEEE754, eFormatFloat,              \
      {ehframe_regs::s##i, dwarf_regs::s##i, INV, INV, reg_s##i}, {}, {},      \
      nullptr)

#define DEF_Q(reg_id, i)                                                       \
  RegisterDescriptor(                                                          \
      reg_id, "q" #i, nullptr, 16, eEncodingVector, eFormatVectorOfUInt8,      \
      {ehframe_regs::q##i, dwarf_regs::q##i, INV, INV, reg_q##i}, {}, {},      \
      nullptr)

// Zero based LLDB register numbers for this register context
enum {
  // General Purpose Registers
  reg_r0,
  reg_r1,
  reg_r2,
  reg_r3,
  reg_r4,
  reg_r5,
  reg_r6,
  reg_r7,
  reg_r8,
  reg_r9,
  reg_r10,
  reg_r11,
  reg_r12,
  reg_sp,
  reg_lr,
  reg_pc,
  reg_cpsr,
  // Floating Point Registers
  reg_fpscr,
  reg_d0,
  reg_d1,
  reg_d2,
  reg_d3,
  reg_d4,
  reg_d5,
  reg_d6,
  reg_d7,
  reg_d8,
  reg_d9,
  reg_d10,
  reg_d11,
  reg_d12,
  reg_d13,
  reg_d14,
  reg_d15,
  reg_d16,
  reg_d17,
  reg_d18,
  reg_d19,
  reg_d20,
  reg_d21,
  reg_d22,
  reg_d23,
  reg_d24,
  reg_d25,
  reg_d26,
  reg_d27,
  reg_d28,
  reg_d29,
  reg_d30,
  reg_d31,
  reg_s0,
  reg_s1,
  reg_s2,
  reg_s3,
  reg_s4,
  reg_s5,
  reg_s6,
  reg_s7,
  reg_s8,
  reg_s9,
  reg_s10,
  reg_s11,
  reg_s12,
  reg_s13,
  reg_s14,
  reg_s15,
  reg_s16,
  reg_s17,
  reg_s18,
  reg_s19,
  reg_s20,
  reg_s21,
  reg_s22,
  reg_s23,
  reg_s24,
  reg_s25,
  reg_s26,
  reg_s27,
  reg_s28,
  reg_s29,
  reg_s30,
  reg_s31,
  reg_q0,
  reg_q1,
  reg_q2,
  reg_q3,
  reg_q4,
  reg_q5,
  reg_q6,
  reg_q7,
  reg_q8,
  reg_q9,
  reg_q10,
  reg_q11,
  reg_q12,
  reg_q13,
  reg_q14,
  reg_q15,
  k_num_regs
};

static RegisterDescriptor g_reg_info_apple_fp = RegisterDescriptor(
    reg_r7, "fp", "r7", 4, eEncodingUint, eFormatHex,
    {ehframe_regs::r7, dwarf_regs::r7, LLDB_REGNUM_GENERIC_FP, INV, reg_r7}, {},
    {}, nullptr);

static RegisterDescriptor g_reg_info_fp = RegisterDescriptor(
    reg_r11, "fp", "r11", 4, eEncodingUint, eFormatHex,
    {ehframe_regs::r11, dwarf_regs::r11, LLDB_REGNUM_GENERIC_FP, INV, reg_r11}, {}, {},
    nullptr);

// Register info definitions for this register context
static RegisterDescriptor g_reg_infos[] = {
    DEF_R_ARG(reg_r0, 0, 1),
    DEF_R_ARG(reg_r1, 1, 2),
    DEF_R_ARG(reg_r2, 2, 3),
    DEF_R_ARG(reg_r3, 3, 4),
    DEF_R(reg_r4, 4),
    DEF_R(reg_r5, 5),
    DEF_R(reg_r6, 6),
    DEF_R(reg_r7, 7),
    DEF_R(reg_r8, 8),
    DEF_R(reg_r9, 9),
    DEF_R(reg_r10, 10),
    DEF_R(reg_r11, 11),
    DEF_R(reg_r12, 12),
    RegisterDescriptor(
        reg_sp, "sp", "r13", 4, eEncodingUint, eFormatHex,
        {ehframe_regs::sp, dwarf_regs::sp, LLDB_REGNUM_GENERIC_SP, INV, reg_sp}, {}, {},
        nullptr),
    RegisterDescriptor(
        reg_lr, "lr", "r14", 4, eEncodingUint, eFormatHex,
        {ehframe_regs::lr, dwarf_regs::lr, LLDB_REGNUM_GENERIC_RA, INV, reg_lr}, {}, {},
        nullptr),
    RegisterDescriptor(
        reg_pc, "pc", "r15", 4, eEncodingUint, eFormatHex,
        {ehframe_regs::pc, dwarf_regs::pc, LLDB_REGNUM_GENERIC_PC, INV, reg_pc}, {}, {},
        nullptr),
    RegisterDescriptor(
        reg_cpsr, "cpsr", "psr", 4, eEncodingUint, eFormatHex,
        {ehframe_regs::cpsr, dwarf_regs::cpsr, LLDB_REGNUM_GENERIC_FLAGS, INV, reg_cpsr}, {},
        {}, nullptr),
    RegisterDescriptor(reg_fpscr, "fpscr", nullptr, 8, eEncodingUint,
                       eFormatHex, {INV, INV, INV, INV, reg_fpscr}, {}, {},
                       nullptr),
    DEF_D(reg_d0, 0),
    DEF_D(reg_d1, 1),
    DEF_D(reg_d2, 2),
    DEF_D(reg_d3, 3),
    DEF_D(reg_d4, 4),
    DEF_D(reg_d5, 5),
    DEF_D(reg_d6, 6),
    DEF_D(reg_d7, 7),
    DEF_D(reg_d8, 8),
    DEF_D(reg_d9, 9),
    DEF_D(reg_d10, 10),
    DEF_D(reg_d11, 11),
    DEF_D(reg_d12, 12),
    DEF_D(reg_d13, 13),
    DEF_D(reg_d14, 14),
    DEF_D(reg_d15, 15),
    DEF_D(reg_d16, 16),
    DEF_D(reg_d17, 17),
    DEF_D(reg_d18, 18),
    DEF_D(reg_d19, 19),
    DEF_D(reg_d20, 20),
    DEF_D(reg_d21, 21),
    DEF_D(reg_d22, 22),
    DEF_D(reg_d23, 23),
    DEF_D(reg_d24, 24),
    DEF_D(reg_d25, 25),
    DEF_D(reg_d26, 26),
    DEF_D(reg_d27, 27),
    DEF_D(reg_d28, 28),
    DEF_D(reg_d29, 29),
    DEF_D(reg_d30, 30),
    DEF_D(reg_d31, 31),
    DEF_S(reg_s0, 0),
    DEF_S(reg_s1, 1),
    DEF_S(reg_s2, 2),
    DEF_S(reg_s3, 3),
    DEF_S(reg_s4, 4),
    DEF_S(reg_s5, 5),
    DEF_S(reg_s6, 6),
    DEF_S(reg_s7, 7),
    DEF_S(reg_s8, 8),
    DEF_S(reg_s9, 9),
    DEF_S(reg_s10, 10),
    DEF_S(reg_s11, 11),
    DEF_S(reg_s12, 12),
    DEF_S(reg_s13, 13),
    DEF_S(reg_s14, 14),
    DEF_S(reg_s15, 15),
    DEF_S(reg_s16, 16),
    DEF_S(reg_s17, 17),
    DEF_S(reg_s18, 18),
    DEF_S(reg_s19, 19),
    DEF_S(reg_s20, 20),
    DEF_S(reg_s21, 21),
    DEF_S(reg_s22, 22),
    DEF_S(reg_s23, 23),
    DEF_S(reg_s24, 24),
    DEF_S(reg_s25, 25),
    DEF_S(reg_s26, 26),
    DEF_S(reg_s27, 27),
    DEF_S(reg_s28, 28),
    DEF_S(reg_s29, 29),
    DEF_S(reg_s30, 30),
    DEF_S(reg_s31, 31),
    DEF_Q(reg_q0, 0),
    DEF_Q(reg_q1, 1),
    DEF_Q(reg_q2, 2),
    DEF_Q(reg_q3, 3),
    DEF_Q(reg_q4, 4),
    DEF_Q(reg_q5, 5),
    DEF_Q(reg_q6, 6),
    DEF_Q(reg_q7, 7),
    DEF_Q(reg_q8, 8),
    DEF_Q(reg_q9, 9),
    DEF_Q(reg_q10, 10),
    DEF_Q(reg_q11, 11),
    DEF_Q(reg_q12, 12),
    DEF_Q(reg_q13, 13),
    DEF_Q(reg_q14, 14),
    DEF_Q(reg_q15, 15)};

// ARM general purpose registers.
const uint32_t g_gpr_regnums[] = {
    reg_r0,
    reg_r1,
    reg_r2,
    reg_r3,
    reg_r4,
    reg_r5,
    reg_r6,
    reg_r7,
    reg_r8,
    reg_r9,
    reg_r10,
    reg_r11,
    reg_r12,
    reg_sp,
    reg_lr,
    reg_pc,
    reg_cpsr,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
const uint32_t g_fpu_regnums[] = {
    reg_fpscr,
    reg_d0,
    reg_d1,
    reg_d2,
    reg_d3,
    reg_d4,
    reg_d5,
    reg_d6,
    reg_d7,
    reg_d8,
    reg_d9,
    reg_d10,
    reg_d11,
    reg_d12,
    reg_d13,
    reg_d14,
    reg_d15,
    reg_d16,
    reg_d17,
    reg_d18,
    reg_d19,
    reg_d20,
    reg_d21,
    reg_d22,
    reg_d23,
    reg_d24,
    reg_d25,
    reg_d26,
    reg_d27,
    reg_d28,
    reg_d29,
    reg_d30,
    reg_d31,
    reg_s0,
    reg_s1,
    reg_s2,
    reg_s3,
    reg_s4,
    reg_s5,
    reg_s6,
    reg_s7,
    reg_s8,
    reg_s9,
    reg_s10,
    reg_s11,
    reg_s12,
    reg_s13,
    reg_s14,
    reg_s15,
    reg_s16,
    reg_s17,
    reg_s18,
    reg_s19,
    reg_s20,
    reg_s21,
    reg_s22,
    reg_s23,
    reg_s24,
    reg_s25,
    reg_s26,
    reg_s27,
    reg_s28,
    reg_s29,
    reg_s30,
    reg_s31,
    reg_q0,
    reg_q1,
    reg_q2,
    reg_q3,
    reg_q4,
    reg_q5,
    reg_q6,
    reg_q7,
    reg_q8,
    reg_q9,
    reg_q10,
    reg_q11,
    reg_q12,
    reg_q13,
    reg_q14,
    reg_q15,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};

// Skip the last LLDB_INVALID_REGNUM in each count below by subtracting 1
constexpr size_t k_num_gpr_regs = std::size(g_gpr_regnums) - 1;
constexpr size_t k_num_fpu_regs = std::size(g_fpu_regnums) - 1;

static RegisterGroup g_reg_sets[] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_regnums, g_gpr_regnums + k_num_gpr_regs)},
    {"Floating Point Registers", "fpu",
     std::vector<RegisterID>(g_fpu_regnums, g_fpu_regnums + k_num_fpu_regs)}};

RegisterLayout
lldb_private::minidump::CreateMinidumpArmRegisterLayout(bool apple) {
  std::vector<RegisterDescriptor> registers(std::begin(g_reg_infos),
                                            std::end(g_reg_infos));
  registers[apple ? reg_r7 : reg_r11] =
      apple ? g_reg_info_apple_fp : g_reg_info_fp;
  auto description = std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create(
          std::move(registers),
          {std::begin(g_reg_sets), std::end(g_reg_sets)})));
  RegisterLayout layout(eByteOrderLittle);
  llvm::cantFail(
      layout.AddBank(0, sizeof(RegisterContextMinidump_ARM::Context)));
  struct Field {
    RegisterID id;
    uint32_t offset;
    uint32_t size;
  };
  const Field fields[] = {
      {reg_r0, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 0 * 4, 4},
      {reg_r1, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 1 * 4, 4},
      {reg_r2, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 2 * 4, 4},
      {reg_r3, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 3 * 4, 4},
      {reg_r4, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 4 * 4, 4},
      {reg_r5, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 5 * 4, 4},
      {reg_r6, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 6 * 4, 4},
      {reg_r7, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 7 * 4, 4},
      {reg_r8, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 8 * 4, 4},
      {reg_r9, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 9 * 4, 4},
      {reg_r10, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 10 * 4,
       4},
      {reg_r11, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 11 * 4,
       4},
      {reg_r12, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 12 * 4,
       4},
      {reg_sp, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 13 * 4, 4},
      {reg_lr, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 14 * 4, 4},
      {reg_pc, (offsetof(RegisterContextMinidump_ARM::Context, r)) + 15 * 4, 4},
      {reg_cpsr, (offsetof(RegisterContextMinidump_ARM::Context, cpsr)), 4},
      {reg_fpscr, (offsetof(RegisterContextMinidump_ARM::Context, fpscr)), 8},
      {reg_d0, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 0 * 8, 8},
      {reg_d1, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 1 * 8, 8},
      {reg_d2, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 2 * 8, 8},
      {reg_d3, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 3 * 8, 8},
      {reg_d4, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 4 * 8, 8},
      {reg_d5, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 5 * 8, 8},
      {reg_d6, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 6 * 8, 8},
      {reg_d7, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 7 * 8, 8},
      {reg_d8, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 8 * 8, 8},
      {reg_d9, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 9 * 8, 8},
      {reg_d10, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 10 * 8,
       8},
      {reg_d11, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 11 * 8,
       8},
      {reg_d12, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 12 * 8,
       8},
      {reg_d13, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 13 * 8,
       8},
      {reg_d14, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 14 * 8,
       8},
      {reg_d15, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 15 * 8,
       8},
      {reg_d16, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 16 * 8,
       8},
      {reg_d17, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 17 * 8,
       8},
      {reg_d18, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 18 * 8,
       8},
      {reg_d19, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 19 * 8,
       8},
      {reg_d20, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 20 * 8,
       8},
      {reg_d21, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 21 * 8,
       8},
      {reg_d22, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 22 * 8,
       8},
      {reg_d23, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 23 * 8,
       8},
      {reg_d24, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 24 * 8,
       8},
      {reg_d25, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 25 * 8,
       8},
      {reg_d26, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 26 * 8,
       8},
      {reg_d27, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 27 * 8,
       8},
      {reg_d28, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 28 * 8,
       8},
      {reg_d29, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 29 * 8,
       8},
      {reg_d30, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 30 * 8,
       8},
      {reg_d31, (offsetof(RegisterContextMinidump_ARM::Context, d)) + 31 * 8,
       8},
      {reg_s0, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 0 * 4, 4},
      {reg_s1, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 1 * 4, 4},
      {reg_s2, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 2 * 4, 4},
      {reg_s3, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 3 * 4, 4},
      {reg_s4, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 4 * 4, 4},
      {reg_s5, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 5 * 4, 4},
      {reg_s6, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 6 * 4, 4},
      {reg_s7, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 7 * 4, 4},
      {reg_s8, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 8 * 4, 4},
      {reg_s9, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 9 * 4, 4},
      {reg_s10, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 10 * 4,
       4},
      {reg_s11, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 11 * 4,
       4},
      {reg_s12, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 12 * 4,
       4},
      {reg_s13, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 13 * 4,
       4},
      {reg_s14, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 14 * 4,
       4},
      {reg_s15, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 15 * 4,
       4},
      {reg_s16, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 16 * 4,
       4},
      {reg_s17, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 17 * 4,
       4},
      {reg_s18, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 18 * 4,
       4},
      {reg_s19, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 19 * 4,
       4},
      {reg_s20, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 20 * 4,
       4},
      {reg_s21, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 21 * 4,
       4},
      {reg_s22, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 22 * 4,
       4},
      {reg_s23, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 23 * 4,
       4},
      {reg_s24, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 24 * 4,
       4},
      {reg_s25, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 25 * 4,
       4},
      {reg_s26, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 26 * 4,
       4},
      {reg_s27, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 27 * 4,
       4},
      {reg_s28, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 28 * 4,
       4},
      {reg_s29, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 29 * 4,
       4},
      {reg_s30, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 30 * 4,
       4},
      {reg_s31, (offsetof(RegisterContextMinidump_ARM::Context, s)) + 31 * 4,
       4},
      {reg_q0, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 0 * 16,
       16},
      {reg_q1, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 1 * 16,
       16},
      {reg_q2, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 2 * 16,
       16},
      {reg_q3, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 3 * 16,
       16},
      {reg_q4, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 4 * 16,
       16},
      {reg_q5, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 5 * 16,
       16},
      {reg_q6, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 6 * 16,
       16},
      {reg_q7, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 7 * 16,
       16},
      {reg_q8, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 8 * 16,
       16},
      {reg_q9, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 9 * 16,
       16},
      {reg_q10, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 10 * 16,
       16},
      {reg_q11, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 11 * 16,
       16},
      {reg_q12, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 12 * 16,
       16},
      {reg_q13, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 13 * 16,
       16},
      {reg_q14, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 14 * 16,
       16},
      {reg_q15, (offsetof(RegisterContextMinidump_ARM::Context, q)) + 15 * 16,
       16},
  };
  for (const auto &field : fields) {
    RegisterLocation location;
    location.kind = RegisterLocation::Stored;
    location.pieces.push_back({0, field.offset, field.size});
    llvm::cantFail(layout.Add(field.id, std::move(location)));
  }
  return llvm::cantFail(
      RegisterLayout::Create(std::move(description), std::move(layout)));
}

RegisterContextMinidump_ARM::RegisterContextMinidump_ARM(
    lldb_private::Thread &thread, const DataExtractor &data, bool apple)
    : ThreadRegisterContext(thread, 0, CreateMinidumpArmRegisterLayout(apple)) {
  auto bytes = data.GetData().take_front(data.GetByteSize());
  m_context.assign(bytes.begin(), bytes.end());
}

// Used for unit testing so we can verify register info is filled in for
// all register flavors (DWARF, EH Frame, generic, etc).

llvm::Expected<RegisterValue>
RegisterContextMinidump_ARM::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  auto value = GetLayout().Read(reg_info->id, {{0, llvm::ArrayRef(m_context)}});
  return value;
}

llvm::Error
RegisterContextMinidump_ARM::WriteRegister(RegisterID id,
                                           const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}
