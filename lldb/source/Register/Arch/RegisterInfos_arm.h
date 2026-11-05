//===-- RegisterInfos_arm.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Textual register-table fragment; intentionally has no include guard.
// Include its ordinary dependencies at file scope before defining layout macros.
// See README.md for the inclusion contract.

#if !defined(DECLARE_REGISTER_INFOS_ARM_STRUCT)
#error Select a register-table generation mode before including this fragment
#endif

#ifdef DECLARE_REGISTER_INFOS_ARM_STRUCT


#ifndef GPR_OFFSET
#error GPR_OFFSET must be defined before including this header file
#endif

#ifndef FPU_OFFSET
#error FPU_OFFSET must be defined before including this header file
#endif

#ifndef FPSCR_OFFSET
#error FPSCR_OFFSET must be defined before including this header file
#endif

#ifndef TLS_OFFSET
#error TLS_OFFSET must be defined before including this header file
#endif

#ifndef EXC_OFFSET
#error EXC_OFFSET must be defined before including this header file
#endif

#ifndef DEFINE_DBG
#error DEFINE_DBG must be defined before including this header file
#endif

static uint32_t g_s0_invalidates[] = {lldb_private::registers::arm::id::fpu_d0,
                                      lldb_private::registers::arm::id::fpu_q0,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_s1_invalidates[] = {lldb_private::registers::arm::id::fpu_d0,
                                      lldb_private::registers::arm::id::fpu_q0,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_s2_invalidates[] = {lldb_private::registers::arm::id::fpu_d1,
                                      lldb_private::registers::arm::id::fpu_q0,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_s3_invalidates[] = {lldb_private::registers::arm::id::fpu_d1,
                                      lldb_private::registers::arm::id::fpu_q0,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_s4_invalidates[] = {lldb_private::registers::arm::id::fpu_d2,
                                      lldb_private::registers::arm::id::fpu_q1,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_s5_invalidates[] = {lldb_private::registers::arm::id::fpu_d2,
                                      lldb_private::registers::arm::id::fpu_q1,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_s6_invalidates[] = {lldb_private::registers::arm::id::fpu_d3,
                                      lldb_private::registers::arm::id::fpu_q1,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_s7_invalidates[] = {lldb_private::registers::arm::id::fpu_d3,
                                      lldb_private::registers::arm::id::fpu_q1,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_s8_invalidates[] = {lldb_private::registers::arm::id::fpu_d4,
                                      lldb_private::registers::arm::id::fpu_q2,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_s9_invalidates[] = {lldb_private::registers::arm::id::fpu_d4,
                                      lldb_private::registers::arm::id::fpu_q2,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_s10_invalidates[] = {lldb_private::registers::arm::id::fpu_d5,
                                       lldb_private::registers::arm::id::fpu_q2,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_s11_invalidates[] = {lldb_private::registers::arm::id::fpu_d5,
                                       lldb_private::registers::arm::id::fpu_q2,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_s12_invalidates[] = {lldb_private::registers::arm::id::fpu_d6,
                                       lldb_private::registers::arm::id::fpu_q3,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_s13_invalidates[] = {lldb_private::registers::arm::id::fpu_d6,
                                       lldb_private::registers::arm::id::fpu_q3,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_s14_invalidates[] = {lldb_private::registers::arm::id::fpu_d7,
                                       lldb_private::registers::arm::id::fpu_q3,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_s15_invalidates[] = {lldb_private::registers::arm::id::fpu_d7,
                                       lldb_private::registers::arm::id::fpu_q3,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_s16_invalidates[] = {lldb_private::registers::arm::id::fpu_d8,
                                       lldb_private::registers::arm::id::fpu_q4,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_s17_invalidates[] = {lldb_private::registers::arm::id::fpu_d8,
                                       lldb_private::registers::arm::id::fpu_q4,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_s18_invalidates[] = {lldb_private::registers::arm::id::fpu_d9,
                                       lldb_private::registers::arm::id::fpu_q4,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_s19_invalidates[] = {lldb_private::registers::arm::id::fpu_d9,
                                       lldb_private::registers::arm::id::fpu_q4,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_s20_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d10,
    lldb_private::registers::arm::id::fpu_q5, LLDB_INVALID_REGNUM};
static uint32_t g_s21_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d10,
    lldb_private::registers::arm::id::fpu_q5, LLDB_INVALID_REGNUM};
static uint32_t g_s22_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d11,
    lldb_private::registers::arm::id::fpu_q5, LLDB_INVALID_REGNUM};
static uint32_t g_s23_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d11,
    lldb_private::registers::arm::id::fpu_q5, LLDB_INVALID_REGNUM};
static uint32_t g_s24_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d12,
    lldb_private::registers::arm::id::fpu_q6, LLDB_INVALID_REGNUM};
static uint32_t g_s25_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d12,
    lldb_private::registers::arm::id::fpu_q6, LLDB_INVALID_REGNUM};
static uint32_t g_s26_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d13,
    lldb_private::registers::arm::id::fpu_q6, LLDB_INVALID_REGNUM};
static uint32_t g_s27_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d13,
    lldb_private::registers::arm::id::fpu_q6, LLDB_INVALID_REGNUM};
static uint32_t g_s28_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d14,
    lldb_private::registers::arm::id::fpu_q7, LLDB_INVALID_REGNUM};
static uint32_t g_s29_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d14,
    lldb_private::registers::arm::id::fpu_q7, LLDB_INVALID_REGNUM};
static uint32_t g_s30_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d15,
    lldb_private::registers::arm::id::fpu_q7, LLDB_INVALID_REGNUM};
static uint32_t g_s31_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d15,
    lldb_private::registers::arm::id::fpu_q7, LLDB_INVALID_REGNUM};

static uint32_t g_d0_invalidates[] = {lldb_private::registers::arm::id::fpu_q0,
                                      lldb_private::registers::arm::id::fpu_s0,
                                      lldb_private::registers::arm::id::fpu_s1,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_d1_invalidates[] = {lldb_private::registers::arm::id::fpu_q0,
                                      lldb_private::registers::arm::id::fpu_s2,
                                      lldb_private::registers::arm::id::fpu_s3,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_d2_invalidates[] = {lldb_private::registers::arm::id::fpu_q1,
                                      lldb_private::registers::arm::id::fpu_s4,
                                      lldb_private::registers::arm::id::fpu_s5,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_d3_invalidates[] = {lldb_private::registers::arm::id::fpu_q1,
                                      lldb_private::registers::arm::id::fpu_s6,
                                      lldb_private::registers::arm::id::fpu_s7,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_d4_invalidates[] = {lldb_private::registers::arm::id::fpu_q2,
                                      lldb_private::registers::arm::id::fpu_s8,
                                      lldb_private::registers::arm::id::fpu_s9,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_d5_invalidates[] = {lldb_private::registers::arm::id::fpu_q2,
                                      lldb_private::registers::arm::id::fpu_s10,
                                      lldb_private::registers::arm::id::fpu_s11,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_d6_invalidates[] = {lldb_private::registers::arm::id::fpu_q3,
                                      lldb_private::registers::arm::id::fpu_s12,
                                      lldb_private::registers::arm::id::fpu_s13,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_d7_invalidates[] = {lldb_private::registers::arm::id::fpu_q3,
                                      lldb_private::registers::arm::id::fpu_s14,
                                      lldb_private::registers::arm::id::fpu_s15,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_d8_invalidates[] = {lldb_private::registers::arm::id::fpu_q4,
                                      lldb_private::registers::arm::id::fpu_s16,
                                      lldb_private::registers::arm::id::fpu_s17,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_d9_invalidates[] = {lldb_private::registers::arm::id::fpu_q4,
                                      lldb_private::registers::arm::id::fpu_s18,
                                      lldb_private::registers::arm::id::fpu_s19,
                                      LLDB_INVALID_REGNUM};
static uint32_t g_d10_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q5,
    lldb_private::registers::arm::id::fpu_s20,
    lldb_private::registers::arm::id::fpu_s21, LLDB_INVALID_REGNUM};
static uint32_t g_d11_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q5,
    lldb_private::registers::arm::id::fpu_s22,
    lldb_private::registers::arm::id::fpu_s23, LLDB_INVALID_REGNUM};
static uint32_t g_d12_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q6,
    lldb_private::registers::arm::id::fpu_s24,
    lldb_private::registers::arm::id::fpu_s25, LLDB_INVALID_REGNUM};
static uint32_t g_d13_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q6,
    lldb_private::registers::arm::id::fpu_s26,
    lldb_private::registers::arm::id::fpu_s27, LLDB_INVALID_REGNUM};
static uint32_t g_d14_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q7,
    lldb_private::registers::arm::id::fpu_s28,
    lldb_private::registers::arm::id::fpu_s29, LLDB_INVALID_REGNUM};
static uint32_t g_d15_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q7,
    lldb_private::registers::arm::id::fpu_s30,
    lldb_private::registers::arm::id::fpu_s31, LLDB_INVALID_REGNUM};
static uint32_t g_d16_invalidates[] = {lldb_private::registers::arm::id::fpu_q8,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_d17_invalidates[] = {lldb_private::registers::arm::id::fpu_q8,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_d18_invalidates[] = {lldb_private::registers::arm::id::fpu_q9,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_d19_invalidates[] = {lldb_private::registers::arm::id::fpu_q9,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_d14_invalidates[] = {lldb_private::registers::arm::id::fpu_q7, lldb_private::registers::arm::id::fpu_s28, lldb_private::registers::arm::id::fpu_s29,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_d15_invalidates[] = {lldb_private::registers::arm::id::fpu_q7, lldb_private::registers::arm::id::fpu_s30, lldb_private::registers::arm::id::fpu_s31,
                                       LLDB_INVALID_REGNUM};
static uint32_t g_d16_invalidates[] = {lldb_private::registers::arm::id::fpu_q8, LLDB_INVALID_REGNUM};
static uint32_t g_d17_invalidates[] = {lldb_private::registers::arm::id::fpu_q8, LLDB_INVALID_REGNUM};
static uint32_t g_d18_invalidates[] = {lldb_private::registers::arm::id::fpu_q9, LLDB_INVALID_REGNUM};
static uint32_t g_d19_invalidates[] = {lldb_private::registers::arm::id::fpu_q9, LLDB_INVALID_REGNUM};
static uint32_t g_d20_invalidates[] = {lldb_private::registers::arm::id::fpu_q10, LLDB_INVALID_REGNUM};
static uint32_t g_d21_invalidates[] = {lldb_private::registers::arm::id::fpu_q10, LLDB_INVALID_REGNUM};
static uint32_t g_d22_invalidates[] = {lldb_private::registers::arm::id::fpu_q11, LLDB_INVALID_REGNUM};
static uint32_t g_d23_invalidates[] = {lldb_private::registers::arm::id::fpu_q11, LLDB_INVALID_REGNUM};
static uint32_t g_d24_invalidates[] = {lldb_private::registers::arm::id::fpu_q12, LLDB_INVALID_REGNUM};
static uint32_t g_d25_invalidates[] = {lldb_private::registers::arm::id::fpu_q12, LLDB_INVALID_REGNUM};
static uint32_t g_d26_invalidates[] = {lldb_private::registers::arm::id::fpu_q13, LLDB_INVALID_REGNUM};
static uint32_t g_d27_invalidates[] = {lldb_private::registers::arm::id::fpu_q13, LLDB_INVALID_REGNUM};
static uint32_t g_d28_invalidates[] = {lldb_private::registers::arm::id::fpu_q14, LLDB_INVALID_REGNUM};
static uint32_t g_d29_invalidates[] = {lldb_private::registers::arm::id::fpu_q14, LLDB_INVALID_REGNUM};
static uint32_t g_d30_invalidates[] = {lldb_private::registers::arm::id::fpu_q15, LLDB_INVALID_REGNUM};
static uint32_t g_d31_invalidates[] = {lldb_private::registers::arm::id::fpu_q15, LLDB_INVALID_REGNUM};

static uint32_t g_q0_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d0, lldb_private::registers::arm::id::fpu_d1, lldb_private::registers::arm::id::fpu_s0, lldb_private::registers::arm::id::fpu_s1, lldb_private::registers::arm::id::fpu_s2, lldb_private::registers::arm::id::fpu_s3, LLDB_INVALID_REGNUM};
static uint32_t g_q1_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d2, lldb_private::registers::arm::id::fpu_d3, lldb_private::registers::arm::id::fpu_s4, lldb_private::registers::arm::id::fpu_s5, lldb_private::registers::arm::id::fpu_s6, lldb_private::registers::arm::id::fpu_s7, LLDB_INVALID_REGNUM};
static uint32_t g_q2_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d4, lldb_private::registers::arm::id::fpu_d5, lldb_private::registers::arm::id::fpu_s8, lldb_private::registers::arm::id::fpu_s9, lldb_private::registers::arm::id::fpu_s10, lldb_private::registers::arm::id::fpu_s11, LLDB_INVALID_REGNUM};
static uint32_t g_q3_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d6, lldb_private::registers::arm::id::fpu_d7, lldb_private::registers::arm::id::fpu_s12, lldb_private::registers::arm::id::fpu_s13, lldb_private::registers::arm::id::fpu_s14, lldb_private::registers::arm::id::fpu_s15, LLDB_INVALID_REGNUM};
static uint32_t g_q4_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d8, lldb_private::registers::arm::id::fpu_d9, lldb_private::registers::arm::id::fpu_s16, lldb_private::registers::arm::id::fpu_s17, lldb_private::registers::arm::id::fpu_s18, lldb_private::registers::arm::id::fpu_s19, LLDB_INVALID_REGNUM};
static uint32_t g_q5_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d10, lldb_private::registers::arm::id::fpu_d11, lldb_private::registers::arm::id::fpu_s20, lldb_private::registers::arm::id::fpu_s21, lldb_private::registers::arm::id::fpu_s22, lldb_private::registers::arm::id::fpu_s23, LLDB_INVALID_REGNUM};
static uint32_t g_q6_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d12, lldb_private::registers::arm::id::fpu_d13, lldb_private::registers::arm::id::fpu_s24, lldb_private::registers::arm::id::fpu_s25, lldb_private::registers::arm::id::fpu_s26, lldb_private::registers::arm::id::fpu_s27, LLDB_INVALID_REGNUM};
static uint32_t g_q7_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d14, lldb_private::registers::arm::id::fpu_d15, lldb_private::registers::arm::id::fpu_s28, lldb_private::registers::arm::id::fpu_s29, lldb_private::registers::arm::id::fpu_s30, lldb_private::registers::arm::id::fpu_s31, LLDB_INVALID_REGNUM};
static uint32_t g_q8_invalidates[] = {lldb_private::registers::arm::id::fpu_d16, lldb_private::registers::arm::id::fpu_d17, LLDB_INVALID_REGNUM};
static uint32_t g_q9_invalidates[] = {lldb_private::registers::arm::id::fpu_d18, lldb_private::registers::arm::id::fpu_d19, LLDB_INVALID_REGNUM};
static uint32_t g_q10_invalidates[] = {lldb_private::registers::arm::id::fpu_d20, lldb_private::registers::arm::id::fpu_d21, LLDB_INVALID_REGNUM};
static uint32_t g_q11_invalidates[] = {lldb_private::registers::arm::id::fpu_d22, lldb_private::registers::arm::id::fpu_d23, LLDB_INVALID_REGNUM};
static uint32_t g_q12_invalidates[] = {lldb_private::registers::arm::id::fpu_d24, lldb_private::registers::arm::id::fpu_d25, LLDB_INVALID_REGNUM};
static uint32_t g_q13_invalidates[] = {lldb_private::registers::arm::id::fpu_d26, lldb_private::registers::arm::id::fpu_d27, LLDB_INVALID_REGNUM};
static uint32_t g_q14_invalidates[] = {lldb_private::registers::arm::id::fpu_d28, lldb_private::registers::arm::id::fpu_d29, LLDB_INVALID_REGNUM};
static uint32_t g_q15_invalidates[] = {lldb_private::registers::arm::id::fpu_d30, lldb_private::registers::arm::id::fpu_d31, LLDB_INVALID_REGNUM};

static uint32_t g_q0_contained[] = {lldb_private::registers::arm::id::fpu_q0,
                                    LLDB_INVALID_REGNUM};
static uint32_t g_q1_contained[] = {lldb_private::registers::arm::id::fpu_q1,
                                    LLDB_INVALID_REGNUM};
static uint32_t g_q2_contained[] = {lldb_private::registers::arm::id::fpu_q2,
                                    LLDB_INVALID_REGNUM};
static uint32_t g_q3_contained[] = {lldb_private::registers::arm::id::fpu_q3,
                                    LLDB_INVALID_REGNUM};
static uint32_t g_q4_contained[] = {lldb_private::registers::arm::id::fpu_q4,
                                    LLDB_INVALID_REGNUM};
static uint32_t g_q5_contained[] = {lldb_private::registers::arm::id::fpu_q5,
                                    LLDB_INVALID_REGNUM};
static uint32_t g_q6_contained[] = {lldb_private::registers::arm::id::fpu_q6,
                                    LLDB_INVALID_REGNUM};
static uint32_t g_q7_contained[] = {lldb_private::registers::arm::id::fpu_q7,
                                    LLDB_INVALID_REGNUM};
static uint32_t g_q8_contained[] = {lldb_private::registers::arm::id::fpu_q8,
                                    LLDB_INVALID_REGNUM};
static uint32_t g_q9_contained[] = {lldb_private::registers::arm::id::fpu_q9,
                                    LLDB_INVALID_REGNUM};
static uint32_t g_q10_contained[] = {lldb_private::registers::arm::id::fpu_q10,
                                     LLDB_INVALID_REGNUM};
static uint32_t g_q11_contained[] = {lldb_private::registers::arm::id::fpu_q11,
                                     LLDB_INVALID_REGNUM};
static uint32_t g_q12_contained[] = {lldb_private::registers::arm::id::fpu_q12,
                                     LLDB_INVALID_REGNUM};
static uint32_t g_q13_contained[] = {lldb_private::registers::arm::id::fpu_q13,
                                     LLDB_INVALID_REGNUM};
static uint32_t g_q14_contained[] = {lldb_private::registers::arm::id::fpu_q14,
                                     LLDB_INVALID_REGNUM};
static uint32_t g_q15_contained[] = {lldb_private::registers::arm::id::fpu_q15,
                                     LLDB_INVALID_REGNUM};

#define FPU_REG(name, size, offset, qreg)                                      \
  {                                                                            \
    #name, nullptr, size, FPU_OFFSET(offset), lldb::eEncodingIEEE754, lldb::eFormatFloat,  \
        {LLDB_INVALID_REGNUM, lldb_private::registers::arm::dwarf::name, LLDB_INVALID_REGNUM,               \
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::fpu_##name },                                    \
         g_##qreg##_contained, g_##name##_invalidates, nullptr,                \
  }

#define FPU_QREG(name, offset)                                                 \
  {                                                                            \
    #name, nullptr, 16, FPU_OFFSET(offset), lldb::eEncodingVector,                   \
        lldb::eFormatVectorOfUInt8,                                                  \
        {LLDB_INVALID_REGNUM, lldb_private::registers::arm::dwarf::name, LLDB_INVALID_REGNUM,               \
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::fpu_##name },                                    \
         nullptr, g_##name##_invalidates, nullptr,                             \
  }

static RegisterInfo g_register_infos_arm[] = {
    {
        "r0",
        nullptr,
        4,
        GPR_OFFSET(0),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::r0,
         lldb_private::registers::arm::dwarf::r0, LLDB_REGNUM_GENERIC_ARG1,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r0},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "r1",
        nullptr,
        4,
        GPR_OFFSET(1),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::r1,
         lldb_private::registers::arm::dwarf::r1, LLDB_REGNUM_GENERIC_ARG2,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r1},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "r2",
        nullptr,
        4,
        GPR_OFFSET(2),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::r2,
         lldb_private::registers::arm::dwarf::r2, LLDB_REGNUM_GENERIC_ARG3,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r2},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "r3",
        nullptr,
        4,
        GPR_OFFSET(3),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::r3,
         lldb_private::registers::arm::dwarf::r3, LLDB_REGNUM_GENERIC_ARG4,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r3},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "r4",
        nullptr,
        4,
        GPR_OFFSET(4),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::r4,
         lldb_private::registers::arm::dwarf::r4, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r4},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "r5",
        nullptr,
        4,
        GPR_OFFSET(5),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::r5,
         lldb_private::registers::arm::dwarf::r5, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r5},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "r6",
        nullptr,
        4,
        GPR_OFFSET(6),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::r6,
         lldb_private::registers::arm::dwarf::r6, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r6},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "r7",
        nullptr,
        4,
        GPR_OFFSET(7),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::r7,
         lldb_private::registers::arm::dwarf::r7, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r7},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "r8",
        nullptr,
        4,
        GPR_OFFSET(8),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::r8,
         lldb_private::registers::arm::dwarf::r8, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r8},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "r9",
        nullptr,
        4,
        GPR_OFFSET(9),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::r9,
         lldb_private::registers::arm::dwarf::r9, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r9},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "r10",
        nullptr,
        4,
        GPR_OFFSET(10),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::r10,
         lldb_private::registers::arm::dwarf::r10, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r10},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "r11",
        nullptr,
        4,
        GPR_OFFSET(11),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::r11,
         lldb_private::registers::arm::dwarf::r11, LLDB_REGNUM_GENERIC_FP,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r11},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "r12",
        nullptr,
        4,
        GPR_OFFSET(12),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::r12,
         lldb_private::registers::arm::dwarf::r12, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r12},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "sp",
        "r13",
        4,
        GPR_OFFSET(13),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::sp,
         lldb_private::registers::arm::dwarf::sp, LLDB_REGNUM_GENERIC_SP,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_sp},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "lr",
        "r14",
        4,
        GPR_OFFSET(14),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::lr,
         lldb_private::registers::arm::dwarf::lr, LLDB_REGNUM_GENERIC_RA,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_lr},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "pc",
        "r15",
        4,
        GPR_OFFSET(15),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::pc,
         lldb_private::registers::arm::dwarf::pc, LLDB_REGNUM_GENERIC_PC,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_pc},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "cpsr",
        "psr",
        4,
        GPR_OFFSET(16),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {lldb_private::registers::arm::ehframe::cpsr,
         lldb_private::registers::arm::dwarf::cpsr, LLDB_REGNUM_GENERIC_FLAGS,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_cpsr},
        nullptr,
        nullptr,
        nullptr,
    },

    FPU_REG(s0, 4, 0, q0),
    FPU_REG(s1, 4, 1, q0),
    FPU_REG(s2, 4, 2, q0),
    FPU_REG(s3, 4, 3, q0),
    FPU_REG(s4, 4, 4, q1),
    FPU_REG(s5, 4, 5, q1),
    FPU_REG(s6, 4, 6, q1),
    FPU_REG(s7, 4, 7, q1),
    FPU_REG(s8, 4, 8, q2),
    FPU_REG(s9, 4, 9, q2),
    FPU_REG(s10, 4, 10, q2),
    FPU_REG(s11, 4, 11, q2),
    FPU_REG(s12, 4, 12, q3),
    FPU_REG(s13, 4, 13, q3),
    FPU_REG(s14, 4, 14, q3),
    FPU_REG(s15, 4, 15, q3),
    FPU_REG(s16, 4, 16, q4),
    FPU_REG(s17, 4, 17, q4),
    FPU_REG(s18, 4, 18, q4),
    FPU_REG(s19, 4, 19, q4),
    FPU_REG(s20, 4, 20, q5),
    FPU_REG(s21, 4, 21, q5),
    FPU_REG(s22, 4, 22, q5),
    FPU_REG(s23, 4, 23, q5),
    FPU_REG(s24, 4, 24, q6),
    FPU_REG(s25, 4, 25, q6),
    FPU_REG(s26, 4, 26, q6),
    FPU_REG(s27, 4, 27, q6),
    FPU_REG(s28, 4, 28, q7),
    FPU_REG(s29, 4, 29, q7),
    FPU_REG(s30, 4, 30, q7),
    FPU_REG(s31, 4, 31, q7),

    {
        "fpscr",
        nullptr,
        4,
        FPSCR_OFFSET,
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::fpu_fpscr},
        nullptr,
        nullptr,
        nullptr,
    },

    FPU_REG(d0, 8, 0, q0),
    FPU_REG(d1, 8, 2, q0),
    FPU_REG(d2, 8, 4, q1),
    FPU_REG(d3, 8, 6, q1),
    FPU_REG(d4, 8, 8, q2),
    FPU_REG(d5, 8, 10, q2),
    FPU_REG(d6, 8, 12, q3),
    FPU_REG(d7, 8, 14, q3),
    FPU_REG(d8, 8, 16, q4),
    FPU_REG(d9, 8, 18, q4),
    FPU_REG(d10, 8, 20, q5),
    FPU_REG(d11, 8, 22, q5),
    FPU_REG(d12, 8, 24, q6),
    FPU_REG(d13, 8, 26, q6),
    FPU_REG(d14, 8, 28, q7),
    FPU_REG(d15, 8, 30, q7),
    FPU_REG(d16, 8, 32, q8),
    FPU_REG(d17, 8, 34, q8),
    FPU_REG(d18, 8, 36, q9),
    FPU_REG(d19, 8, 38, q9),
    FPU_REG(d20, 8, 40, q10),
    FPU_REG(d21, 8, 42, q10),
    FPU_REG(d22, 8, 44, q11),
    FPU_REG(d23, 8, 46, q11),
    FPU_REG(d24, 8, 48, q12),
    FPU_REG(d25, 8, 50, q12),
    FPU_REG(d26, 8, 52, q13),
    FPU_REG(d27, 8, 54, q13),
    FPU_REG(d28, 8, 56, q14),
    FPU_REG(d29, 8, 58, q14),
    FPU_REG(d30, 8, 60, q15),
    FPU_REG(d31, 8, 62, q15),

    FPU_QREG(q0, 0),
    FPU_QREG(q1, 4),
    FPU_QREG(q2, 8),
    FPU_QREG(q3, 12),
    FPU_QREG(q4, 16),
    FPU_QREG(q5, 20),
    FPU_QREG(q6, 24),
    FPU_QREG(q7, 28),
    FPU_QREG(q8, 32),
    FPU_QREG(q9, 36),
    FPU_QREG(q10, 40),
    FPU_QREG(q11, 44),
    FPU_QREG(q12, 48),
    FPU_QREG(q13, 52),
    FPU_QREG(q14, 56),
    FPU_QREG(q15, 60),

    {
        "tpidruro",
        nullptr,
        4,
        TLS_OFFSET,
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_REGNUM_GENERIC_TP,
         LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::tls_tpidruro},
        nullptr,
        nullptr,
        nullptr,
    },

    {
        "exception",
        nullptr,
        4,
        EXC_OFFSET(0),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM,
         lldb_private::registers::arm::id::exc_exception_with_tls},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "fsr",
        nullptr,
        4,
        EXC_OFFSET(1),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM,
         lldb_private::registers::arm::id::exc_fsr_with_tls},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "far",
        nullptr,
        4,
        EXC_OFFSET(2),
        lldb::eEncodingUint,
        lldb::eFormatHex,
        {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM,
         lldb_private::registers::arm::id::exc_far_with_tls},
        nullptr,
        nullptr,
        nullptr,
    },

    {DEFINE_DBG(bvr, 0)},
    {DEFINE_DBG(bvr, 1)},
    {DEFINE_DBG(bvr, 2)},
    {DEFINE_DBG(bvr, 3)},
    {DEFINE_DBG(bvr, 4)},
    {DEFINE_DBG(bvr, 5)},
    {DEFINE_DBG(bvr, 6)},
    {DEFINE_DBG(bvr, 7)},
    {DEFINE_DBG(bvr, 8)},
    {DEFINE_DBG(bvr, 9)},
    {DEFINE_DBG(bvr, 10)},
    {DEFINE_DBG(bvr, 11)},
    {DEFINE_DBG(bvr, 12)},
    {DEFINE_DBG(bvr, 13)},
    {DEFINE_DBG(bvr, 14)},
    {DEFINE_DBG(bvr, 15)},

    {DEFINE_DBG(bcr, 0)},
    {DEFINE_DBG(bcr, 1)},
    {DEFINE_DBG(bcr, 2)},
    {DEFINE_DBG(bcr, 3)},
    {DEFINE_DBG(bcr, 4)},
    {DEFINE_DBG(bcr, 5)},
    {DEFINE_DBG(bcr, 6)},
    {DEFINE_DBG(bcr, 7)},
    {DEFINE_DBG(bcr, 8)},
    {DEFINE_DBG(bcr, 9)},
    {DEFINE_DBG(bcr, 10)},
    {DEFINE_DBG(bcr, 11)},
    {DEFINE_DBG(bcr, 12)},
    {DEFINE_DBG(bcr, 13)},
    {DEFINE_DBG(bcr, 14)},
    {DEFINE_DBG(bcr, 15)},

    {DEFINE_DBG(wvr, 0)},
    {DEFINE_DBG(wvr, 1)},
    {DEFINE_DBG(wvr, 2)},
    {DEFINE_DBG(wvr, 3)},
    {DEFINE_DBG(wvr, 4)},
    {DEFINE_DBG(wvr, 5)},
    {DEFINE_DBG(wvr, 6)},
    {DEFINE_DBG(wvr, 7)},
    {DEFINE_DBG(wvr, 8)},
    {DEFINE_DBG(wvr, 9)},
    {DEFINE_DBG(wvr, 10)},
    {DEFINE_DBG(wvr, 11)},
    {DEFINE_DBG(wvr, 12)},
    {DEFINE_DBG(wvr, 13)},
    {DEFINE_DBG(wvr, 14)},
    {DEFINE_DBG(wvr, 15)},

    {DEFINE_DBG(wcr, 0)},
    {DEFINE_DBG(wcr, 1)},
    {DEFINE_DBG(wcr, 2)},
    {DEFINE_DBG(wcr, 3)},
    {DEFINE_DBG(wcr, 4)},
    {DEFINE_DBG(wcr, 5)},
    {DEFINE_DBG(wcr, 6)},
    {DEFINE_DBG(wcr, 7)},
    {DEFINE_DBG(wcr, 8)},
    {DEFINE_DBG(wcr, 9)},
    {DEFINE_DBG(wcr, 10)},
    {DEFINE_DBG(wcr, 11)},
    {DEFINE_DBG(wcr, 12)},
    {DEFINE_DBG(wcr, 13)},
    {DEFINE_DBG(wcr, 14)},
    {DEFINE_DBG(wcr, 15)}};


#endif // DECLARE_REGISTER_INFOS_ARM_STRUCT

#undef FPU_REG
#undef FPU_QREG
