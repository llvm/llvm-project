//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

static uint32_t g_s0_invalidates[] = {lldb_private::registers::arm::id::fpu_d0,
                                      lldb_private::registers::arm::id::fpu_q0};
static uint32_t g_s1_invalidates[] = {lldb_private::registers::arm::id::fpu_d0,
                                      lldb_private::registers::arm::id::fpu_q0};
static uint32_t g_s2_invalidates[] = {lldb_private::registers::arm::id::fpu_d1,
                                      lldb_private::registers::arm::id::fpu_q0};
static uint32_t g_s3_invalidates[] = {lldb_private::registers::arm::id::fpu_d1,
                                      lldb_private::registers::arm::id::fpu_q0};
static uint32_t g_s4_invalidates[] = {lldb_private::registers::arm::id::fpu_d2,
                                      lldb_private::registers::arm::id::fpu_q1};
static uint32_t g_s5_invalidates[] = {lldb_private::registers::arm::id::fpu_d2,
                                      lldb_private::registers::arm::id::fpu_q1};
static uint32_t g_s6_invalidates[] = {lldb_private::registers::arm::id::fpu_d3,
                                      lldb_private::registers::arm::id::fpu_q1};
static uint32_t g_s7_invalidates[] = {lldb_private::registers::arm::id::fpu_d3,
                                      lldb_private::registers::arm::id::fpu_q1};
static uint32_t g_s8_invalidates[] = {lldb_private::registers::arm::id::fpu_d4,
                                      lldb_private::registers::arm::id::fpu_q2};
static uint32_t g_s9_invalidates[] = {lldb_private::registers::arm::id::fpu_d4,
                                      lldb_private::registers::arm::id::fpu_q2};
static uint32_t g_s10_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d5,
    lldb_private::registers::arm::id::fpu_q2};
static uint32_t g_s11_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d5,
    lldb_private::registers::arm::id::fpu_q2};
static uint32_t g_s12_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d6,
    lldb_private::registers::arm::id::fpu_q3};
static uint32_t g_s13_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d6,
    lldb_private::registers::arm::id::fpu_q3};
static uint32_t g_s14_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d7,
    lldb_private::registers::arm::id::fpu_q3};
static uint32_t g_s15_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d7,
    lldb_private::registers::arm::id::fpu_q3};
static uint32_t g_s16_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d8,
    lldb_private::registers::arm::id::fpu_q4};
static uint32_t g_s17_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d8,
    lldb_private::registers::arm::id::fpu_q4};
static uint32_t g_s18_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d9,
    lldb_private::registers::arm::id::fpu_q4};
static uint32_t g_s19_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d9,
    lldb_private::registers::arm::id::fpu_q4};
static uint32_t g_s20_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d10,
    lldb_private::registers::arm::id::fpu_q5};
static uint32_t g_s21_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d10,
    lldb_private::registers::arm::id::fpu_q5};
static uint32_t g_s22_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d11,
    lldb_private::registers::arm::id::fpu_q5};
static uint32_t g_s23_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d11,
    lldb_private::registers::arm::id::fpu_q5};
static uint32_t g_s24_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d12,
    lldb_private::registers::arm::id::fpu_q6};
static uint32_t g_s25_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d12,
    lldb_private::registers::arm::id::fpu_q6};
static uint32_t g_s26_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d13,
    lldb_private::registers::arm::id::fpu_q6};
static uint32_t g_s27_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d13,
    lldb_private::registers::arm::id::fpu_q6};
static uint32_t g_s28_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d14,
    lldb_private::registers::arm::id::fpu_q7};
static uint32_t g_s29_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d14,
    lldb_private::registers::arm::id::fpu_q7};
static uint32_t g_s30_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d15,
    lldb_private::registers::arm::id::fpu_q7};
static uint32_t g_s31_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d15,
    lldb_private::registers::arm::id::fpu_q7};

static uint32_t g_d0_invalidates[] = {lldb_private::registers::arm::id::fpu_q0,
                                      lldb_private::registers::arm::id::fpu_s0,
                                      lldb_private::registers::arm::id::fpu_s1};
static uint32_t g_d1_invalidates[] = {lldb_private::registers::arm::id::fpu_q0,
                                      lldb_private::registers::arm::id::fpu_s2,
                                      lldb_private::registers::arm::id::fpu_s3};
static uint32_t g_d2_invalidates[] = {lldb_private::registers::arm::id::fpu_q1,
                                      lldb_private::registers::arm::id::fpu_s4,
                                      lldb_private::registers::arm::id::fpu_s5};
static uint32_t g_d3_invalidates[] = {lldb_private::registers::arm::id::fpu_q1,
                                      lldb_private::registers::arm::id::fpu_s6,
                                      lldb_private::registers::arm::id::fpu_s7};
static uint32_t g_d4_invalidates[] = {lldb_private::registers::arm::id::fpu_q2,
                                      lldb_private::registers::arm::id::fpu_s8,
                                      lldb_private::registers::arm::id::fpu_s9};
static uint32_t g_d5_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q2,
    lldb_private::registers::arm::id::fpu_s10,
    lldb_private::registers::arm::id::fpu_s11};
static uint32_t g_d6_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q3,
    lldb_private::registers::arm::id::fpu_s12,
    lldb_private::registers::arm::id::fpu_s13};
static uint32_t g_d7_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q3,
    lldb_private::registers::arm::id::fpu_s14,
    lldb_private::registers::arm::id::fpu_s15};
static uint32_t g_d8_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q4,
    lldb_private::registers::arm::id::fpu_s16,
    lldb_private::registers::arm::id::fpu_s17};
static uint32_t g_d9_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q4,
    lldb_private::registers::arm::id::fpu_s18,
    lldb_private::registers::arm::id::fpu_s19};
static uint32_t g_d10_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q5,
    lldb_private::registers::arm::id::fpu_s20,
    lldb_private::registers::arm::id::fpu_s21};
static uint32_t g_d11_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q5,
    lldb_private::registers::arm::id::fpu_s22,
    lldb_private::registers::arm::id::fpu_s23};
static uint32_t g_d12_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q6,
    lldb_private::registers::arm::id::fpu_s24,
    lldb_private::registers::arm::id::fpu_s25};
static uint32_t g_d13_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q6,
    lldb_private::registers::arm::id::fpu_s26,
    lldb_private::registers::arm::id::fpu_s27};
static uint32_t g_d14_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q7,
    lldb_private::registers::arm::id::fpu_s28,
    lldb_private::registers::arm::id::fpu_s29};
static uint32_t g_d15_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q7,
    lldb_private::registers::arm::id::fpu_s30,
    lldb_private::registers::arm::id::fpu_s31};
static uint32_t g_d16_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q8};
static uint32_t g_d17_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q8};
static uint32_t g_d18_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q9};
static uint32_t g_d19_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q9};
static uint32_t g_d20_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q10};
static uint32_t g_d21_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q10};
static uint32_t g_d22_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q11};
static uint32_t g_d23_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q11};
static uint32_t g_d24_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q12};
static uint32_t g_d25_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q12};
static uint32_t g_d26_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q13};
static uint32_t g_d27_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q13};
static uint32_t g_d28_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q14};
static uint32_t g_d29_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q14};
static uint32_t g_d30_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q15};
static uint32_t g_d31_invalidates[] = {
    lldb_private::registers::arm::id::fpu_q15};

static uint32_t g_q0_invalidates[] = {lldb_private::registers::arm::id::fpu_d0,
                                      lldb_private::registers::arm::id::fpu_d1,
                                      lldb_private::registers::arm::id::fpu_s0,
                                      lldb_private::registers::arm::id::fpu_s1,
                                      lldb_private::registers::arm::id::fpu_s2,
                                      lldb_private::registers::arm::id::fpu_s3};
static uint32_t g_q1_invalidates[] = {lldb_private::registers::arm::id::fpu_d2,
                                      lldb_private::registers::arm::id::fpu_d3,
                                      lldb_private::registers::arm::id::fpu_s4,
                                      lldb_private::registers::arm::id::fpu_s5,
                                      lldb_private::registers::arm::id::fpu_s6,
                                      lldb_private::registers::arm::id::fpu_s7};
static uint32_t g_q2_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d4,
    lldb_private::registers::arm::id::fpu_d5,
    lldb_private::registers::arm::id::fpu_s8,
    lldb_private::registers::arm::id::fpu_s9,
    lldb_private::registers::arm::id::fpu_s10,
    lldb_private::registers::arm::id::fpu_s11};
static uint32_t g_q3_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d6,
    lldb_private::registers::arm::id::fpu_d7,
    lldb_private::registers::arm::id::fpu_s12,
    lldb_private::registers::arm::id::fpu_s13,
    lldb_private::registers::arm::id::fpu_s14,
    lldb_private::registers::arm::id::fpu_s15};
static uint32_t g_q4_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d8,
    lldb_private::registers::arm::id::fpu_d9,
    lldb_private::registers::arm::id::fpu_s16,
    lldb_private::registers::arm::id::fpu_s17,
    lldb_private::registers::arm::id::fpu_s18,
    lldb_private::registers::arm::id::fpu_s19};
static uint32_t g_q5_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d10,
    lldb_private::registers::arm::id::fpu_d11,
    lldb_private::registers::arm::id::fpu_s20,
    lldb_private::registers::arm::id::fpu_s21,
    lldb_private::registers::arm::id::fpu_s22,
    lldb_private::registers::arm::id::fpu_s23};
static uint32_t g_q6_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d12,
    lldb_private::registers::arm::id::fpu_d13,
    lldb_private::registers::arm::id::fpu_s24,
    lldb_private::registers::arm::id::fpu_s25,
    lldb_private::registers::arm::id::fpu_s26,
    lldb_private::registers::arm::id::fpu_s27};
static uint32_t g_q7_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d14,
    lldb_private::registers::arm::id::fpu_d15,
    lldb_private::registers::arm::id::fpu_s28,
    lldb_private::registers::arm::id::fpu_s29,
    lldb_private::registers::arm::id::fpu_s30,
    lldb_private::registers::arm::id::fpu_s31};
static uint32_t g_q8_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d16,
    lldb_private::registers::arm::id::fpu_d17};
static uint32_t g_q9_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d18,
    lldb_private::registers::arm::id::fpu_d19};
static uint32_t g_q10_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d20,
    lldb_private::registers::arm::id::fpu_d21};
static uint32_t g_q11_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d22,
    lldb_private::registers::arm::id::fpu_d23};
static uint32_t g_q12_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d24,
    lldb_private::registers::arm::id::fpu_d25};
static uint32_t g_q13_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d26,
    lldb_private::registers::arm::id::fpu_d27};
static uint32_t g_q14_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d28,
    lldb_private::registers::arm::id::fpu_d29};
static uint32_t g_q15_invalidates[] = {
    lldb_private::registers::arm::id::fpu_d30,
    lldb_private::registers::arm::id::fpu_d31};

static uint32_t g_q0_contained[] = {lldb_private::registers::arm::id::fpu_q0};
static uint32_t g_q1_contained[] = {lldb_private::registers::arm::id::fpu_q1};
static uint32_t g_q2_contained[] = {lldb_private::registers::arm::id::fpu_q2};
static uint32_t g_q3_contained[] = {lldb_private::registers::arm::id::fpu_q3};
static uint32_t g_q4_contained[] = {lldb_private::registers::arm::id::fpu_q4};
static uint32_t g_q5_contained[] = {lldb_private::registers::arm::id::fpu_q5};
static uint32_t g_q6_contained[] = {lldb_private::registers::arm::id::fpu_q6};
static uint32_t g_q7_contained[] = {lldb_private::registers::arm::id::fpu_q7};
static uint32_t g_q8_contained[] = {lldb_private::registers::arm::id::fpu_q8};
static uint32_t g_q9_contained[] = {lldb_private::registers::arm::id::fpu_q9};
static uint32_t g_q10_contained[] = {lldb_private::registers::arm::id::fpu_q10};
static uint32_t g_q11_contained[] = {lldb_private::registers::arm::id::fpu_q11};
static uint32_t g_q12_contained[] = {lldb_private::registers::arm::id::fpu_q12};
static uint32_t g_q13_contained[] = {lldb_private::registers::arm::id::fpu_q13};
static uint32_t g_q14_contained[] = {lldb_private::registers::arm::id::fpu_q14};
static uint32_t g_q15_contained[] = {lldb_private::registers::arm::id::fpu_q15};

#define FPU_REG(name, size, qreg)                                              \
  RegisterDescriptor(lldb_private::registers::arm::id::fpu_##name, #name, nullptr, size,                   \
                     lldb::eEncodingIEEE754, lldb::eFormatFloat,               \
                     {LLDB_INVALID_REGNUM, lldb_private::registers::arm::dwarf::name, LLDB_INVALID_REGNUM,  \
                      LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::fpu_##name},                  \
                     g_##qreg##_contained, g_##name##_invalidates, nullptr)

#define FPU_QREG(name)                                                         \
  RegisterDescriptor(lldb_private::registers::arm::id::fpu_##name, #name, nullptr, 16,                     \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     {LLDB_INVALID_REGNUM, lldb_private::registers::arm::dwarf::name, LLDB_INVALID_REGNUM,  \
                      LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::fpu_##name},                  \
                     {}, g_##name##_invalidates, nullptr)

static const RegisterDescriptor g_register_infos_arm[] = {
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_r0, "r0", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::r0, lldb_private::registers::arm::dwarf::r0, LLDB_REGNUM_GENERIC_ARG1,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r0},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_r1, "r1", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::r1, lldb_private::registers::arm::dwarf::r1, LLDB_REGNUM_GENERIC_ARG2,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r1},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_r2, "r2", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::r2, lldb_private::registers::arm::dwarf::r2, LLDB_REGNUM_GENERIC_ARG3,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r2},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_r3, "r3", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::r3, lldb_private::registers::arm::dwarf::r3, LLDB_REGNUM_GENERIC_ARG4,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r3},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_r4, "r4", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::r4, lldb_private::registers::arm::dwarf::r4, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r4},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_r5, "r5", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::r5, lldb_private::registers::arm::dwarf::r5, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r5},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_r6, "r6", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::r6, lldb_private::registers::arm::dwarf::r6, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r6},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_r7, "r7", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::r7, lldb_private::registers::arm::dwarf::r7, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r7},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_r8, "r8", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::r8, lldb_private::registers::arm::dwarf::r8, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r8},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_r9, "r9", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::r9, lldb_private::registers::arm::dwarf::r9, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r9},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_r10, "r10", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::r10, lldb_private::registers::arm::dwarf::r10, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r10},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_r11, "r11", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::r11, lldb_private::registers::arm::dwarf::r11, LLDB_REGNUM_GENERIC_FP,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r11},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_r12, "r12", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::r12, lldb_private::registers::arm::dwarf::r12, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_r12},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_sp, "sp", "r13", 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::sp, lldb_private::registers::arm::dwarf::sp, LLDB_REGNUM_GENERIC_SP,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_sp},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_lr, "lr", "r14", 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::lr, lldb_private::registers::arm::dwarf::lr, LLDB_REGNUM_GENERIC_RA,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_lr},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_pc, "pc", "r15", 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::pc, lldb_private::registers::arm::dwarf::pc, LLDB_REGNUM_GENERIC_PC,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_pc},
                       {}, {}, nullptr),
    RegisterDescriptor(lldb_private::registers::arm::id::gpr_cpsr, "cpsr", "psr", 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {lldb_private::registers::arm::ehframe::cpsr, lldb_private::registers::arm::dwarf::cpsr, LLDB_REGNUM_GENERIC_FLAGS,
                        LLDB_INVALID_REGNUM, lldb_private::registers::arm::id::gpr_cpsr},
                       {}, {}, nullptr),

    FPU_REG(s0, 4, q0),
    FPU_REG(s1, 4, q0),
    FPU_REG(s2, 4, q0),
    FPU_REG(s3, 4, q0),
    FPU_REG(s4, 4, q1),
    FPU_REG(s5, 4, q1),
    FPU_REG(s6, 4, q1),
    FPU_REG(s7, 4, q1),
    FPU_REG(s8, 4, q2),
    FPU_REG(s9, 4, q2),
    FPU_REG(s10, 4, q2),
    FPU_REG(s11, 4, q2),
    FPU_REG(s12, 4, q3),
    FPU_REG(s13, 4, q3),
    FPU_REG(s14, 4, q3),
    FPU_REG(s15, 4, q3),
    FPU_REG(s16, 4, q4),
    FPU_REG(s17, 4, q4),
    FPU_REG(s18, 4, q4),
    FPU_REG(s19, 4, q4),
    FPU_REG(s20, 4, q5),
    FPU_REG(s21, 4, q5),
    FPU_REG(s22, 4, q5),
    FPU_REG(s23, 4, q5),
    FPU_REG(s24, 4, q6),
    FPU_REG(s25, 4, q6),
    FPU_REG(s26, 4, q6),
    FPU_REG(s27, 4, q6),
    FPU_REG(s28, 4, q7),
    FPU_REG(s29, 4, q7),
    FPU_REG(s30, 4, q7),
    FPU_REG(s31, 4, q7),

    RegisterDescriptor(lldb_private::registers::arm::id::fpu_fpscr, "fpscr", nullptr, 4, lldb::eEncodingUint,
                       lldb::eFormatHex,
                       {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                        lldb_private::registers::arm::id::fpu_fpscr},
                       {}, {}, nullptr),

    FPU_REG(d0, 8, q0),
    FPU_REG(d1, 8, q0),
    FPU_REG(d2, 8, q1),
    FPU_REG(d3, 8, q1),
    FPU_REG(d4, 8, q2),
    FPU_REG(d5, 8, q2),
    FPU_REG(d6, 8, q3),
    FPU_REG(d7, 8, q3),
    FPU_REG(d8, 8, q4),
    FPU_REG(d9, 8, q4),
    FPU_REG(d10, 8, q5),
    FPU_REG(d11, 8, q5),
    FPU_REG(d12, 8, q6),
    FPU_REG(d13, 8, q6),
    FPU_REG(d14, 8, q7),
    FPU_REG(d15, 8, q7),
    FPU_REG(d16, 8, q8),
    FPU_REG(d17, 8, q8),
    FPU_REG(d18, 8, q9),
    FPU_REG(d19, 8, q9),
    FPU_REG(d20, 8, q10),
    FPU_REG(d21, 8, q10),
    FPU_REG(d22, 8, q11),
    FPU_REG(d23, 8, q11),
    FPU_REG(d24, 8, q12),
    FPU_REG(d25, 8, q12),
    FPU_REG(d26, 8, q13),
    FPU_REG(d27, 8, q13),
    FPU_REG(d28, 8, q14),
    FPU_REG(d29, 8, q14),
    FPU_REG(d30, 8, q15),
    FPU_REG(d31, 8, q15),

    FPU_QREG(q0),
    FPU_QREG(q1),
    FPU_QREG(q2),
    FPU_QREG(q3),
    FPU_QREG(q4),
    FPU_QREG(q5),
    FPU_QREG(q6),
    FPU_QREG(q7),
    FPU_QREG(q8),
    FPU_QREG(q9),
    FPU_QREG(q10),
    FPU_QREG(q11),
    FPU_QREG(q12),
    FPU_QREG(q13),
    FPU_QREG(q14),
    FPU_QREG(q15),

    RegisterDescriptor(lldb_private::registers::arm::id::tls_tpidruro, "tpidruro", nullptr, 4,
                       lldb::eEncodingUint, lldb::eFormatHex,
                       {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                        LLDB_REGNUM_GENERIC_TP, LLDB_INVALID_REGNUM,
                        lldb_private::registers::arm::id::tls_tpidruro},
                       {}, {}, nullptr),

};
#undef FPU_REG
#undef FPU_QREG
