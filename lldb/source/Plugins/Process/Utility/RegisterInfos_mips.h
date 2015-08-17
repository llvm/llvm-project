//===-- RegisterInfos_mips.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
#include "llvm/Support/Compiler.h"

#include <stddef.h>

#ifdef DECLARE_REGISTER_INFOS_MIPS_STRUCT

// Computes the offset of the given GPR in the user data area.
#define GPR_OFFSET(regname) \
    (LLVM_EXTENSION offsetof(UserArea, gpr) + \
     LLVM_EXTENSION offsetof(GPR_linux_mips, regname))

// Computes the offset of the given FPR in the extended data area.
#define FPR_OFFSET(regname)  \
     (LLVM_EXTENSION offsetof(UserArea, fpr) + \
      LLVM_EXTENSION offsetof(FPR_linux_mips, regname))

// Computes the offset of the given MSA in the extended data area.
#define MSA_OFFSET(regname)  \
     (LLVM_EXTENSION offsetof(UserArea, msa) + \
      LLVM_EXTENSION offsetof(MSA_linux_mips, regname))

// Note that the size and offset will be updated by platform-specific classes.
#define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)            \
    { #reg, alt, sizeof(((GPR_linux_mips*)NULL)->reg) / 2, GPR_OFFSET(reg), eEncodingUint,  \
      eFormatHex, { kind1, kind2, kind3, kind4, gpr_##reg##_mips }, NULL, NULL }

#define DEFINE_FPR(reg, alt, kind1, kind2, kind3, kind4)           \
    { #reg, alt, sizeof(((FPR_linux_mips*)NULL)->reg), FPR_OFFSET(reg), eEncodingUint,   \
      eFormatHex, { kind1, kind2, kind3, kind4, fpr_##reg##_mips }, NULL, NULL }

#define DEFINE_MSA(reg, alt, kind1, kind2, kind3, kind4)    \
    { #reg, alt, sizeof(((MSA_linux_mips*)0)->reg), MSA_OFFSET(reg), eEncodingVector,   \
      eFormatVectorOfUInt8, { kind1, kind2, kind3, kind4, msa_##reg##_mips }, NULL, NULL }

#define DEFINE_MSA_INFO(reg, alt, kind1, kind2, kind3, kind4)    \
    { #reg, alt, sizeof(((MSA_linux_mips*)0)->reg), MSA_OFFSET(reg), eEncodingUint, \
      eFormatHex, { kind1, kind2, kind3, kind4, msa_##reg##_mips }, NULL, NULL }

// RegisterKind: GCC, DWARF, Generic, GDB, LLDB

static RegisterInfo
g_register_infos_mips[] =
{
    DEFINE_GPR (zero,     "zero", gcc_dwarf_zero_mips,  gcc_dwarf_zero_mips,  LLDB_INVALID_REGNUM,    gdb_zero_mips),
    DEFINE_GPR (r1,       "at",   gcc_dwarf_r1_mips,    gcc_dwarf_r1_mips,    LLDB_INVALID_REGNUM,    gdb_r1_mips),
    DEFINE_GPR (r2,       NULL,   gcc_dwarf_r2_mips,    gcc_dwarf_r2_mips,    LLDB_INVALID_REGNUM,    gdb_r2_mips),
    DEFINE_GPR (r3,       NULL,   gcc_dwarf_r3_mips,    gcc_dwarf_r3_mips,    LLDB_INVALID_REGNUM,    gdb_r3_mips),
    DEFINE_GPR (r4,       NULL,   gcc_dwarf_r4_mips,    gcc_dwarf_r4_mips,    LLDB_REGNUM_GENERIC_ARG1,    gdb_r4_mips),
    DEFINE_GPR (r5,       NULL,   gcc_dwarf_r5_mips,    gcc_dwarf_r5_mips,    LLDB_REGNUM_GENERIC_ARG2,    gdb_r5_mips),
    DEFINE_GPR (r6,       NULL,   gcc_dwarf_r6_mips,    gcc_dwarf_r6_mips,    LLDB_REGNUM_GENERIC_ARG3,    gdb_r6_mips),
    DEFINE_GPR (r7,       NULL,   gcc_dwarf_r7_mips,    gcc_dwarf_r7_mips,    LLDB_REGNUM_GENERIC_ARG4,    gdb_r7_mips),
    DEFINE_GPR (r8,       NULL,   gcc_dwarf_r8_mips,    gcc_dwarf_r8_mips,    LLDB_INVALID_REGNUM,    gdb_r8_mips),
    DEFINE_GPR (r9,       NULL,   gcc_dwarf_r9_mips,    gcc_dwarf_r9_mips,    LLDB_INVALID_REGNUM,    gdb_r9_mips),
    DEFINE_GPR (r10,      NULL,   gcc_dwarf_r10_mips,   gcc_dwarf_r10_mips,   LLDB_INVALID_REGNUM,    gdb_r10_mips),
    DEFINE_GPR (r11,      NULL,   gcc_dwarf_r11_mips,   gcc_dwarf_r11_mips,   LLDB_INVALID_REGNUM,    gdb_r11_mips),
    DEFINE_GPR (r12,      NULL,   gcc_dwarf_r12_mips,   gcc_dwarf_r12_mips,   LLDB_INVALID_REGNUM,    gdb_r12_mips),
    DEFINE_GPR (r13,      NULL,   gcc_dwarf_r13_mips,   gcc_dwarf_r13_mips,   LLDB_INVALID_REGNUM,    gdb_r13_mips),
    DEFINE_GPR (r14,      NULL,   gcc_dwarf_r14_mips,   gcc_dwarf_r14_mips,   LLDB_INVALID_REGNUM,    gdb_r14_mips),
    DEFINE_GPR (r15,      NULL,   gcc_dwarf_r15_mips,   gcc_dwarf_r15_mips,   LLDB_INVALID_REGNUM,    gdb_r15_mips),
    DEFINE_GPR (r16,      NULL,   gcc_dwarf_r16_mips,   gcc_dwarf_r16_mips,   LLDB_INVALID_REGNUM,    gdb_r16_mips),
    DEFINE_GPR (r17,      NULL,   gcc_dwarf_r17_mips,   gcc_dwarf_r17_mips,   LLDB_INVALID_REGNUM,    gdb_r17_mips),
    DEFINE_GPR (r18,      NULL,   gcc_dwarf_r18_mips,   gcc_dwarf_r18_mips,   LLDB_INVALID_REGNUM,    gdb_r18_mips),
    DEFINE_GPR (r19,      NULL,   gcc_dwarf_r19_mips,   gcc_dwarf_r19_mips,   LLDB_INVALID_REGNUM,    gdb_r19_mips),
    DEFINE_GPR (r20,      NULL,   gcc_dwarf_r20_mips,   gcc_dwarf_r20_mips,   LLDB_INVALID_REGNUM,    gdb_r20_mips),
    DEFINE_GPR (r21,      NULL,   gcc_dwarf_r21_mips,   gcc_dwarf_r21_mips,   LLDB_INVALID_REGNUM,    gdb_r21_mips),
    DEFINE_GPR (r22,      NULL,   gcc_dwarf_r22_mips,   gcc_dwarf_r22_mips,   LLDB_INVALID_REGNUM,    gdb_r22_mips),
    DEFINE_GPR (r23,      NULL,   gcc_dwarf_r23_mips,   gcc_dwarf_r23_mips,   LLDB_INVALID_REGNUM,    gdb_r23_mips),
    DEFINE_GPR (r24,      NULL,   gcc_dwarf_r24_mips,   gcc_dwarf_r24_mips,   LLDB_INVALID_REGNUM,    gdb_r24_mips),
    DEFINE_GPR (r25,      NULL,   gcc_dwarf_r25_mips,   gcc_dwarf_r25_mips,   LLDB_INVALID_REGNUM,    gdb_r25_mips),
    DEFINE_GPR (r26,      NULL,   gcc_dwarf_r26_mips,   gcc_dwarf_r26_mips,   LLDB_INVALID_REGNUM,    gdb_r26_mips),
    DEFINE_GPR (r27,      NULL,   gcc_dwarf_r27_mips,   gcc_dwarf_r27_mips,   LLDB_INVALID_REGNUM,    gdb_r27_mips),
    DEFINE_GPR (gp,       "gp",   gcc_dwarf_gp_mips,    gcc_dwarf_gp_mips,    LLDB_INVALID_REGNUM,    gdb_gp_mips),
    DEFINE_GPR (sp,       "sp",   gcc_dwarf_sp_mips,    gcc_dwarf_sp_mips,    LLDB_REGNUM_GENERIC_SP, gdb_sp_mips),
    DEFINE_GPR (r30,      "fp",   gcc_dwarf_r30_mips,   gcc_dwarf_r30_mips,   LLDB_REGNUM_GENERIC_FP, gdb_r30_mips),
    DEFINE_GPR (ra,       "ra",   gcc_dwarf_ra_mips,    gcc_dwarf_ra_mips,    LLDB_REGNUM_GENERIC_RA, gdb_ra_mips),
    DEFINE_GPR (sr,   "status",   gcc_dwarf_sr_mips,    gcc_dwarf_sr_mips,    LLDB_REGNUM_GENERIC_FLAGS,    LLDB_INVALID_REGNUM),
    DEFINE_GPR (mullo,    NULL,   gcc_dwarf_lo_mips,    gcc_dwarf_lo_mips,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR (mulhi,    NULL,   gcc_dwarf_hi_mips,    gcc_dwarf_hi_mips,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR (badvaddr, NULL,   gcc_dwarf_bad_mips,    gcc_dwarf_bad_mips,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR (cause,    NULL,   gcc_dwarf_cause_mips,    gcc_dwarf_cause_mips,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR (pc,       NULL,   gcc_dwarf_pc_mips,    gcc_dwarf_pc_mips,    LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_REGNUM),
    DEFINE_GPR (config5,    NULL,   gcc_dwarf_config5_mips,    gcc_dwarf_config5_mips,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_FPR (f0,    NULL,   gcc_dwarf_f0_mips,   gcc_dwarf_f0_mips,   LLDB_INVALID_REGNUM,    gdb_f0_mips),
    DEFINE_FPR (f1,    NULL,   gcc_dwarf_f1_mips,   gcc_dwarf_f1_mips,   LLDB_INVALID_REGNUM,    gdb_f1_mips),
    DEFINE_FPR (f2,    NULL,   gcc_dwarf_f2_mips,   gcc_dwarf_f2_mips,   LLDB_INVALID_REGNUM,    gdb_f2_mips),
    DEFINE_FPR (f3,    NULL,   gcc_dwarf_f3_mips,   gcc_dwarf_f3_mips,   LLDB_INVALID_REGNUM,    gdb_f3_mips),
    DEFINE_FPR (f4,    NULL,   gcc_dwarf_f4_mips,   gcc_dwarf_f4_mips,   LLDB_INVALID_REGNUM,    gdb_f4_mips),
    DEFINE_FPR (f5,    NULL,   gcc_dwarf_f5_mips,   gcc_dwarf_f5_mips,   LLDB_INVALID_REGNUM,    gdb_f5_mips),
    DEFINE_FPR (f6,    NULL,   gcc_dwarf_f6_mips,   gcc_dwarf_f6_mips,   LLDB_INVALID_REGNUM,    gdb_f6_mips),
    DEFINE_FPR (f7,    NULL,   gcc_dwarf_f7_mips,   gcc_dwarf_f7_mips,   LLDB_INVALID_REGNUM,    gdb_f7_mips),
    DEFINE_FPR (f8,    NULL,   gcc_dwarf_f8_mips,   gcc_dwarf_f8_mips,   LLDB_INVALID_REGNUM,    gdb_f8_mips),
    DEFINE_FPR (f9,    NULL,   gcc_dwarf_f9_mips,   gcc_dwarf_f9_mips,   LLDB_INVALID_REGNUM,    gdb_f9_mips),
    DEFINE_FPR (f10,   NULL,   gcc_dwarf_f10_mips,  gcc_dwarf_f10_mips,  LLDB_INVALID_REGNUM,    gdb_f10_mips),
    DEFINE_FPR (f11,   NULL,   gcc_dwarf_f11_mips,  gcc_dwarf_f11_mips,  LLDB_INVALID_REGNUM,    gdb_f11_mips),
    DEFINE_FPR (f12,   NULL,   gcc_dwarf_f12_mips,  gcc_dwarf_f12_mips,  LLDB_INVALID_REGNUM,    gdb_f12_mips),
    DEFINE_FPR (f13,   NULL,   gcc_dwarf_f13_mips,  gcc_dwarf_f13_mips,  LLDB_INVALID_REGNUM,    gdb_f13_mips),
    DEFINE_FPR (f14,   NULL,   gcc_dwarf_f14_mips,  gcc_dwarf_f14_mips,  LLDB_INVALID_REGNUM,    gdb_f14_mips),
    DEFINE_FPR (f15,   NULL,   gcc_dwarf_f15_mips,  gcc_dwarf_f15_mips,  LLDB_INVALID_REGNUM,    gdb_f15_mips),
    DEFINE_FPR (f16,   NULL,   gcc_dwarf_f16_mips,  gcc_dwarf_f16_mips,  LLDB_INVALID_REGNUM,    gdb_f16_mips),
    DEFINE_FPR (f17,   NULL,   gcc_dwarf_f17_mips,  gcc_dwarf_f17_mips,  LLDB_INVALID_REGNUM,    gdb_f17_mips),
    DEFINE_FPR (f18,   NULL,   gcc_dwarf_f18_mips,  gcc_dwarf_f18_mips,  LLDB_INVALID_REGNUM,    gdb_f18_mips),
    DEFINE_FPR (f19,   NULL,   gcc_dwarf_f19_mips,  gcc_dwarf_f19_mips,  LLDB_INVALID_REGNUM,    gdb_f19_mips),
    DEFINE_FPR (f20,   NULL,   gcc_dwarf_f20_mips,  gcc_dwarf_f20_mips,  LLDB_INVALID_REGNUM,    gdb_f20_mips),
    DEFINE_FPR (f21,   NULL,   gcc_dwarf_f21_mips,  gcc_dwarf_f21_mips,  LLDB_INVALID_REGNUM,    gdb_f21_mips),
    DEFINE_FPR (f22,   NULL,   gcc_dwarf_f22_mips,  gcc_dwarf_f22_mips,  LLDB_INVALID_REGNUM,    gdb_f22_mips),
    DEFINE_FPR (f23,   NULL,   gcc_dwarf_f23_mips,  gcc_dwarf_f23_mips,  LLDB_INVALID_REGNUM,    gdb_f23_mips),
    DEFINE_FPR (f24,   NULL,   gcc_dwarf_f24_mips,  gcc_dwarf_f24_mips,  LLDB_INVALID_REGNUM,    gdb_f24_mips),
    DEFINE_FPR (f25,   NULL,   gcc_dwarf_f25_mips,  gcc_dwarf_f25_mips,  LLDB_INVALID_REGNUM,    gdb_f25_mips),
    DEFINE_FPR (f26,   NULL,   gcc_dwarf_f26_mips,  gcc_dwarf_f26_mips,  LLDB_INVALID_REGNUM,    gdb_f26_mips),
    DEFINE_FPR (f27,   NULL,   gcc_dwarf_f27_mips,  gcc_dwarf_f27_mips,  LLDB_INVALID_REGNUM,    gdb_f27_mips),
    DEFINE_FPR (f28,   NULL,   gcc_dwarf_f28_mips,  gcc_dwarf_f28_mips,  LLDB_INVALID_REGNUM,    gdb_f28_mips),
    DEFINE_FPR (f29,   NULL,   gcc_dwarf_f29_mips,  gcc_dwarf_f29_mips,  LLDB_INVALID_REGNUM,    gdb_f29_mips),
    DEFINE_FPR (f30,   NULL,   gcc_dwarf_f30_mips,  gcc_dwarf_f30_mips,  LLDB_INVALID_REGNUM,    gdb_f30_mips),
    DEFINE_FPR (f31,   NULL,   gcc_dwarf_f31_mips,  gcc_dwarf_f31_mips,  LLDB_INVALID_REGNUM,    gdb_f31_mips),
    DEFINE_FPR (fcsr,  NULL,   gcc_dwarf_fcsr_mips, gcc_dwarf_fcsr_mips, LLDB_INVALID_REGNUM,    gdb_fcsr_mips),
    DEFINE_FPR (fir,   NULL,   gcc_dwarf_fir_mips,  gcc_dwarf_fir_mips,  LLDB_INVALID_REGNUM,    gdb_fir_mips),
    DEFINE_FPR (config5,   NULL,   gcc_dwarf_config5_mips,  gcc_dwarf_config5_mips,  LLDB_INVALID_REGNUM,    gdb_config5_mips),
    DEFINE_MSA (w0,    NULL,   gcc_dwarf_w0_mips,   gcc_dwarf_w0_mips,   LLDB_INVALID_REGNUM,    gdb_w0_mips),
    DEFINE_MSA (w1,    NULL,   gcc_dwarf_w1_mips,   gcc_dwarf_w1_mips,   LLDB_INVALID_REGNUM,    gdb_w1_mips),
    DEFINE_MSA (w2,    NULL,   gcc_dwarf_w2_mips,   gcc_dwarf_w2_mips,   LLDB_INVALID_REGNUM,    gdb_w2_mips),
    DEFINE_MSA (w3,    NULL,   gcc_dwarf_w3_mips,   gcc_dwarf_w3_mips,   LLDB_INVALID_REGNUM,    gdb_w3_mips),
    DEFINE_MSA (w4,    NULL,   gcc_dwarf_w4_mips,   gcc_dwarf_w4_mips,   LLDB_INVALID_REGNUM,    gdb_w4_mips),
    DEFINE_MSA (w5,    NULL,   gcc_dwarf_w5_mips,   gcc_dwarf_w5_mips,   LLDB_INVALID_REGNUM,    gdb_w5_mips),
    DEFINE_MSA (w6,    NULL,   gcc_dwarf_w6_mips,   gcc_dwarf_w6_mips,   LLDB_INVALID_REGNUM,    gdb_w6_mips),
    DEFINE_MSA (w7,    NULL,   gcc_dwarf_w7_mips,   gcc_dwarf_w7_mips,   LLDB_INVALID_REGNUM,    gdb_w7_mips),
    DEFINE_MSA (w8,    NULL,   gcc_dwarf_w8_mips,   gcc_dwarf_w8_mips,   LLDB_INVALID_REGNUM,    gdb_w8_mips),
    DEFINE_MSA (w9,    NULL,   gcc_dwarf_w9_mips,   gcc_dwarf_w9_mips,   LLDB_INVALID_REGNUM,    gdb_w9_mips),
    DEFINE_MSA (w10,   NULL,   gcc_dwarf_w10_mips,  gcc_dwarf_w10_mips,  LLDB_INVALID_REGNUM,    gdb_w10_mips),
    DEFINE_MSA (w11,   NULL,   gcc_dwarf_w11_mips,  gcc_dwarf_w11_mips,  LLDB_INVALID_REGNUM,    gdb_w11_mips),
    DEFINE_MSA (w12,   NULL,   gcc_dwarf_w12_mips,  gcc_dwarf_w12_mips,  LLDB_INVALID_REGNUM,    gdb_w12_mips),
    DEFINE_MSA (w13,   NULL,   gcc_dwarf_w13_mips,  gcc_dwarf_w13_mips,  LLDB_INVALID_REGNUM,    gdb_w13_mips),
    DEFINE_MSA (w14,   NULL,   gcc_dwarf_w14_mips,  gcc_dwarf_w14_mips,  LLDB_INVALID_REGNUM,    gdb_w14_mips),
    DEFINE_MSA (w15,   NULL,   gcc_dwarf_w15_mips,  gcc_dwarf_w15_mips,  LLDB_INVALID_REGNUM,    gdb_w15_mips),
    DEFINE_MSA (w16,   NULL,   gcc_dwarf_w16_mips,  gcc_dwarf_w16_mips,  LLDB_INVALID_REGNUM,    gdb_w16_mips),
    DEFINE_MSA (w17,   NULL,   gcc_dwarf_w17_mips,  gcc_dwarf_w17_mips,  LLDB_INVALID_REGNUM,    gdb_w17_mips),
    DEFINE_MSA (w18,   NULL,   gcc_dwarf_w18_mips,  gcc_dwarf_w18_mips,  LLDB_INVALID_REGNUM,    gdb_w18_mips),
    DEFINE_MSA (w19,   NULL,   gcc_dwarf_w19_mips,  gcc_dwarf_w19_mips,  LLDB_INVALID_REGNUM,    gdb_w19_mips),
    DEFINE_MSA (w20,   NULL,   gcc_dwarf_w10_mips,  gcc_dwarf_w20_mips,  LLDB_INVALID_REGNUM,    gdb_w20_mips),
    DEFINE_MSA (w21,   NULL,   gcc_dwarf_w21_mips,  gcc_dwarf_w21_mips,  LLDB_INVALID_REGNUM,    gdb_w21_mips),
    DEFINE_MSA (w22,   NULL,   gcc_dwarf_w22_mips,  gcc_dwarf_w22_mips,  LLDB_INVALID_REGNUM,    gdb_w22_mips),
    DEFINE_MSA (w23,   NULL,   gcc_dwarf_w23_mips,  gcc_dwarf_w23_mips,  LLDB_INVALID_REGNUM,    gdb_w23_mips),
    DEFINE_MSA (w24,   NULL,   gcc_dwarf_w24_mips,  gcc_dwarf_w24_mips,  LLDB_INVALID_REGNUM,    gdb_w24_mips),
    DEFINE_MSA (w25,   NULL,   gcc_dwarf_w25_mips,  gcc_dwarf_w25_mips,  LLDB_INVALID_REGNUM,    gdb_w25_mips),
    DEFINE_MSA (w26,   NULL,   gcc_dwarf_w26_mips,  gcc_dwarf_w26_mips,  LLDB_INVALID_REGNUM,    gdb_w26_mips),
    DEFINE_MSA (w27,   NULL,   gcc_dwarf_w27_mips,  gcc_dwarf_w27_mips,  LLDB_INVALID_REGNUM,    gdb_w27_mips),
    DEFINE_MSA (w28,   NULL,   gcc_dwarf_w28_mips,  gcc_dwarf_w28_mips,  LLDB_INVALID_REGNUM,    gdb_w28_mips),
    DEFINE_MSA (w29,   NULL,   gcc_dwarf_w29_mips,  gcc_dwarf_w29_mips,  LLDB_INVALID_REGNUM,    gdb_w29_mips),
    DEFINE_MSA (w30,   NULL,   gcc_dwarf_w30_mips,  gcc_dwarf_w30_mips,  LLDB_INVALID_REGNUM,    gdb_w30_mips),
    DEFINE_MSA (w31,   NULL,   gcc_dwarf_w31_mips,  gcc_dwarf_w31_mips,  LLDB_INVALID_REGNUM,    gdb_w31_mips),
    DEFINE_MSA_INFO (mcsr,  NULL,   gcc_dwarf_mcsr_mips, gcc_dwarf_mcsr_mips, LLDB_INVALID_REGNUM,    gdb_mcsr_mips),
    DEFINE_MSA_INFO (mir,   NULL,   gcc_dwarf_mir_mips,  gcc_dwarf_mir_mips,  LLDB_INVALID_REGNUM,    gdb_mir_mips),
    DEFINE_MSA_INFO (fcsr,  NULL,   gcc_dwarf_fcsr_mips, gcc_dwarf_fcsr_mips, LLDB_INVALID_REGNUM,    gdb_fcsr_mips),
    DEFINE_MSA_INFO (fir,   NULL,   gcc_dwarf_fir_mips,  gcc_dwarf_fir_mips,  LLDB_INVALID_REGNUM,    gdb_fir_mips),
    DEFINE_MSA_INFO (config5, NULL,   gcc_dwarf_config5_mips,  gcc_dwarf_config5_mips,  LLDB_INVALID_REGNUM,    gdb_config5_mips)
};
static_assert((sizeof(g_register_infos_mips) / sizeof(g_register_infos_mips[0])) == k_num_registers_mips,
    "g_register_infos_mips has wrong number of register infos");

#undef GPR_OFFSET
#undef FPR_OFFSET
#undef MSA_OFFSET
#undef DEFINE_GPR
#undef DEFINE_FPR
#undef DEFINE_MSA
#undef DEFINE_MSA_INFO

#endif // DECLARE_REGISTER_INFOS_MIPS_STRUCT
