//===-- RegisterInfos_mips64.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
#include "llvm/Support/Compiler.h"

#include <stddef.h>

#ifdef DECLARE_REGISTER_INFOS_MIPS64_STRUCT

// Computes the offset of the given GPR in the user data area.
#ifdef LINUX_MIPS64
    #define GPR_OFFSET(regname) \
        (LLVM_EXTENSION offsetof(UserArea, gpr) + \
         LLVM_EXTENSION offsetof(GPR_linux_mips, regname))
#else
    #define GPR_OFFSET(regname) \
        (LLVM_EXTENSION offsetof(GPR_freebsd_mips, regname))
#endif

// Computes the offset of the given FPR in the extended data area.
#define FPR_OFFSET(regname) \
     (LLVM_EXTENSION offsetof(UserArea, fpr) + \
      LLVM_EXTENSION offsetof(FPR_linux_mips, regname))

// Computes the offset of the given MSA in the extended data area.
#define MSA_OFFSET(regname) \
     (LLVM_EXTENSION offsetof(UserArea, msa) + \
      LLVM_EXTENSION offsetof(MSA_linux_mips, regname))

// RegisterKind: GCC, DWARF, Generic, GDB, LLDB

// Note that the size and offset will be updated by platform-specific classes.
#ifdef LINUX_MIPS64
    #define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4) \
         { #reg, alt, sizeof(((GPR_linux_mips*)0)->reg), GPR_OFFSET(reg), eEncodingUint, \
          eFormatHex, { kind1, kind2, kind3, kind4, gpr_##reg##_mips64 }, NULL, NULL }
#else
    #define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)    \
         { #reg, alt, sizeof(((GPR_freebsd_mips*)0)->reg), GPR_OFFSET(reg), eEncodingUint, \
          eFormatHex, { kind1, kind2, kind3, kind4, gpr_##reg##_mips64 }, NULL, NULL }
#endif

#define DEFINE_GPR_INFO(reg, alt, kind1, kind2, kind3, kind4)    \
    { #reg, alt, sizeof(((GPR_linux_mips*)0)->reg) / 2, GPR_OFFSET(reg), eEncodingUint, \
      eFormatHex, { kind1, kind2, kind3, kind4, gpr_##reg##_mips64 }, NULL, NULL }

#define DEFINE_FPR(reg, alt, kind1, kind2, kind3, kind4)    \
    { #reg, alt, sizeof(((FPR_linux_mips*)0)->reg), FPR_OFFSET(reg), eEncodingUint,   \
      eFormatHex, { kind1, kind2, kind3, kind4, fpr_##reg##_mips64 }, NULL, NULL }

#define DEFINE_MSA(reg, alt, kind1, kind2, kind3, kind4)    \
    { #reg, alt, sizeof(((MSA_linux_mips*)0)->reg), MSA_OFFSET(reg), eEncodingVector,   \
      eFormatVectorOfUInt8, { kind1, kind2, kind3, kind4, msa_##reg##_mips64 }, NULL, NULL }

#define DEFINE_MSA_INFO(reg, alt, kind1, kind2, kind3, kind4)    \
    { #reg, alt, sizeof(((MSA_linux_mips*)0)->reg), MSA_OFFSET(reg), eEncodingUint,   \
      eFormatHex, { kind1, kind2, kind3, kind4, msa_##reg##_mips64 }, NULL, NULL }

static RegisterInfo
g_register_infos_mips64[] =
{
    // General purpose registers.                 GCC,                  DWARF,              Generic,                GDB
#ifndef LINUX_MIPS64
    DEFINE_GPR(zero,     "r0",  gcc_dwarf_zero_mips64,  gcc_dwarf_zero_mips64,  LLDB_INVALID_REGNUM,    gdb_zero_mips64),
    DEFINE_GPR(r1,       NULL,  gcc_dwarf_r1_mips64,    gcc_dwarf_r1_mips64,    LLDB_INVALID_REGNUM,    gdb_r1_mips64),
    DEFINE_GPR(r2,       NULL,  gcc_dwarf_r2_mips64,    gcc_dwarf_r2_mips64,    LLDB_INVALID_REGNUM,    gdb_r2_mips64),
    DEFINE_GPR(r3,       NULL,  gcc_dwarf_r3_mips64,    gcc_dwarf_r3_mips64,    LLDB_INVALID_REGNUM,    gdb_r3_mips64),
    DEFINE_GPR(r4,       NULL,  gcc_dwarf_r4_mips64,    gcc_dwarf_r4_mips64,    LLDB_REGNUM_GENERIC_ARG1,    gdb_r4_mips64),
    DEFINE_GPR(r5,       NULL,  gcc_dwarf_r5_mips64,    gcc_dwarf_r5_mips64,    LLDB_REGNUM_GENERIC_ARG2,    gdb_r5_mips64),
    DEFINE_GPR(r6,       NULL,  gcc_dwarf_r6_mips64,    gcc_dwarf_r6_mips64,    LLDB_REGNUM_GENERIC_ARG3,    gdb_r6_mips64),
    DEFINE_GPR(r7,       NULL,  gcc_dwarf_r7_mips64,    gcc_dwarf_r7_mips64,    LLDB_REGNUM_GENERIC_ARG4,    gdb_r7_mips64),
    DEFINE_GPR(r8,       NULL,  gcc_dwarf_r8_mips64,    gcc_dwarf_r8_mips64,    LLDB_REGNUM_GENERIC_ARG5,    gdb_r8_mips64),
    DEFINE_GPR(r9,       NULL,  gcc_dwarf_r9_mips64,    gcc_dwarf_r9_mips64,    LLDB_REGNUM_GENERIC_ARG6,    gdb_r9_mips64),
    DEFINE_GPR(r10,      NULL,  gcc_dwarf_r10_mips64,   gcc_dwarf_r10_mips64,   LLDB_REGNUM_GENERIC_ARG7,    gdb_r10_mips64),
    DEFINE_GPR(r11,      NULL,  gcc_dwarf_r11_mips64,   gcc_dwarf_r11_mips64,   LLDB_REGNUM_GENERIC_ARG8,    gdb_r11_mips64),
    DEFINE_GPR(r12,      NULL,  gcc_dwarf_r12_mips64,   gcc_dwarf_r12_mips64,   LLDB_INVALID_REGNUM,    gdb_r12_mips64),
    DEFINE_GPR(r13,      NULL,  gcc_dwarf_r13_mips64,   gcc_dwarf_r13_mips64,   LLDB_INVALID_REGNUM,    gdb_r13_mips64),
    DEFINE_GPR(r14,      NULL,  gcc_dwarf_r14_mips64,   gcc_dwarf_r14_mips64,   LLDB_INVALID_REGNUM,    gdb_r14_mips64),
    DEFINE_GPR(r15,      NULL,  gcc_dwarf_r15_mips64,   gcc_dwarf_r15_mips64,   LLDB_INVALID_REGNUM,    gdb_r15_mips64),
    DEFINE_GPR(r16,      NULL,  gcc_dwarf_r16_mips64,   gcc_dwarf_r16_mips64,   LLDB_INVALID_REGNUM,    gdb_r16_mips64),
    DEFINE_GPR(r17,      NULL,  gcc_dwarf_r17_mips64,   gcc_dwarf_r17_mips64,   LLDB_INVALID_REGNUM,    gdb_r17_mips64),
    DEFINE_GPR(r18,      NULL,  gcc_dwarf_r18_mips64,   gcc_dwarf_r18_mips64,   LLDB_INVALID_REGNUM,    gdb_r18_mips64),
    DEFINE_GPR(r19,      NULL,  gcc_dwarf_r19_mips64,   gcc_dwarf_r19_mips64,   LLDB_INVALID_REGNUM,    gdb_r19_mips64),
    DEFINE_GPR(r20,      NULL,  gcc_dwarf_r20_mips64,   gcc_dwarf_r20_mips64,   LLDB_INVALID_REGNUM,    gdb_r20_mips64),
    DEFINE_GPR(r21,      NULL,  gcc_dwarf_r21_mips64,   gcc_dwarf_r21_mips64,   LLDB_INVALID_REGNUM,    gdb_r21_mips64),
    DEFINE_GPR(r22,      NULL,  gcc_dwarf_r22_mips64,   gcc_dwarf_r22_mips64,   LLDB_INVALID_REGNUM,    gdb_r22_mips64),
    DEFINE_GPR(r23,      NULL,  gcc_dwarf_r23_mips64,   gcc_dwarf_r23_mips64,   LLDB_INVALID_REGNUM,    gdb_r23_mips64),
    DEFINE_GPR(r24,      NULL,  gcc_dwarf_r24_mips64,   gcc_dwarf_r24_mips64,   LLDB_INVALID_REGNUM,    gdb_r24_mips64),
    DEFINE_GPR(r25,      NULL,  gcc_dwarf_r25_mips64,   gcc_dwarf_r25_mips64,   LLDB_INVALID_REGNUM,    gdb_r25_mips64),
    DEFINE_GPR(r26,      NULL,  gcc_dwarf_r26_mips64,   gcc_dwarf_r26_mips64,   LLDB_INVALID_REGNUM,    gdb_r26_mips64),
    DEFINE_GPR(r27,      NULL,  gcc_dwarf_r27_mips64,   gcc_dwarf_r27_mips64,   LLDB_INVALID_REGNUM,    gdb_r27_mips64),
    DEFINE_GPR(gp,       "r28", gcc_dwarf_gp_mips64,    gcc_dwarf_gp_mips64,    LLDB_INVALID_REGNUM,    gdb_gp_mips64),
    DEFINE_GPR(sp,       "r29", gcc_dwarf_sp_mips64,    gcc_dwarf_sp_mips64,    LLDB_REGNUM_GENERIC_SP, gdb_sp_mips64),
    DEFINE_GPR(r30,      NULL,  gcc_dwarf_r30_mips64,   gcc_dwarf_r30_mips64,   LLDB_REGNUM_GENERIC_FP,    gdb_r30_mips64),
    DEFINE_GPR(ra,       "r31", gcc_dwarf_ra_mips64,    gcc_dwarf_ra_mips64,    LLDB_REGNUM_GENERIC_RA,    gdb_ra_mips64),
    DEFINE_GPR(sr,       NULL,  gcc_dwarf_sr_mips64,    gcc_dwarf_sr_mips64,    LLDB_REGNUM_GENERIC_FLAGS,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(mullo,    NULL,  gcc_dwarf_lo_mips64,    gcc_dwarf_lo_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(mulhi,    NULL,  gcc_dwarf_hi_mips64,    gcc_dwarf_hi_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(badvaddr, NULL,  gcc_dwarf_bad_mips64,   gcc_dwarf_bad_mips64,   LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(cause,    NULL,  gcc_dwarf_cause_mips64, gcc_dwarf_cause_mips64, LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(pc,       "pc",  gcc_dwarf_pc_mips64,    gcc_dwarf_pc_mips64,    LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_REGNUM),
    DEFINE_GPR(ic,       NULL,  gcc_dwarf_ic_mips64,    gcc_dwarf_ic_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(dummy,    NULL,  gcc_dwarf_dummy_mips64, gcc_dwarf_dummy_mips64, LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
#else

    DEFINE_GPR(zero,     "r0",  gcc_dwarf_zero_mips64,  gcc_dwarf_zero_mips64,  LLDB_INVALID_REGNUM,    gdb_zero_mips64),
    DEFINE_GPR(r1,       NULL,  gcc_dwarf_r1_mips64,    gcc_dwarf_r1_mips64,    LLDB_INVALID_REGNUM,    gdb_r1_mips64),
    DEFINE_GPR(r2,       NULL,  gcc_dwarf_r2_mips64,    gcc_dwarf_r2_mips64,    LLDB_INVALID_REGNUM,    gdb_r2_mips64),
    DEFINE_GPR(r3,       NULL,  gcc_dwarf_r3_mips64,    gcc_dwarf_r3_mips64,    LLDB_INVALID_REGNUM,    gdb_r3_mips64),
    DEFINE_GPR(r4,       NULL,  gcc_dwarf_r4_mips64,    gcc_dwarf_r4_mips64,    LLDB_REGNUM_GENERIC_ARG1,    gdb_r4_mips64),
    DEFINE_GPR(r5,       NULL,  gcc_dwarf_r5_mips64,    gcc_dwarf_r5_mips64,    LLDB_REGNUM_GENERIC_ARG2,    gdb_r5_mips64),
    DEFINE_GPR(r6,       NULL,  gcc_dwarf_r6_mips64,    gcc_dwarf_r6_mips64,    LLDB_REGNUM_GENERIC_ARG3,    gdb_r6_mips64),
    DEFINE_GPR(r7,       NULL,  gcc_dwarf_r7_mips64,    gcc_dwarf_r7_mips64,    LLDB_REGNUM_GENERIC_ARG4,    gdb_r7_mips64),
    DEFINE_GPR(r8,       NULL,  gcc_dwarf_r8_mips64,    gcc_dwarf_r8_mips64,    LLDB_REGNUM_GENERIC_ARG5,    gdb_r8_mips64),
    DEFINE_GPR(r9,       NULL,  gcc_dwarf_r9_mips64,    gcc_dwarf_r9_mips64,    LLDB_REGNUM_GENERIC_ARG6,    gdb_r9_mips64),
    DEFINE_GPR(r10,      NULL,  gcc_dwarf_r10_mips64,   gcc_dwarf_r10_mips64,   LLDB_REGNUM_GENERIC_ARG7,    gdb_r10_mips64),
    DEFINE_GPR(r11,      NULL,  gcc_dwarf_r11_mips64,   gcc_dwarf_r11_mips64,   LLDB_REGNUM_GENERIC_ARG8,    gdb_r11_mips64),
    DEFINE_GPR(r12,      NULL,  gcc_dwarf_r12_mips64,   gcc_dwarf_r12_mips64,   LLDB_INVALID_REGNUM,    gdb_r12_mips64),
    DEFINE_GPR(r13,      NULL,  gcc_dwarf_r13_mips64,   gcc_dwarf_r13_mips64,   LLDB_INVALID_REGNUM,    gdb_r13_mips64),
    DEFINE_GPR(r14,      NULL,  gcc_dwarf_r14_mips64,   gcc_dwarf_r14_mips64,   LLDB_INVALID_REGNUM,    gdb_r14_mips64),
    DEFINE_GPR(r15,      NULL,  gcc_dwarf_r15_mips64,   gcc_dwarf_r15_mips64,   LLDB_INVALID_REGNUM,    gdb_r15_mips64),
    DEFINE_GPR(r16,      NULL,  gcc_dwarf_r16_mips64,   gcc_dwarf_r16_mips64,   LLDB_INVALID_REGNUM,    gdb_r16_mips64),
    DEFINE_GPR(r17,      NULL,  gcc_dwarf_r17_mips64,   gcc_dwarf_r17_mips64,   LLDB_INVALID_REGNUM,    gdb_r17_mips64),
    DEFINE_GPR(r18,      NULL,  gcc_dwarf_r18_mips64,   gcc_dwarf_r18_mips64,   LLDB_INVALID_REGNUM,    gdb_r18_mips64),
    DEFINE_GPR(r19,      NULL,  gcc_dwarf_r19_mips64,   gcc_dwarf_r19_mips64,   LLDB_INVALID_REGNUM,    gdb_r19_mips64),
    DEFINE_GPR(r20,      NULL,  gcc_dwarf_r20_mips64,   gcc_dwarf_r20_mips64,   LLDB_INVALID_REGNUM,    gdb_r20_mips64),
    DEFINE_GPR(r21,      NULL,  gcc_dwarf_r21_mips64,   gcc_dwarf_r21_mips64,   LLDB_INVALID_REGNUM,    gdb_r21_mips64),
    DEFINE_GPR(r22,      NULL,  gcc_dwarf_r22_mips64,   gcc_dwarf_r22_mips64,   LLDB_INVALID_REGNUM,    gdb_r22_mips64),
    DEFINE_GPR(r23,      NULL,  gcc_dwarf_r23_mips64,   gcc_dwarf_r23_mips64,   LLDB_INVALID_REGNUM,    gdb_r23_mips64),
    DEFINE_GPR(r24,      NULL,  gcc_dwarf_r24_mips64,   gcc_dwarf_r24_mips64,   LLDB_INVALID_REGNUM,    gdb_r24_mips64),
    DEFINE_GPR(r25,      NULL,  gcc_dwarf_r25_mips64,   gcc_dwarf_r25_mips64,   LLDB_INVALID_REGNUM,    gdb_r25_mips64),
    DEFINE_GPR(r26,      NULL,  gcc_dwarf_r26_mips64,   gcc_dwarf_r26_mips64,   LLDB_INVALID_REGNUM,    gdb_r26_mips64),
    DEFINE_GPR(r27,      NULL,  gcc_dwarf_r27_mips64,   gcc_dwarf_r27_mips64,   LLDB_INVALID_REGNUM,    gdb_r27_mips64),
    DEFINE_GPR(gp,       "r28", gcc_dwarf_gp_mips64,    gcc_dwarf_gp_mips64,    LLDB_INVALID_REGNUM,    gdb_gp_mips64),
    DEFINE_GPR(sp,       "r29", gcc_dwarf_sp_mips64,    gcc_dwarf_sp_mips64,    LLDB_REGNUM_GENERIC_SP, gdb_sp_mips64),
    DEFINE_GPR(r30,      NULL,  gcc_dwarf_r30_mips64,   gcc_dwarf_r30_mips64,   LLDB_REGNUM_GENERIC_FP,    gdb_r30_mips64),
    DEFINE_GPR(ra,       "r31", gcc_dwarf_ra_mips64,    gcc_dwarf_ra_mips64,    LLDB_REGNUM_GENERIC_RA,    gdb_ra_mips64),
    DEFINE_GPR_INFO(sr,       NULL,  gcc_dwarf_sr_mips64,    gcc_dwarf_sr_mips64,    LLDB_REGNUM_GENERIC_FLAGS,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(mullo,    NULL,  gcc_dwarf_lo_mips64,    gcc_dwarf_lo_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(mulhi,    NULL,  gcc_dwarf_hi_mips64,    gcc_dwarf_hi_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(badvaddr, NULL,  gcc_dwarf_bad_mips64,   gcc_dwarf_bad_mips64,   LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR_INFO(cause,    NULL,  gcc_dwarf_cause_mips64, gcc_dwarf_cause_mips64, LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(pc,       "pc",  gcc_dwarf_pc_mips64,    gcc_dwarf_pc_mips64,    LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_REGNUM),
    DEFINE_GPR_INFO(config5,    NULL,  gcc_dwarf_config5_mips64, gcc_dwarf_config5_mips64, LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_FPR (f0,    NULL,   gcc_dwarf_f0_mips64,   gcc_dwarf_f0_mips64,   LLDB_INVALID_REGNUM,    gdb_f0_mips64),
    DEFINE_FPR (f1,    NULL,   gcc_dwarf_f1_mips64,   gcc_dwarf_f1_mips64,   LLDB_INVALID_REGNUM,    gdb_f1_mips64),
    DEFINE_FPR (f2,    NULL,   gcc_dwarf_f2_mips64,   gcc_dwarf_f2_mips64,   LLDB_INVALID_REGNUM,    gdb_f2_mips64),
    DEFINE_FPR (f3,    NULL,   gcc_dwarf_f3_mips64,   gcc_dwarf_f3_mips64,   LLDB_INVALID_REGNUM,    gdb_f3_mips64),
    DEFINE_FPR (f4,    NULL,   gcc_dwarf_f4_mips64,   gcc_dwarf_f4_mips64,   LLDB_INVALID_REGNUM,    gdb_f4_mips64),
    DEFINE_FPR (f5,    NULL,   gcc_dwarf_f5_mips64,   gcc_dwarf_f5_mips64,   LLDB_INVALID_REGNUM,    gdb_f5_mips64),
    DEFINE_FPR (f6,    NULL,   gcc_dwarf_f6_mips64,   gcc_dwarf_f6_mips64,   LLDB_INVALID_REGNUM,    gdb_f6_mips64),
    DEFINE_FPR (f7,    NULL,   gcc_dwarf_f7_mips64,   gcc_dwarf_f7_mips64,   LLDB_INVALID_REGNUM,    gdb_f7_mips64),
    DEFINE_FPR (f8,    NULL,   gcc_dwarf_f8_mips64,   gcc_dwarf_f8_mips64,   LLDB_INVALID_REGNUM,    gdb_f8_mips64),
    DEFINE_FPR (f9,    NULL,   gcc_dwarf_f9_mips64,   gcc_dwarf_f9_mips64,   LLDB_INVALID_REGNUM,    gdb_f9_mips64),
    DEFINE_FPR (f10,   NULL,   gcc_dwarf_f10_mips64,  gcc_dwarf_f10_mips64,  LLDB_INVALID_REGNUM,    gdb_f10_mips64),
    DEFINE_FPR (f11,   NULL,   gcc_dwarf_f11_mips64,  gcc_dwarf_f11_mips64,  LLDB_INVALID_REGNUM,    gdb_f11_mips64),
    DEFINE_FPR (f12,   NULL,   gcc_dwarf_f12_mips64,  gcc_dwarf_f12_mips64,  LLDB_INVALID_REGNUM,    gdb_f12_mips64),
    DEFINE_FPR (f13,   NULL,   gcc_dwarf_f13_mips64,  gcc_dwarf_f13_mips64,  LLDB_INVALID_REGNUM,    gdb_f13_mips64),
    DEFINE_FPR (f14,   NULL,   gcc_dwarf_f14_mips64,  gcc_dwarf_f14_mips64,  LLDB_INVALID_REGNUM,    gdb_f14_mips64),
    DEFINE_FPR (f15,   NULL,   gcc_dwarf_f15_mips64,  gcc_dwarf_f15_mips64,  LLDB_INVALID_REGNUM,    gdb_f15_mips64),
    DEFINE_FPR (f16,   NULL,   gcc_dwarf_f16_mips64,  gcc_dwarf_f16_mips64,  LLDB_INVALID_REGNUM,    gdb_f16_mips64),
    DEFINE_FPR (f17,   NULL,   gcc_dwarf_f17_mips64,  gcc_dwarf_f17_mips64,  LLDB_INVALID_REGNUM,    gdb_f17_mips64),
    DEFINE_FPR (f18,   NULL,   gcc_dwarf_f18_mips64,  gcc_dwarf_f18_mips64,  LLDB_INVALID_REGNUM,    gdb_f18_mips64),
    DEFINE_FPR (f19,   NULL,   gcc_dwarf_f19_mips64,  gcc_dwarf_f19_mips64,  LLDB_INVALID_REGNUM,    gdb_f19_mips64),
    DEFINE_FPR (f20,   NULL,   gcc_dwarf_f20_mips64,  gcc_dwarf_f20_mips64,  LLDB_INVALID_REGNUM,    gdb_f20_mips64),
    DEFINE_FPR (f21,   NULL,   gcc_dwarf_f21_mips64,  gcc_dwarf_f21_mips64,  LLDB_INVALID_REGNUM,    gdb_f21_mips64),
    DEFINE_FPR (f22,   NULL,   gcc_dwarf_f22_mips64,  gcc_dwarf_f22_mips64,  LLDB_INVALID_REGNUM,    gdb_f22_mips64),
    DEFINE_FPR (f23,   NULL,   gcc_dwarf_f23_mips64,  gcc_dwarf_f23_mips64,  LLDB_INVALID_REGNUM,    gdb_f23_mips64),
    DEFINE_FPR (f24,   NULL,   gcc_dwarf_f24_mips64,  gcc_dwarf_f24_mips64,  LLDB_INVALID_REGNUM,    gdb_f24_mips64),
    DEFINE_FPR (f25,   NULL,   gcc_dwarf_f25_mips64,  gcc_dwarf_f25_mips64,  LLDB_INVALID_REGNUM,    gdb_f25_mips64),
    DEFINE_FPR (f26,   NULL,   gcc_dwarf_f26_mips64,  gcc_dwarf_f26_mips64,  LLDB_INVALID_REGNUM,    gdb_f26_mips64),
    DEFINE_FPR (f27,   NULL,   gcc_dwarf_f27_mips64,  gcc_dwarf_f27_mips64,  LLDB_INVALID_REGNUM,    gdb_f27_mips64),
    DEFINE_FPR (f28,   NULL,   gcc_dwarf_f28_mips64,  gcc_dwarf_f28_mips64,  LLDB_INVALID_REGNUM,    gdb_f28_mips64),
    DEFINE_FPR (f29,   NULL,   gcc_dwarf_f29_mips64,  gcc_dwarf_f29_mips64,  LLDB_INVALID_REGNUM,    gdb_f29_mips64),
    DEFINE_FPR (f30,   NULL,   gcc_dwarf_f30_mips64,  gcc_dwarf_f30_mips64,  LLDB_INVALID_REGNUM,    gdb_f30_mips64),
    DEFINE_FPR (f31,   NULL,   gcc_dwarf_f31_mips64,  gcc_dwarf_f31_mips64,  LLDB_INVALID_REGNUM,    gdb_f31_mips64),
    DEFINE_FPR (fcsr,  NULL,   gcc_dwarf_fcsr_mips64, gcc_dwarf_fcsr_mips64, LLDB_INVALID_REGNUM,    gdb_fcsr_mips64),
    DEFINE_FPR (fir,   NULL,   gcc_dwarf_fir_mips64,  gcc_dwarf_fir_mips64,  LLDB_INVALID_REGNUM,    gdb_fir_mips64),
    DEFINE_FPR (config5,   NULL,   gcc_dwarf_config5_mips64,  gcc_dwarf_config5_mips64,  LLDB_INVALID_REGNUM,    gdb_config5_mips64),
    DEFINE_MSA (w0,    NULL,   gcc_dwarf_w0_mips64,   gcc_dwarf_w0_mips64,   LLDB_INVALID_REGNUM,    gdb_w0_mips64),
    DEFINE_MSA (w1,    NULL,   gcc_dwarf_w1_mips64,   gcc_dwarf_w1_mips64,   LLDB_INVALID_REGNUM,    gdb_w1_mips64),
    DEFINE_MSA (w2,    NULL,   gcc_dwarf_w2_mips64,   gcc_dwarf_w2_mips64,   LLDB_INVALID_REGNUM,    gdb_w2_mips64),
    DEFINE_MSA (w3,    NULL,   gcc_dwarf_w3_mips64,   gcc_dwarf_w3_mips64,   LLDB_INVALID_REGNUM,    gdb_w3_mips64),
    DEFINE_MSA (w4,    NULL,   gcc_dwarf_w4_mips64,   gcc_dwarf_w4_mips64,   LLDB_INVALID_REGNUM,    gdb_w4_mips64),
    DEFINE_MSA (w5,    NULL,   gcc_dwarf_w5_mips64,   gcc_dwarf_w5_mips64,   LLDB_INVALID_REGNUM,    gdb_w5_mips64),
    DEFINE_MSA (w6,    NULL,   gcc_dwarf_w6_mips64,   gcc_dwarf_w6_mips64,   LLDB_INVALID_REGNUM,    gdb_w6_mips64),
    DEFINE_MSA (w7,    NULL,   gcc_dwarf_w7_mips64,   gcc_dwarf_w7_mips64,   LLDB_INVALID_REGNUM,    gdb_w7_mips64),
    DEFINE_MSA (w8,    NULL,   gcc_dwarf_w8_mips64,   gcc_dwarf_w8_mips64,   LLDB_INVALID_REGNUM,    gdb_w8_mips64),
    DEFINE_MSA (w9,    NULL,   gcc_dwarf_w9_mips64,   gcc_dwarf_w9_mips64,   LLDB_INVALID_REGNUM,    gdb_w9_mips64),
    DEFINE_MSA (w10,   NULL,   gcc_dwarf_w10_mips64,  gcc_dwarf_w10_mips64,  LLDB_INVALID_REGNUM,    gdb_w10_mips64),
    DEFINE_MSA (w11,   NULL,   gcc_dwarf_w11_mips64,  gcc_dwarf_w11_mips64,  LLDB_INVALID_REGNUM,    gdb_w11_mips64),
    DEFINE_MSA (w12,   NULL,   gcc_dwarf_w12_mips64,  gcc_dwarf_w12_mips64,  LLDB_INVALID_REGNUM,    gdb_w12_mips64),
    DEFINE_MSA (w13,   NULL,   gcc_dwarf_w13_mips64,  gcc_dwarf_w13_mips64,  LLDB_INVALID_REGNUM,    gdb_w13_mips64),
    DEFINE_MSA (w14,   NULL,   gcc_dwarf_w14_mips64,  gcc_dwarf_w14_mips64,  LLDB_INVALID_REGNUM,    gdb_w14_mips64),
    DEFINE_MSA (w15,   NULL,   gcc_dwarf_w15_mips64,  gcc_dwarf_w15_mips64,  LLDB_INVALID_REGNUM,    gdb_w15_mips64),
    DEFINE_MSA (w16,   NULL,   gcc_dwarf_w16_mips64,  gcc_dwarf_w16_mips64,  LLDB_INVALID_REGNUM,    gdb_w16_mips64),
    DEFINE_MSA (w17,   NULL,   gcc_dwarf_w17_mips64,  gcc_dwarf_w17_mips64,  LLDB_INVALID_REGNUM,    gdb_w17_mips64),
    DEFINE_MSA (w18,   NULL,   gcc_dwarf_w18_mips64,  gcc_dwarf_w18_mips64,  LLDB_INVALID_REGNUM,    gdb_w18_mips64),
    DEFINE_MSA (w19,   NULL,   gcc_dwarf_w19_mips64,  gcc_dwarf_w19_mips64,  LLDB_INVALID_REGNUM,    gdb_w19_mips64),
    DEFINE_MSA (w20,   NULL,   gcc_dwarf_w10_mips64,  gcc_dwarf_w20_mips64,  LLDB_INVALID_REGNUM,    gdb_w20_mips64),
    DEFINE_MSA (w21,   NULL,   gcc_dwarf_w21_mips64,  gcc_dwarf_w21_mips64,  LLDB_INVALID_REGNUM,    gdb_w21_mips64),
    DEFINE_MSA (w22,   NULL,   gcc_dwarf_w22_mips64,  gcc_dwarf_w22_mips64,  LLDB_INVALID_REGNUM,    gdb_w22_mips64),
    DEFINE_MSA (w23,   NULL,   gcc_dwarf_w23_mips64,  gcc_dwarf_w23_mips64,  LLDB_INVALID_REGNUM,    gdb_w23_mips64),
    DEFINE_MSA (w24,   NULL,   gcc_dwarf_w24_mips64,  gcc_dwarf_w24_mips64,  LLDB_INVALID_REGNUM,    gdb_w24_mips64),
    DEFINE_MSA (w25,   NULL,   gcc_dwarf_w25_mips64,  gcc_dwarf_w25_mips64,  LLDB_INVALID_REGNUM,    gdb_w25_mips64),
    DEFINE_MSA (w26,   NULL,   gcc_dwarf_w26_mips64,  gcc_dwarf_w26_mips64,  LLDB_INVALID_REGNUM,    gdb_w26_mips64),
    DEFINE_MSA (w27,   NULL,   gcc_dwarf_w27_mips64,  gcc_dwarf_w27_mips64,  LLDB_INVALID_REGNUM,    gdb_w27_mips64),
    DEFINE_MSA (w28,   NULL,   gcc_dwarf_w28_mips64,  gcc_dwarf_w28_mips64,  LLDB_INVALID_REGNUM,    gdb_w28_mips64),
    DEFINE_MSA (w29,   NULL,   gcc_dwarf_w29_mips64,  gcc_dwarf_w29_mips64,  LLDB_INVALID_REGNUM,    gdb_w29_mips64),
    DEFINE_MSA (w30,   NULL,   gcc_dwarf_w30_mips64,  gcc_dwarf_w30_mips64,  LLDB_INVALID_REGNUM,    gdb_w30_mips64),
    DEFINE_MSA (w31,   NULL,   gcc_dwarf_w31_mips64,  gcc_dwarf_w31_mips64,  LLDB_INVALID_REGNUM,    gdb_w31_mips64),
    DEFINE_MSA_INFO (mcsr,  NULL,   gcc_dwarf_mcsr_mips64, gcc_dwarf_mcsr_mips64, LLDB_INVALID_REGNUM,    gdb_mcsr_mips64),
    DEFINE_MSA_INFO (mir,   NULL,   gcc_dwarf_mir_mips64,  gcc_dwarf_mir_mips64,  LLDB_INVALID_REGNUM,    gdb_mir_mips64),
    DEFINE_MSA_INFO (fcsr,  NULL,   gcc_dwarf_fcsr_mips64, gcc_dwarf_fcsr_mips64, LLDB_INVALID_REGNUM,    gdb_fcsr_mips64),
    DEFINE_MSA_INFO (fir,   NULL,   gcc_dwarf_fir_mips64,  gcc_dwarf_fir_mips64,  LLDB_INVALID_REGNUM,    gdb_fir_mips64),
    DEFINE_MSA_INFO (config5, NULL,   gcc_dwarf_config5_mips64,  gcc_dwarf_config5_mips64,  LLDB_INVALID_REGNUM,    gdb_config5_mips64)
#endif
};

static_assert((sizeof(g_register_infos_mips64) / sizeof(g_register_infos_mips64[0])) == k_num_registers_mips64,
    "g_register_infos_mips64 has wrong number of register infos");

#undef DEFINE_GPR
#undef DEFINE_GPR_INFO
#undef DEFINE_FPR
#undef DEFINE_MSA
#undef DEFINE_MSA_INFO
#undef GPR_OFFSET
#undef FPR_OFFSET
#undef MSA_OFFSET

#endif // DECLARE_REGISTER_INFOS_MIPS64_STRUCT
