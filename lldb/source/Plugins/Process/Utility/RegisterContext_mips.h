//===-- RegisterContext_mips.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContext_mips64_H_
#define liblldb_RegisterContext_mips64_H_

// eh_frame and DWARF Register numbers (eRegisterKindEHFrame & eRegisterKindDWARF)

enum
{
    // GP Registers
    gcc_dwarf_zero_mips = 0,
    gcc_dwarf_r1_mips,
    gcc_dwarf_r2_mips,
    gcc_dwarf_r3_mips,
    gcc_dwarf_r4_mips,
    gcc_dwarf_r5_mips,
    gcc_dwarf_r6_mips,
    gcc_dwarf_r7_mips,
    gcc_dwarf_r8_mips,
    gcc_dwarf_r9_mips,
    gcc_dwarf_r10_mips,
    gcc_dwarf_r11_mips,
    gcc_dwarf_r12_mips,
    gcc_dwarf_r13_mips,
    gcc_dwarf_r14_mips,
    gcc_dwarf_r15_mips,
    gcc_dwarf_r16_mips,
    gcc_dwarf_r17_mips,
    gcc_dwarf_r18_mips,
    gcc_dwarf_r19_mips,
    gcc_dwarf_r20_mips,
    gcc_dwarf_r21_mips,
    gcc_dwarf_r22_mips,
    gcc_dwarf_r23_mips,
    gcc_dwarf_r24_mips,
    gcc_dwarf_r25_mips,
    gcc_dwarf_r26_mips,
    gcc_dwarf_r27_mips,
    gcc_dwarf_gp_mips,
    gcc_dwarf_sp_mips,
    gcc_dwarf_r30_mips,
    gcc_dwarf_ra_mips,
    gcc_dwarf_sr_mips,
    gcc_dwarf_lo_mips,
    gcc_dwarf_hi_mips,
    gcc_dwarf_bad_mips,
    gcc_dwarf_cause_mips,
    gcc_dwarf_pc_mips,
    gcc_dwarf_f0_mips,
    gcc_dwarf_f1_mips,
    gcc_dwarf_f2_mips,
    gcc_dwarf_f3_mips,
    gcc_dwarf_f4_mips,
    gcc_dwarf_f5_mips,
    gcc_dwarf_f6_mips,
    gcc_dwarf_f7_mips,
    gcc_dwarf_f8_mips,
    gcc_dwarf_f9_mips,
    gcc_dwarf_f10_mips,
    gcc_dwarf_f11_mips,
    gcc_dwarf_f12_mips,
    gcc_dwarf_f13_mips,
    gcc_dwarf_f14_mips,
    gcc_dwarf_f15_mips,
    gcc_dwarf_f16_mips,
    gcc_dwarf_f17_mips,
    gcc_dwarf_f18_mips,
    gcc_dwarf_f19_mips,
    gcc_dwarf_f20_mips,
    gcc_dwarf_f21_mips,
    gcc_dwarf_f22_mips,
    gcc_dwarf_f23_mips,
    gcc_dwarf_f24_mips,
    gcc_dwarf_f25_mips,
    gcc_dwarf_f26_mips,
    gcc_dwarf_f27_mips,
    gcc_dwarf_f28_mips,
    gcc_dwarf_f29_mips,
    gcc_dwarf_f30_mips,
    gcc_dwarf_f31_mips,
    gcc_dwarf_fcsr_mips,
    gcc_dwarf_fir_mips,
    gcc_dwarf_w0_mips,
    gcc_dwarf_w1_mips,
    gcc_dwarf_w2_mips,
    gcc_dwarf_w3_mips,
    gcc_dwarf_w4_mips,
    gcc_dwarf_w5_mips,
    gcc_dwarf_w6_mips,
    gcc_dwarf_w7_mips,
    gcc_dwarf_w8_mips,
    gcc_dwarf_w9_mips,
    gcc_dwarf_w10_mips,
    gcc_dwarf_w11_mips,
    gcc_dwarf_w12_mips,
    gcc_dwarf_w13_mips,
    gcc_dwarf_w14_mips,
    gcc_dwarf_w15_mips,
    gcc_dwarf_w16_mips,
    gcc_dwarf_w17_mips,
    gcc_dwarf_w18_mips,
    gcc_dwarf_w19_mips,
    gcc_dwarf_w20_mips,
    gcc_dwarf_w21_mips,
    gcc_dwarf_w22_mips,
    gcc_dwarf_w23_mips,
    gcc_dwarf_w24_mips,
    gcc_dwarf_w25_mips,
    gcc_dwarf_w26_mips,
    gcc_dwarf_w27_mips,
    gcc_dwarf_w28_mips,
    gcc_dwarf_w29_mips,
    gcc_dwarf_w30_mips,
    gcc_dwarf_w31_mips,
    gcc_dwarf_mcsr_mips,
    gcc_dwarf_mir_mips,
    gcc_dwarf_config5_mips,
    gcc_dwarf_ic_mips,
    gcc_dwarf_dummy_mips
};

enum
{
    gcc_dwarf_zero_mips64 = 0,
    gcc_dwarf_r1_mips64,
    gcc_dwarf_r2_mips64,
    gcc_dwarf_r3_mips64,
    gcc_dwarf_r4_mips64,
    gcc_dwarf_r5_mips64,
    gcc_dwarf_r6_mips64,
    gcc_dwarf_r7_mips64,
    gcc_dwarf_r8_mips64,
    gcc_dwarf_r9_mips64,
    gcc_dwarf_r10_mips64,
    gcc_dwarf_r11_mips64,
    gcc_dwarf_r12_mips64,
    gcc_dwarf_r13_mips64,
    gcc_dwarf_r14_mips64,
    gcc_dwarf_r15_mips64,
    gcc_dwarf_r16_mips64,
    gcc_dwarf_r17_mips64,
    gcc_dwarf_r18_mips64,
    gcc_dwarf_r19_mips64,
    gcc_dwarf_r20_mips64,
    gcc_dwarf_r21_mips64,
    gcc_dwarf_r22_mips64,
    gcc_dwarf_r23_mips64,
    gcc_dwarf_r24_mips64,
    gcc_dwarf_r25_mips64,
    gcc_dwarf_r26_mips64,
    gcc_dwarf_r27_mips64,
    gcc_dwarf_gp_mips64,
    gcc_dwarf_sp_mips64,
    gcc_dwarf_r30_mips64,
    gcc_dwarf_ra_mips64,
    gcc_dwarf_sr_mips64,
    gcc_dwarf_lo_mips64,
    gcc_dwarf_hi_mips64,
    gcc_dwarf_bad_mips64,
    gcc_dwarf_cause_mips64,
    gcc_dwarf_pc_mips64,
    gcc_dwarf_f0_mips64,
    gcc_dwarf_f1_mips64,
    gcc_dwarf_f2_mips64,
    gcc_dwarf_f3_mips64,
    gcc_dwarf_f4_mips64,
    gcc_dwarf_f5_mips64,
    gcc_dwarf_f6_mips64,
    gcc_dwarf_f7_mips64,
    gcc_dwarf_f8_mips64,
    gcc_dwarf_f9_mips64,
    gcc_dwarf_f10_mips64,
    gcc_dwarf_f11_mips64,
    gcc_dwarf_f12_mips64,
    gcc_dwarf_f13_mips64,
    gcc_dwarf_f14_mips64,
    gcc_dwarf_f15_mips64,
    gcc_dwarf_f16_mips64,
    gcc_dwarf_f17_mips64,
    gcc_dwarf_f18_mips64,
    gcc_dwarf_f19_mips64,
    gcc_dwarf_f20_mips64,
    gcc_dwarf_f21_mips64,
    gcc_dwarf_f22_mips64,
    gcc_dwarf_f23_mips64,
    gcc_dwarf_f24_mips64,
    gcc_dwarf_f25_mips64,
    gcc_dwarf_f26_mips64,
    gcc_dwarf_f27_mips64,
    gcc_dwarf_f28_mips64,
    gcc_dwarf_f29_mips64,
    gcc_dwarf_f30_mips64,
    gcc_dwarf_f31_mips64,
    gcc_dwarf_fcsr_mips64,
    gcc_dwarf_fir_mips64,
    gcc_dwarf_ic_mips64,
    gcc_dwarf_dummy_mips64,
    gcc_dwarf_w0_mips64,
    gcc_dwarf_w1_mips64,
    gcc_dwarf_w2_mips64,
    gcc_dwarf_w3_mips64,
    gcc_dwarf_w4_mips64,
    gcc_dwarf_w5_mips64,
    gcc_dwarf_w6_mips64,
    gcc_dwarf_w7_mips64,
    gcc_dwarf_w8_mips64,
    gcc_dwarf_w9_mips64,
    gcc_dwarf_w10_mips64,
    gcc_dwarf_w11_mips64,
    gcc_dwarf_w12_mips64,
    gcc_dwarf_w13_mips64,
    gcc_dwarf_w14_mips64,
    gcc_dwarf_w15_mips64,
    gcc_dwarf_w16_mips64,
    gcc_dwarf_w17_mips64,
    gcc_dwarf_w18_mips64,
    gcc_dwarf_w19_mips64,
    gcc_dwarf_w20_mips64,
    gcc_dwarf_w21_mips64,
    gcc_dwarf_w22_mips64,
    gcc_dwarf_w23_mips64,
    gcc_dwarf_w24_mips64,
    gcc_dwarf_w25_mips64,
    gcc_dwarf_w26_mips64,
    gcc_dwarf_w27_mips64,
    gcc_dwarf_w28_mips64,
    gcc_dwarf_w29_mips64,
    gcc_dwarf_w30_mips64,
    gcc_dwarf_w31_mips64,
    gcc_dwarf_mcsr_mips64,
    gcc_dwarf_mir_mips64,
    gcc_dwarf_config5_mips64,
};

// GDB Register numbers (eRegisterKindGDB)
enum
{
    gdb_zero_mips = 0,
    gdb_r1_mips,
    gdb_r2_mips,
    gdb_r3_mips,
    gdb_r4_mips,
    gdb_r5_mips,
    gdb_r6_mips,
    gdb_r7_mips,
    gdb_r8_mips,
    gdb_r9_mips,
    gdb_r10_mips,
    gdb_r11_mips,
    gdb_r12_mips,
    gdb_r13_mips,
    gdb_r14_mips,
    gdb_r15_mips,
    gdb_r16_mips,
    gdb_r17_mips,
    gdb_r18_mips,
    gdb_r19_mips,
    gdb_r20_mips,
    gdb_r21_mips,
    gdb_r22_mips,
    gdb_r23_mips,
    gdb_r24_mips,
    gdb_r25_mips,
    gdb_r26_mips,
    gdb_r27_mips,
    gdb_gp_mips,
    gdb_sp_mips,
    gdb_r30_mips,
    gdb_ra_mips,
    gdb_sr_mips,
    gdb_lo_mips,
    gdb_hi_mips,
    gdb_bad_mips,
    gdb_cause_mips,
    gdb_pc_mips,
    gdb_f0_mips,
    gdb_f1_mips,
    gdb_f2_mips,
    gdb_f3_mips,
    gdb_f4_mips,
    gdb_f5_mips,
    gdb_f6_mips,
    gdb_f7_mips,
    gdb_f8_mips,
    gdb_f9_mips,
    gdb_f10_mips,
    gdb_f11_mips,
    gdb_f12_mips,
    gdb_f13_mips,
    gdb_f14_mips,
    gdb_f15_mips,
    gdb_f16_mips,
    gdb_f17_mips,
    gdb_f18_mips,
    gdb_f19_mips,
    gdb_f20_mips,
    gdb_f21_mips,
    gdb_f22_mips,
    gdb_f23_mips,
    gdb_f24_mips,
    gdb_f25_mips,
    gdb_f26_mips,
    gdb_f27_mips,
    gdb_f28_mips,
    gdb_f29_mips,
    gdb_f30_mips,
    gdb_f31_mips,
    gdb_fcsr_mips,
    gdb_fir_mips,
    gdb_w0_mips,
    gdb_w1_mips,
    gdb_w2_mips,
    gdb_w3_mips,
    gdb_w4_mips,
    gdb_w5_mips,
    gdb_w6_mips,
    gdb_w7_mips,
    gdb_w8_mips,
    gdb_w9_mips,
    gdb_w10_mips,
    gdb_w11_mips,
    gdb_w12_mips,
    gdb_w13_mips,
    gdb_w14_mips,
    gdb_w15_mips,
    gdb_w16_mips,
    gdb_w17_mips,
    gdb_w18_mips,
    gdb_w19_mips,
    gdb_w20_mips,
    gdb_w21_mips,
    gdb_w22_mips,
    gdb_w23_mips,
    gdb_w24_mips,
    gdb_w25_mips,
    gdb_w26_mips,
    gdb_w27_mips,
    gdb_w28_mips,
    gdb_w29_mips,
    gdb_w30_mips,
    gdb_w31_mips,
    gdb_mcsr_mips,
    gdb_mir_mips,
    gdb_config5_mips,
    gdb_ic_mips,
    gdb_dummy_mips
};

enum
{
    gdb_zero_mips64 = 0,
    gdb_r1_mips64,
    gdb_r2_mips64,
    gdb_r3_mips64,
    gdb_r4_mips64,
    gdb_r5_mips64,
    gdb_r6_mips64,
    gdb_r7_mips64,
    gdb_r8_mips64,
    gdb_r9_mips64,
    gdb_r10_mips64,
    gdb_r11_mips64,
    gdb_r12_mips64,
    gdb_r13_mips64,
    gdb_r14_mips64,
    gdb_r15_mips64,
    gdb_r16_mips64,
    gdb_r17_mips64,
    gdb_r18_mips64,
    gdb_r19_mips64,
    gdb_r20_mips64,
    gdb_r21_mips64,
    gdb_r22_mips64,
    gdb_r23_mips64,
    gdb_r24_mips64,
    gdb_r25_mips64,
    gdb_r26_mips64,
    gdb_r27_mips64,
    gdb_gp_mips64,
    gdb_sp_mips64,
    gdb_r30_mips64,
    gdb_ra_mips64,
    gdb_sr_mips64,
    gdb_lo_mips64,
    gdb_hi_mips64,
    gdb_bad_mips64,
    gdb_cause_mips64,
    gdb_pc_mips64,
    gdb_f0_mips64,
    gdb_f1_mips64,
    gdb_f2_mips64,
    gdb_f3_mips64,
    gdb_f4_mips64,
    gdb_f5_mips64,
    gdb_f6_mips64,
    gdb_f7_mips64,
    gdb_f8_mips64,
    gdb_f9_mips64,
    gdb_f10_mips64,
    gdb_f11_mips64,
    gdb_f12_mips64,
    gdb_f13_mips64,
    gdb_f14_mips64,
    gdb_f15_mips64,
    gdb_f16_mips64,
    gdb_f17_mips64,
    gdb_f18_mips64,
    gdb_f19_mips64,
    gdb_f20_mips64,
    gdb_f21_mips64,
    gdb_f22_mips64,
    gdb_f23_mips64,
    gdb_f24_mips64,
    gdb_f25_mips64,
    gdb_f26_mips64,
    gdb_f27_mips64,
    gdb_f28_mips64,
    gdb_f29_mips64,
    gdb_f30_mips64,
    gdb_f31_mips64,
    gdb_fcsr_mips64,
    gdb_fir_mips64,
    gdb_ic_mips64,
    gdb_dummy_mips64,
    gdb_w0_mips64,
    gdb_w1_mips64,
    gdb_w2_mips64,
    gdb_w3_mips64,
    gdb_w4_mips64,
    gdb_w5_mips64,
    gdb_w6_mips64,
    gdb_w7_mips64,
    gdb_w8_mips64,
    gdb_w9_mips64,
    gdb_w10_mips64,
    gdb_w11_mips64,
    gdb_w12_mips64,
    gdb_w13_mips64,
    gdb_w14_mips64,
    gdb_w15_mips64,
    gdb_w16_mips64,
    gdb_w17_mips64,
    gdb_w18_mips64,
    gdb_w19_mips64,
    gdb_w20_mips64,
    gdb_w21_mips64,
    gdb_w22_mips64,
    gdb_w23_mips64,
    gdb_w24_mips64,
    gdb_w25_mips64,
    gdb_w26_mips64,
    gdb_w27_mips64,
    gdb_w28_mips64,
    gdb_w29_mips64,
    gdb_w30_mips64,
    gdb_w31_mips64,
    gdb_mcsr_mips64,
    gdb_mir_mips64,
    gdb_config5_mips64,
};

struct IOVEC_mips
{
    void    *iov_base;
    size_t   iov_len;
};

// GP registers
struct GPR_linux_mips
{
    uint64_t zero;
    uint64_t r1;
    uint64_t r2;
    uint64_t r3;
    uint64_t r4;
    uint64_t r5;
    uint64_t r6;
    uint64_t r7;
    uint64_t r8;
    uint64_t r9;
    uint64_t r10;
    uint64_t r11;
    uint64_t r12;
    uint64_t r13;
    uint64_t r14;
    uint64_t r15;
    uint64_t r16;
    uint64_t r17;
    uint64_t r18;
    uint64_t r19;
    uint64_t r20;
    uint64_t r21;
    uint64_t r22;
    uint64_t r23;
    uint64_t r24;
    uint64_t r25;
    uint64_t r26;
    uint64_t r27;
    uint64_t gp;
    uint64_t sp;
    uint64_t r30;
    uint64_t ra;
    uint64_t mullo;
    uint64_t mulhi;
    uint64_t pc;
    uint64_t badvaddr;
    uint64_t sr;
    uint64_t cause;
    uint64_t config5;
};

struct FPR_linux_mips
{
     uint64_t f0;
     uint64_t f1;
     uint64_t f2;
     uint64_t f3;
     uint64_t f4;
     uint64_t f5;
     uint64_t f6;
     uint64_t f7;
     uint64_t f8;
     uint64_t f9;
     uint64_t f10;
     uint64_t f11;
     uint64_t f12;
     uint64_t f13;
     uint64_t f14;
     uint64_t f15;
     uint64_t f16;
     uint64_t f17;
     uint64_t f18;
     uint64_t f19;
     uint64_t f20;
     uint64_t f21;
     uint64_t f22;
     uint64_t f23;
     uint64_t f24;
     uint64_t f25;
     uint64_t f26;
     uint64_t f27;
     uint64_t f28;
     uint64_t f29;
     uint64_t f30;
     uint64_t f31;
     uint32_t fcsr;
     uint32_t fir;
     uint32_t config5;
};

struct MSAReg
{
    uint8_t byte[16];
};

struct MSA_linux_mips
{
    MSAReg  w0;
    MSAReg  w1;
    MSAReg  w2;
    MSAReg  w3;
    MSAReg  w4;
    MSAReg  w5;
    MSAReg  w6;
    MSAReg  w7;
    MSAReg  w8;
    MSAReg  w9;
    MSAReg  w10;
    MSAReg  w11;
    MSAReg  w12;
    MSAReg  w13;
    MSAReg  w14;
    MSAReg  w15;
    MSAReg  w16;
    MSAReg  w17;
    MSAReg  w18;
    MSAReg  w19;
    MSAReg  w20;
    MSAReg  w21;
    MSAReg  w22;
    MSAReg  w23;
    MSAReg  w24;
    MSAReg  w25;
    MSAReg  w26;
    MSAReg  w27;
    MSAReg  w28;
    MSAReg  w29;
    MSAReg  w30;
    MSAReg  w31;
    uint32_t fcsr;      /* FPU control status register */
    uint32_t fir;       /* FPU implementaion revision */
    uint32_t mcsr;      /* MSA control status register */
    uint32_t mir;       /* MSA implementation revision */
    uint32_t config5;   /* Config5 register */
};

struct UserArea
{
    GPR_linux_mips gpr; // General purpose registers.
    FPR_linux_mips fpr; // Floating point registers.
    MSA_linux_mips msa; // MSA registers.
};

#endif // liblldb_RegisterContext_mips64_H_
