//===-- CodeViewRegisterMapping.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodeViewRegisterMapping.h"

#include "lldb/lldb-defines.h"

#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"

using namespace lldb_private;

static const uint32_t g_code_view_to_lldb_registers_arm64[] = {
    LLDB_INVALID_REGNUM, // NONE
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    registers::arm64::id::gpr_w0,  // ARM64_W0, 10)
    registers::arm64::id::gpr_w1,  // ARM64_W1, 11)
    registers::arm64::id::gpr_w2,  // ARM64_W2, 12)
    registers::arm64::id::gpr_w3,  // ARM64_W3, 13)
    registers::arm64::id::gpr_w4,  // ARM64_W4, 14)
    registers::arm64::id::gpr_w5,  // ARM64_W5, 15)
    registers::arm64::id::gpr_w6,  // ARM64_W6, 16)
    registers::arm64::id::gpr_w7,  // ARM64_W7, 17)
    registers::arm64::id::gpr_w8,  // ARM64_W8, 18)
    registers::arm64::id::gpr_w9,  // ARM64_W9, 19)
    registers::arm64::id::gpr_w10, // ARM64_W10, 20)
    registers::arm64::id::gpr_w11, // ARM64_W11, 21)
    registers::arm64::id::gpr_w12, // ARM64_W12, 22)
    registers::arm64::id::gpr_w13, // ARM64_W13, 23)
    registers::arm64::id::gpr_w14, // ARM64_W14, 24)
    registers::arm64::id::gpr_w15, // ARM64_W15, 25)
    registers::arm64::id::gpr_w16, // ARM64_W16, 26)
    registers::arm64::id::gpr_w17, // ARM64_W17, 27)
    registers::arm64::id::gpr_w18, // ARM64_W18, 28)
    registers::arm64::id::gpr_w19, // ARM64_W19, 29)
    registers::arm64::id::gpr_w20, // ARM64_W20, 30)
    registers::arm64::id::gpr_w21, // ARM64_W21, 31)
    registers::arm64::id::gpr_w22, // ARM64_W22, 32)
    registers::arm64::id::gpr_w23, // ARM64_W23, 33)
    registers::arm64::id::gpr_w24, // ARM64_W24, 34)
    registers::arm64::id::gpr_w25, // ARM64_W25, 35)
    registers::arm64::id::gpr_w26, // ARM64_W26, 36)
    registers::arm64::id::gpr_w27, // ARM64_W27, 37)
    registers::arm64::id::gpr_w28, // ARM64_W28, 38)
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    registers::arm64::id::gpr_x0,  // ARM64_X0, 50)
    registers::arm64::id::gpr_x1,  // ARM64_X1, 51)
    registers::arm64::id::gpr_x2,  // ARM64_X2, 52)
    registers::arm64::id::gpr_x3,  // ARM64_X3, 53)
    registers::arm64::id::gpr_x4,  // ARM64_X4, 54)
    registers::arm64::id::gpr_x5,  // ARM64_X5, 55)
    registers::arm64::id::gpr_x6,  // ARM64_X6, 56)
    registers::arm64::id::gpr_x7,  // ARM64_X7, 57)
    registers::arm64::id::gpr_x8,  // ARM64_X8, 58)
    registers::arm64::id::gpr_x9,  // ARM64_X9, 59)
    registers::arm64::id::gpr_x10, // ARM64_X10, 60)
    registers::arm64::id::gpr_x11, // ARM64_X11, 61)
    registers::arm64::id::gpr_x12, // ARM64_X12, 62)
    registers::arm64::id::gpr_x13, // ARM64_X13, 63)
    registers::arm64::id::gpr_x14, // ARM64_X14, 64)
    registers::arm64::id::gpr_x15, // ARM64_X15, 65)
    registers::arm64::id::gpr_x16, // ARM64_X16, 66)
    registers::arm64::id::gpr_x17, // ARM64_X17, 67)
    registers::arm64::id::gpr_x18, // ARM64_X18, 68)
    registers::arm64::id::gpr_x19, // ARM64_X19, 69)
    registers::arm64::id::gpr_x20, // ARM64_X20, 70)
    registers::arm64::id::gpr_x21, // ARM64_X21, 71)
    registers::arm64::id::gpr_x22, // ARM64_X22, 72)
    registers::arm64::id::gpr_x23, // ARM64_X23, 73)
    registers::arm64::id::gpr_x24, // ARM64_X24, 74)
    registers::arm64::id::gpr_x25, // ARM64_X25, 75)
    registers::arm64::id::gpr_x26, // ARM64_X26, 76)
    registers::arm64::id::gpr_x27, // ARM64_X27, 77)
    registers::arm64::id::gpr_x28, // ARM64_X28, 78)
    registers::arm64::id::gpr_fp,  // ARM64_FP, 79)
    registers::arm64::id::gpr_lr,  // ARM64_LR, 80)
    registers::arm64::id::gpr_sp,  // ARM64_SP, 81)
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    registers::arm64::id::gpr_cpsr, // ARM64_NZCV, 90)
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    registers::arm64::id::fpu_s0,  // (ARM64_S0, 100)
    registers::arm64::id::fpu_s1,  // (ARM64_S1, 101)
    registers::arm64::id::fpu_s2,  // (ARM64_S2, 102)
    registers::arm64::id::fpu_s3,  // (ARM64_S3, 103)
    registers::arm64::id::fpu_s4,  // (ARM64_S4, 104)
    registers::arm64::id::fpu_s5,  // (ARM64_S5, 105)
    registers::arm64::id::fpu_s6,  // (ARM64_S6, 106)
    registers::arm64::id::fpu_s7,  // (ARM64_S7, 107)
    registers::arm64::id::fpu_s8,  // (ARM64_S8, 108)
    registers::arm64::id::fpu_s9,  // (ARM64_S9, 109)
    registers::arm64::id::fpu_s10, // (ARM64_S10, 110)
    registers::arm64::id::fpu_s11, // (ARM64_S11, 111)
    registers::arm64::id::fpu_s12, // (ARM64_S12, 112)
    registers::arm64::id::fpu_s13, // (ARM64_S13, 113)
    registers::arm64::id::fpu_s14, // (ARM64_S14, 114)
    registers::arm64::id::fpu_s15, // (ARM64_S15, 115)
    registers::arm64::id::fpu_s16, // (ARM64_S16, 116)
    registers::arm64::id::fpu_s17, // (ARM64_S17, 117)
    registers::arm64::id::fpu_s18, // (ARM64_S18, 118)
    registers::arm64::id::fpu_s19, // (ARM64_S19, 119)
    registers::arm64::id::fpu_s20, // (ARM64_S20, 120)
    registers::arm64::id::fpu_s21, // (ARM64_S21, 121)
    registers::arm64::id::fpu_s22, // (ARM64_S22, 122)
    registers::arm64::id::fpu_s23, // (ARM64_S23, 123)
    registers::arm64::id::fpu_s24, // (ARM64_S24, 124)
    registers::arm64::id::fpu_s25, // (ARM64_S25, 125)
    registers::arm64::id::fpu_s26, // (ARM64_S26, 126)
    registers::arm64::id::fpu_s27, // (ARM64_S27, 127)
    registers::arm64::id::fpu_s28, // (ARM64_S28, 128)
    registers::arm64::id::fpu_s29, // (ARM64_S29, 129)
    registers::arm64::id::fpu_s30, // (ARM64_S30, 130)
    registers::arm64::id::fpu_s31, // (ARM64_S31, 131)
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    registers::arm64::id::fpu_d0,  // (ARM64_D0, 140)
    registers::arm64::id::fpu_d1,  // (ARM64_D1, 141)
    registers::arm64::id::fpu_d2,  // (ARM64_D2, 142)
    registers::arm64::id::fpu_d3,  // (ARM64_D3, 143)
    registers::arm64::id::fpu_d4,  // (ARM64_D4, 144)
    registers::arm64::id::fpu_d5,  // (ARM64_D5, 145)
    registers::arm64::id::fpu_d6,  // (ARM64_D6, 146)
    registers::arm64::id::fpu_d7,  // (ARM64_D7, 147)
    registers::arm64::id::fpu_d8,  // (ARM64_D8, 148)
    registers::arm64::id::fpu_d9,  // (ARM64_D9, 149)
    registers::arm64::id::fpu_d10, // (ARM64_D10, 150)
    registers::arm64::id::fpu_d11, // (ARM64_D11, 151)
    registers::arm64::id::fpu_d12, // (ARM64_D12, 152)
    registers::arm64::id::fpu_d13, // (ARM64_D13, 153)
    registers::arm64::id::fpu_d14, // (ARM64_D14, 154)
    registers::arm64::id::fpu_d15, // (ARM64_D15, 155)
    registers::arm64::id::fpu_d16, // (ARM64_D16, 156)
    registers::arm64::id::fpu_d17, // (ARM64_D17, 157)
    registers::arm64::id::fpu_d18, // (ARM64_D18, 158)
    registers::arm64::id::fpu_d19, // (ARM64_D19, 159)
    registers::arm64::id::fpu_d20, // (ARM64_D20, 160)
    registers::arm64::id::fpu_d21, // (ARM64_D21, 161)
    registers::arm64::id::fpu_d22, // (ARM64_D22, 162)
    registers::arm64::id::fpu_d23, // (ARM64_D23, 163)
    registers::arm64::id::fpu_d24, // (ARM64_D24, 164)
    registers::arm64::id::fpu_d25, // (ARM64_D25, 165)
    registers::arm64::id::fpu_d26, // (ARM64_D26, 166)
    registers::arm64::id::fpu_d27, // (ARM64_D27, 167)
    registers::arm64::id::fpu_d28, // (ARM64_D28, 168)
    registers::arm64::id::fpu_d29, // (ARM64_D29, 169)
    registers::arm64::id::fpu_d30, // (ARM64_D30, 170)
    registers::arm64::id::fpu_d31, // (ARM64_D31, 171)
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    registers::arm64::id::fpu_v0,  // (ARM64_Q0, 180)
    registers::arm64::id::fpu_v1,  // (ARM64_Q1, 181)
    registers::arm64::id::fpu_v2,  // (ARM64_Q2, 182)
    registers::arm64::id::fpu_v3,  // (ARM64_Q3, 183)
    registers::arm64::id::fpu_v4,  // (ARM64_Q4, 184)
    registers::arm64::id::fpu_v5,  // (ARM64_Q5, 185)
    registers::arm64::id::fpu_v6,  // (ARM64_Q6, 186)
    registers::arm64::id::fpu_v7,  // (ARM64_Q7, 187)
    registers::arm64::id::fpu_v8,  // (ARM64_Q8, 188)
    registers::arm64::id::fpu_v9,  // (ARM64_Q9, 189)
    registers::arm64::id::fpu_v10, // (ARM64_Q10, 190)
    registers::arm64::id::fpu_v11, // (ARM64_Q11, 191)
    registers::arm64::id::fpu_v12, // (ARM64_Q12, 192)
    registers::arm64::id::fpu_v13, // (ARM64_Q13, 193)
    registers::arm64::id::fpu_v14, // (ARM64_Q14, 194)
    registers::arm64::id::fpu_v15, // (ARM64_Q15, 195)
    registers::arm64::id::fpu_v16, // (ARM64_Q16, 196)
    registers::arm64::id::fpu_v17, // (ARM64_Q17, 197)
    registers::arm64::id::fpu_v18, // (ARM64_Q18, 198)
    registers::arm64::id::fpu_v19, // (ARM64_Q19, 199)
    registers::arm64::id::fpu_v20, // (ARM64_Q20, 200)
    registers::arm64::id::fpu_v21, // (ARM64_Q21, 201)
    registers::arm64::id::fpu_v22, // (ARM64_Q22, 202)
    registers::arm64::id::fpu_v23, // (ARM64_Q23, 203)
    registers::arm64::id::fpu_v24, // (ARM64_Q24, 204)
    registers::arm64::id::fpu_v25, // (ARM64_Q25, 205)
    registers::arm64::id::fpu_v26, // (ARM64_Q26, 206)
    registers::arm64::id::fpu_v27, // (ARM64_Q27, 207)
    registers::arm64::id::fpu_v28, // (ARM64_Q28, 208)
    registers::arm64::id::fpu_v29, // (ARM64_Q29, 209)
    registers::arm64::id::fpu_v30, // (ARM64_Q30, 210)
    registers::arm64::id::fpu_v31, // (ARM64_Q31, 211)
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    registers::arm64::id::fpu_fpsr // ARM64_FPSR, 220)
};

static const uint32_t g_code_view_to_lldb_registers_x86[] = {
    LLDB_INVALID_REGNUM, // NONE
    registers::x86::id::al,        // AL
    registers::x86::id::cl,        // CL
    registers::x86::id::dl,        // DL
    registers::x86::id::bl,        // BL
    registers::x86::id::ah,        // AH
    registers::x86::id::ch,        // CH
    registers::x86::id::dh,        // DH
    registers::x86::id::bh,        // BH
    registers::x86::id::ax,        // AX
    registers::x86::id::cx,        // CX
    registers::x86::id::dx,        // DX
    registers::x86::id::bx,        // BX
    registers::x86::id::sp,        // SP
    registers::x86::id::bp,        // BP
    registers::x86::id::si,        // SI
    registers::x86::id::di,        // DI
    registers::x86::id::eax,       // EAX
    registers::x86::id::ecx,       // ECX
    registers::x86::id::edx,       // EDX
    registers::x86::id::ebx,       // EBX
    registers::x86::id::esp,       // ESP
    registers::x86::id::ebp,       // EBP
    registers::x86::id::esi,       // ESI
    registers::x86::id::edi,       // EDI
    registers::x86::id::es,        // ES
    registers::x86::id::cs,        // CS
    registers::x86::id::ss,        // SS
    registers::x86::id::ds,        // DS
    registers::x86::id::fs,        // FS
    registers::x86::id::gs,        // GS
    LLDB_INVALID_REGNUM, // IP
    LLDB_INVALID_REGNUM, // FLAGS
    registers::x86::id::eip,       // EIP
    registers::x86::id::eflags,    // EFLAGS
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // TEMP
    LLDB_INVALID_REGNUM, // TEMPH
    LLDB_INVALID_REGNUM, // QUOTE
    LLDB_INVALID_REGNUM, // PCDR3
    LLDB_INVALID_REGNUM, // PCDR4
    LLDB_INVALID_REGNUM, // PCDR5
    LLDB_INVALID_REGNUM, // PCDR6
    LLDB_INVALID_REGNUM, // PCDR7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // CR0
    LLDB_INVALID_REGNUM, // CR1
    LLDB_INVALID_REGNUM, // CR2
    LLDB_INVALID_REGNUM, // CR3
    LLDB_INVALID_REGNUM, // CR4
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    registers::x86::id::dr0, // DR0
    registers::x86::id::dr1, // DR1
    registers::x86::id::dr2, // DR2
    registers::x86::id::dr3, // DR3
    registers::x86::id::dr4, // DR4
    registers::x86::id::dr5, // DR5
    registers::x86::id::dr6, // DR6
    registers::x86::id::dr7, // DR7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // GDTR
    LLDB_INVALID_REGNUM, // GDTL
    LLDB_INVALID_REGNUM, // IDTR
    LLDB_INVALID_REGNUM, // IDTL
    LLDB_INVALID_REGNUM, // LDTR
    LLDB_INVALID_REGNUM, // TR
    LLDB_INVALID_REGNUM, // PSEUDO1
    LLDB_INVALID_REGNUM, // PSEUDO2
    LLDB_INVALID_REGNUM, // PSEUDO3
    LLDB_INVALID_REGNUM, // PSEUDO4
    LLDB_INVALID_REGNUM, // PSEUDO5
    LLDB_INVALID_REGNUM, // PSEUDO6
    LLDB_INVALID_REGNUM, // PSEUDO7
    LLDB_INVALID_REGNUM, // PSEUDO8
    LLDB_INVALID_REGNUM, // PSEUDO9
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    registers::x86::id::st0,       // ST0
    registers::x86::id::st1,       // ST1
    registers::x86::id::st2,       // ST2
    registers::x86::id::st3,       // ST3
    registers::x86::id::st4,       // ST4
    registers::x86::id::st5,       // ST5
    registers::x86::id::st6,       // ST6
    registers::x86::id::st7,       // ST7
    LLDB_INVALID_REGNUM, // CTRL
    LLDB_INVALID_REGNUM, // STAT
    LLDB_INVALID_REGNUM, // TAG
    LLDB_INVALID_REGNUM, // FPIP
    LLDB_INVALID_REGNUM, // FPCS
    LLDB_INVALID_REGNUM, // FPDO
    LLDB_INVALID_REGNUM, // FPDS
    LLDB_INVALID_REGNUM, // ISEM
    LLDB_INVALID_REGNUM, // FPEIP
    LLDB_INVALID_REGNUM, // FPEDO
    registers::x86::id::mm0,       // MM0
    registers::x86::id::mm1,       // MM1
    registers::x86::id::mm2,       // MM2
    registers::x86::id::mm3,       // MM3
    registers::x86::id::mm4,       // MM4
    registers::x86::id::mm5,       // MM5
    registers::x86::id::mm6,       // MM6
    registers::x86::id::mm7,       // MM7
    registers::x86::id::xmm0,      // XMM0
    registers::x86::id::xmm1,      // XMM1
    registers::x86::id::xmm2,      // XMM2
    registers::x86::id::xmm3,      // XMM3
    registers::x86::id::xmm4,      // XMM4
    registers::x86::id::xmm5,      // XMM5
    registers::x86::id::xmm6,      // XMM6
    registers::x86::id::xmm7       // XMM7
};

static const uint32_t g_code_view_to_lldb_registers_x86_64[] = {
    LLDB_INVALID_REGNUM, // NONE
    registers::x86_64::id::al,      // AL
    registers::x86_64::id::cl,      // CL
    registers::x86_64::id::dl,      // DL
    registers::x86_64::id::bl,      // BL
    registers::x86_64::id::ah,      // AH
    registers::x86_64::id::ch,      // CH
    registers::x86_64::id::dh,      // DH
    registers::x86_64::id::bh,      // BH
    registers::x86_64::id::ax,      // AX
    registers::x86_64::id::cx,      // CX
    registers::x86_64::id::dx,      // DX
    registers::x86_64::id::bx,      // BX
    registers::x86_64::id::sp,      // SP
    registers::x86_64::id::bp,      // BP
    registers::x86_64::id::si,      // SI
    registers::x86_64::id::di,      // DI
    registers::x86_64::id::eax,     // EAX
    registers::x86_64::id::ecx,     // ECX
    registers::x86_64::id::edx,     // EDX
    registers::x86_64::id::ebx,     // EBX
    registers::x86_64::id::esp,     // ESP
    registers::x86_64::id::ebp,     // EBP
    registers::x86_64::id::esi,     // ESI
    registers::x86_64::id::edi,     // EDI
    registers::x86_64::id::es,      // ES
    registers::x86_64::id::cs,      // CS
    registers::x86_64::id::ss,      // SS
    registers::x86_64::id::ds,      // DS
    registers::x86_64::id::fs,      // FS
    registers::x86_64::id::gs,      // GS
    LLDB_INVALID_REGNUM, // IP
    LLDB_INVALID_REGNUM, // FLAGS
    LLDB_INVALID_REGNUM, // EIP
    LLDB_INVALID_REGNUM, // EFLAGS
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // TEMP
    LLDB_INVALID_REGNUM, // TEMPH
    LLDB_INVALID_REGNUM, // QUOTE
    LLDB_INVALID_REGNUM, // PCDR3
    LLDB_INVALID_REGNUM, // PCDR4
    LLDB_INVALID_REGNUM, // PCDR5
    LLDB_INVALID_REGNUM, // PCDR6
    LLDB_INVALID_REGNUM, // PCDR7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // CR0
    LLDB_INVALID_REGNUM, // CR1
    LLDB_INVALID_REGNUM, // CR2
    LLDB_INVALID_REGNUM, // CR3
    LLDB_INVALID_REGNUM, // CR4
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    registers::x86_64::id::dr0, // DR0
    registers::x86_64::id::dr1, // DR1
    registers::x86_64::id::dr2, // DR2
    registers::x86_64::id::dr3, // DR3
    registers::x86_64::id::dr4, // DR4
    registers::x86_64::id::dr5, // DR5
    registers::x86_64::id::dr6, // DR6
    registers::x86_64::id::dr7, // DR7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // GDTR
    LLDB_INVALID_REGNUM, // GDTL
    LLDB_INVALID_REGNUM, // IDTR
    LLDB_INVALID_REGNUM, // IDTL
    LLDB_INVALID_REGNUM, // LDTR
    LLDB_INVALID_REGNUM, // TR
    LLDB_INVALID_REGNUM, // PSEUDO1
    LLDB_INVALID_REGNUM, // PSEUDO2
    LLDB_INVALID_REGNUM, // PSEUDO3
    LLDB_INVALID_REGNUM, // PSEUDO4
    LLDB_INVALID_REGNUM, // PSEUDO5
    LLDB_INVALID_REGNUM, // PSEUDO6
    LLDB_INVALID_REGNUM, // PSEUDO7
    LLDB_INVALID_REGNUM, // PSEUDO8
    LLDB_INVALID_REGNUM, // PSEUDO9
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    registers::x86_64::id::st0,     // ST0
    registers::x86_64::id::st1,     // ST1
    registers::x86_64::id::st2,     // ST2
    registers::x86_64::id::st3,     // ST3
    registers::x86_64::id::st4,     // ST4
    registers::x86_64::id::st5,     // ST5
    registers::x86_64::id::st6,     // ST6
    registers::x86_64::id::st7,     // ST7
    LLDB_INVALID_REGNUM, // CTRL
    LLDB_INVALID_REGNUM, // STAT
    LLDB_INVALID_REGNUM, // TAG
    LLDB_INVALID_REGNUM, // FPIP
    LLDB_INVALID_REGNUM, // FPCS
    LLDB_INVALID_REGNUM, // FPDO
    LLDB_INVALID_REGNUM, // FPDS
    LLDB_INVALID_REGNUM, // ISEM
    LLDB_INVALID_REGNUM, // FPEIP
    LLDB_INVALID_REGNUM, // FPEDO
    registers::x86_64::id::mm0,     // MM0
    registers::x86_64::id::mm1,     // MM1
    registers::x86_64::id::mm2,     // MM2
    registers::x86_64::id::mm3,     // MM3
    registers::x86_64::id::mm4,     // MM4
    registers::x86_64::id::mm5,     // MM5
    registers::x86_64::id::mm6,     // MM6
    registers::x86_64::id::mm7,     // MM7
    registers::x86_64::id::xmm0,    // XMM0
    registers::x86_64::id::xmm1,    // XMM1
    registers::x86_64::id::xmm2,    // XMM2
    registers::x86_64::id::xmm3,    // XMM3
    registers::x86_64::id::xmm4,    // XMM4
    registers::x86_64::id::xmm5,    // XMM5
    registers::x86_64::id::xmm6,    // XMM6
    registers::x86_64::id::xmm7,    // XMM7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM,
    registers::x86_64::id::mxcsr,   // MXCSR
    LLDB_INVALID_REGNUM, // EDXEAX
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // EMM0L
    LLDB_INVALID_REGNUM, // EMM1L
    LLDB_INVALID_REGNUM, // EMM2L
    LLDB_INVALID_REGNUM, // EMM3L
    LLDB_INVALID_REGNUM, // EMM4L
    LLDB_INVALID_REGNUM, // EMM5L
    LLDB_INVALID_REGNUM, // EMM6L
    LLDB_INVALID_REGNUM, // EMM7L
    LLDB_INVALID_REGNUM, // EMM0H
    LLDB_INVALID_REGNUM, // EMM1H
    LLDB_INVALID_REGNUM, // EMM2H
    LLDB_INVALID_REGNUM, // EMM3H
    LLDB_INVALID_REGNUM, // EMM4H
    LLDB_INVALID_REGNUM, // EMM5H
    LLDB_INVALID_REGNUM, // EMM6H
    LLDB_INVALID_REGNUM, // EMM7H
    LLDB_INVALID_REGNUM, // MM00
    LLDB_INVALID_REGNUM, // MM01
    LLDB_INVALID_REGNUM, // MM10
    LLDB_INVALID_REGNUM, // MM11
    LLDB_INVALID_REGNUM, // MM20
    LLDB_INVALID_REGNUM, // MM21
    LLDB_INVALID_REGNUM, // MM30
    LLDB_INVALID_REGNUM, // MM31
    LLDB_INVALID_REGNUM, // MM40
    LLDB_INVALID_REGNUM, // MM41
    LLDB_INVALID_REGNUM, // MM50
    LLDB_INVALID_REGNUM, // MM51
    LLDB_INVALID_REGNUM, // MM60
    LLDB_INVALID_REGNUM, // MM61
    LLDB_INVALID_REGNUM, // MM70
    LLDB_INVALID_REGNUM, // MM71
    registers::x86_64::id::xmm8,    // XMM8
    registers::x86_64::id::xmm9,    // XMM9
    registers::x86_64::id::xmm10,   // XMM10
    registers::x86_64::id::xmm11,   // XMM11
    registers::x86_64::id::xmm12,   // XMM12
    registers::x86_64::id::xmm13,   // XMM13
    registers::x86_64::id::xmm14,   // XMM14
    registers::x86_64::id::xmm15,   // XMM15
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM,
    registers::x86_64::id::sil,   // SIL
    registers::x86_64::id::dil,   // DIL
    registers::x86_64::id::bpl,   // BPL
    registers::x86_64::id::spl,   // SPL
    registers::x86_64::id::rax,   // RAX
    registers::x86_64::id::rbx,   // RBX
    registers::x86_64::id::rcx,   // RCX
    registers::x86_64::id::rdx,   // RDX
    registers::x86_64::id::rsi,   // RSI
    registers::x86_64::id::rdi,   // RDI
    registers::x86_64::id::rbp,   // RBP
    registers::x86_64::id::rsp,   // RSP
    registers::x86_64::id::r8,    // R8
    registers::x86_64::id::r9,    // R9
    registers::x86_64::id::r10,   // R10
    registers::x86_64::id::r11,   // R11
    registers::x86_64::id::r12,   // R12
    registers::x86_64::id::r13,   // R13
    registers::x86_64::id::r14,   // R14
    registers::x86_64::id::r15,   // R15
    registers::x86_64::id::r8l,   // R8B
    registers::x86_64::id::r9l,   // R9B
    registers::x86_64::id::r10l,  // R10B
    registers::x86_64::id::r11l,  // R11B
    registers::x86_64::id::r12l,  // R12B
    registers::x86_64::id::r13l,  // R13B
    registers::x86_64::id::r14l,  // R14B
    registers::x86_64::id::r15l,  // R15B
    registers::x86_64::id::r8w,   // R8W
    registers::x86_64::id::r9w,   // R9W
    registers::x86_64::id::r10w,  // R10W
    registers::x86_64::id::r11w,  // R11W
    registers::x86_64::id::r12w,  // R12W
    registers::x86_64::id::r13w,  // R13W
    registers::x86_64::id::r14w,  // R14W
    registers::x86_64::id::r15w,  // R15W
    registers::x86_64::id::r8d,   // R8D
    registers::x86_64::id::r9d,   // R9D
    registers::x86_64::id::r10d,  // R10D
    registers::x86_64::id::r11d,  // R11D
    registers::x86_64::id::r12d,  // R12D
    registers::x86_64::id::r13d,  // R13D
    registers::x86_64::id::r14d,  // R14D
    registers::x86_64::id::r15d,  // R15D
    registers::x86_64::id::ymm0,  // AMD64_YMM0
    registers::x86_64::id::ymm1,  // AMD64_YMM1
    registers::x86_64::id::ymm2,  // AMD64_YMM2
    registers::x86_64::id::ymm3,  // AMD64_YMM3
    registers::x86_64::id::ymm4,  // AMD64_YMM4
    registers::x86_64::id::ymm5,  // AMD64_YMM5
    registers::x86_64::id::ymm6,  // AMD64_YMM6
    registers::x86_64::id::ymm7,  // AMD64_YMM7
    registers::x86_64::id::ymm8,  // AMD64_YMM8
    registers::x86_64::id::ymm9,  // AMD64_YMM9
    registers::x86_64::id::ymm10, // AMD64_YMM10
    registers::x86_64::id::ymm11, // AMD64_YMM11
    registers::x86_64::id::ymm12, // AMD64_YMM12
    registers::x86_64::id::ymm13, // AMD64_YMM13
    registers::x86_64::id::ymm14, // AMD64_YMM14
    registers::x86_64::id::ymm15, // AMD64_YMM15
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    registers::x86_64::id::bnd0, // BND0
    registers::x86_64::id::bnd1, // BND1
    registers::x86_64::id::bnd2  // BND2
};

uint32_t lldb_private::npdb::GetLLDBRegisterNumber(
    llvm::Triple::ArchType arch_type, llvm::codeview::RegisterId register_id) {
  switch (arch_type) {
  case llvm::Triple::aarch64:
    if (static_cast<uint16_t>(register_id) <
        sizeof(g_code_view_to_lldb_registers_arm64) /
            sizeof(g_code_view_to_lldb_registers_arm64[0]))
      return g_code_view_to_lldb_registers_arm64[static_cast<uint16_t>(
          register_id)];

    return LLDB_INVALID_REGNUM;
  case llvm::Triple::x86:
    if (static_cast<uint16_t>(register_id) <
        sizeof(g_code_view_to_lldb_registers_x86) /
            sizeof(g_code_view_to_lldb_registers_x86[0]))
      return g_code_view_to_lldb_registers_x86[static_cast<uint16_t>(
          register_id)];

    switch (register_id) {
    case llvm::codeview::RegisterId::MXCSR:
      return registers::x86::id::mxcsr;
    case llvm::codeview::RegisterId::BND0:
      return registers::x86::id::bnd0;
    case llvm::codeview::RegisterId::BND1:
      return registers::x86::id::bnd1;
    case llvm::codeview::RegisterId::BND2:
      return registers::x86::id::bnd2;
    default:
      return LLDB_INVALID_REGNUM;
    }
  case llvm::Triple::x86_64:
    if (static_cast<uint16_t>(register_id) <
        sizeof(g_code_view_to_lldb_registers_x86_64) /
            sizeof(g_code_view_to_lldb_registers_x86_64[0]))
      return g_code_view_to_lldb_registers_x86_64[static_cast<uint16_t>(
          register_id)];

    return LLDB_INVALID_REGNUM;
  default:
    return LLDB_INVALID_REGNUM;
  }
}

uint32_t
lldb_private::npdb::GetRegisterSize(llvm::codeview::RegisterId register_id) {
  switch(register_id) {
    case llvm::codeview::RegisterId::AL:
    case llvm::codeview::RegisterId::BL:
    case llvm::codeview::RegisterId::CL:
    case llvm::codeview::RegisterId::DL:
    case llvm::codeview::RegisterId::AH:
    case llvm::codeview::RegisterId::BH:
    case llvm::codeview::RegisterId::CH:
    case llvm::codeview::RegisterId::DH:
    case llvm::codeview::RegisterId::SIL:
    case llvm::codeview::RegisterId::DIL:
    case llvm::codeview::RegisterId::BPL:
    case llvm::codeview::RegisterId::SPL:
    case llvm::codeview::RegisterId::R8B:
    case llvm::codeview::RegisterId::R9B:
    case llvm::codeview::RegisterId::R10B:
    case llvm::codeview::RegisterId::R11B:
    case llvm::codeview::RegisterId::R12B:
    case llvm::codeview::RegisterId::R13B:
    case llvm::codeview::RegisterId::R14B:
    case llvm::codeview::RegisterId::R15B:
      return 1;
    case llvm::codeview::RegisterId::AX:
    case llvm::codeview::RegisterId::BX:
    case llvm::codeview::RegisterId::CX:
    case llvm::codeview::RegisterId::DX:
    case llvm::codeview::RegisterId::SP:
    case llvm::codeview::RegisterId::BP:
    case llvm::codeview::RegisterId::SI:
    case llvm::codeview::RegisterId::DI:
    case llvm::codeview::RegisterId::R8W:
    case llvm::codeview::RegisterId::R9W:
    case llvm::codeview::RegisterId::R10W:
    case llvm::codeview::RegisterId::R11W:
    case llvm::codeview::RegisterId::R12W:
    case llvm::codeview::RegisterId::R13W:
    case llvm::codeview::RegisterId::R14W:
    case llvm::codeview::RegisterId::R15W:
      return 2;
    case llvm::codeview::RegisterId::EAX:
    case llvm::codeview::RegisterId::EBX:
    case llvm::codeview::RegisterId::ECX:
    case llvm::codeview::RegisterId::EDX:
    case llvm::codeview::RegisterId::ESP:
    case llvm::codeview::RegisterId::EBP:
    case llvm::codeview::RegisterId::ESI:
    case llvm::codeview::RegisterId::EDI:
    case llvm::codeview::RegisterId::R8D:
    case llvm::codeview::RegisterId::R9D:
    case llvm::codeview::RegisterId::R10D:
    case llvm::codeview::RegisterId::R11D:
    case llvm::codeview::RegisterId::R12D:
    case llvm::codeview::RegisterId::R13D:
    case llvm::codeview::RegisterId::R14D:
    case llvm::codeview::RegisterId::R15D:
      return 4;
    case llvm::codeview::RegisterId::RAX:
    case llvm::codeview::RegisterId::RBX:
    case llvm::codeview::RegisterId::RCX:
    case llvm::codeview::RegisterId::RDX:
    case llvm::codeview::RegisterId::RSI:
    case llvm::codeview::RegisterId::RDI:
    case llvm::codeview::RegisterId::RBP:
    case llvm::codeview::RegisterId::RSP:
    case llvm::codeview::RegisterId::R8:
    case llvm::codeview::RegisterId::R9:
    case llvm::codeview::RegisterId::R10:
    case llvm::codeview::RegisterId::R11:
    case llvm::codeview::RegisterId::R12:
    case llvm::codeview::RegisterId::R13:
    case llvm::codeview::RegisterId::R14:
    case llvm::codeview::RegisterId::R15:
      return 8;
    case llvm::codeview::RegisterId::XMM0:
    case llvm::codeview::RegisterId::XMM1:
    case llvm::codeview::RegisterId::XMM2:
    case llvm::codeview::RegisterId::XMM3:
    case llvm::codeview::RegisterId::XMM4:
    case llvm::codeview::RegisterId::XMM5:
    case llvm::codeview::RegisterId::XMM6:
    case llvm::codeview::RegisterId::XMM7:
    case llvm::codeview::RegisterId::XMM8:
    case llvm::codeview::RegisterId::XMM9:
    case llvm::codeview::RegisterId::XMM10:
    case llvm::codeview::RegisterId::XMM11:
    case llvm::codeview::RegisterId::XMM12:
    case llvm::codeview::RegisterId::XMM13:
    case llvm::codeview::RegisterId::XMM14:
    case llvm::codeview::RegisterId::XMM15:
      return 16;
    default:
      return 0;
  }
}
