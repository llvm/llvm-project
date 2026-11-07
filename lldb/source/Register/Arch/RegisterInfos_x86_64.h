//===-- RegisterInfos_x86_64.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Textual register-table fragment; intentionally has no include guard.
// Include its ordinary dependencies at file scope before defining layout macros.
// See README.md for the inclusion contract.

#if !defined(DECLARE_REGISTER_INFOS_X86_64_STRUCT) && \
    !defined(UPDATE_REGISTER_INFOS_I386_STRUCT_WITH_X86_64_OFFSETS)
#error Select a register-table generation mode before including this fragment
#endif

#ifndef DR_OFFSET
#error DR_OFFSET must be defined before including this fragment
#endif

// Computes the offset of the given GPR in the user data area.
#define GPR_OFFSET(regname) (LLVM_EXTENSION offsetof(GPR, regname))

// Computes the offset of the given FPR in the extended data area.
#define FPR_OFFSET(regname)                                                    \
  (LLVM_EXTENSION offsetof(UserArea, fpr) +                                    \
   LLVM_EXTENSION offsetof(FPR, fxsave) +                                      \
   LLVM_EXTENSION offsetof(FXSAVE, regname))

// Computes the offset of the YMM register assembled from register halves.
// Based on DNBArchImplX86_64.cpp from debugserver
#define YMM_OFFSET(reg_index)                                                  \
  (LLVM_EXTENSION offsetof(UserArea, fpr) +                                    \
   LLVM_EXTENSION offsetof(FPR, xsave) +                                       \
   LLVM_EXTENSION offsetof(XSAVE, ymmh[0]) + (32 * reg_index))

// Guarantees BNDR/BNDC offsets do not overlap with YMM offsets.
#define GDB_REMOTE_OFFSET 128

#define BNDR_OFFSET(reg_index)                                                 \
  (LLVM_EXTENSION offsetof(UserArea, fpr) +                                    \
   LLVM_EXTENSION offsetof(FPR, xsave) +                                       \
   LLVM_EXTENSION offsetof(XSAVE, mpxr[reg_index]) + GDB_REMOTE_OFFSET)

#define BNDC_OFFSET(reg_index)                                                 \
  (LLVM_EXTENSION offsetof(UserArea, fpr) +                                    \
   LLVM_EXTENSION offsetof(FPR, xsave) +                                       \
   LLVM_EXTENSION offsetof(XSAVE, mpxc[reg_index]) + GDB_REMOTE_OFFSET)

#ifdef DECLARE_REGISTER_INFOS_X86_64_STRUCT

// Number of bytes needed to represent a FPR.
#define FPR_SIZE(reg) sizeof(((FXSAVE *)nullptr)->reg)

// Number of bytes needed to represent the i'th FP register.
#define FP_SIZE sizeof(((MMSReg *)nullptr)->bytes)

// Number of bytes needed to represent an XMM register.
#define XMM_SIZE sizeof(XMMReg)

// Number of bytes needed to represent a YMM register.
#define YMM_SIZE sizeof(YMMReg)

// Number of bytes needed to represent MPX registers.
#define BNDR_SIZE sizeof(MPXReg)
#define BNDC_SIZE sizeof(MPXCsr)

#define DR_SIZE sizeof(((DBG *)nullptr)->dr[0])

// RegisterKind: EHFrame, DWARF, Generic, Process Plugin, LLDB

// Note that the size and offset will be updated by platform-specific classes.
#define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)                       \
  {                                                                            \
    #reg, alt, sizeof(((GPR *)nullptr)->reg),                                  \
                      GPR_OFFSET(reg), lldb::eEncodingUint, lldb::eFormatHex,              \
                                 {kind1, kind2, kind3, kind4,                  \
                                  lldb_private::registers::x86_64::id::reg },                       \
                                  nullptr, nullptr, nullptr,                   \
  }

#define DEFINE_FPR(name, reg, kind1, kind2, kind3, kind4)                      \
  {                                                                            \
    #name, nullptr, FPR_SIZE(reg), FPR_OFFSET(reg), lldb::eEncodingUint, lldb::eFormatHex, \
                                           {kind1, kind2, kind3, kind4,        \
                                            lldb_private::registers::x86_64::id::name },            \
                                            nullptr, nullptr, nullptr,         \
  }

#define DEFINE_FP_ST(reg, i)                                                   \
  {                                                                            \
    #reg #i, nullptr, FP_SIZE,                                                 \
        LLVM_EXTENSION FPR_OFFSET(                                             \
            stmm[i]), lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,                   \
            {lldb_private::registers::x86_64::ehframe::st##i, lldb_private::registers::x86_64::dwarf::st##i, LLDB_INVALID_REGNUM,  \
             LLDB_INVALID_REGNUM, lldb_private::registers::x86_64::id::st##i },                       \
             nullptr, nullptr, nullptr,                                        \
  }

#define DEFINE_FP_MM(reg, i, streg)                                            \
  {                                                                            \
    #reg #i, nullptr, sizeof(uint64_t), LLVM_EXTENSION FPR_OFFSET(stmm[i]),    \
    lldb::eEncodingUint, lldb::eFormatHex,                                                 \
    {lldb_private::registers::x86_64::ehframe::mm##i, lldb_private::registers::x86_64::dwarf::mm##i, LLDB_INVALID_REGNUM,          \
     LLDB_INVALID_REGNUM, lldb_private::registers::x86_64::id::mm##i },                               \
    x86_register_info::g_contained_##streg##_64,                               \
    x86_register_info::g_invalidate_##streg##_64,                              \
    nullptr,                                                                   \
  }

#define DEFINE_XMM(reg, i)                                                     \
  {                                                                            \
    #reg #i, nullptr, XMM_SIZE,                                                \
        LLVM_EXTENSION FPR_OFFSET(                                             \
            reg[i]), lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,                    \
            {lldb_private::registers::x86_64::ehframe::reg##i, lldb_private::registers::x86_64::dwarf::reg##i,                 \
             LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                         \
             lldb_private::registers::x86_64::id::reg##i },                                         \
             nullptr, nullptr, nullptr,                                        \
  }

#define DEFINE_YMM(reg, i)                                                     \
  {                                                                            \
    #reg #i, nullptr, YMM_SIZE,                                                \
        LLVM_EXTENSION YMM_OFFSET(i), lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,   \
                                  {lldb_private::registers::x86_64::ehframe::reg##i##h,                   \
                                   lldb_private::registers::x86_64::dwarf::reg##i##h,                   \
                                   LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,   \
                                   lldb_private::registers::x86_64::id::reg##i },                   \
                                   nullptr, nullptr, nullptr,                  \
  }

#define DEFINE_BNDR(reg, i)                                                    \
  {                                                                            \
    #reg #i, nullptr, BNDR_SIZE,                                               \
        LLVM_EXTENSION BNDR_OFFSET(i), lldb::eEncodingVector, lldb::eFormatVectorOfUInt64, \
                                   {lldb_private::registers::x86_64::ehframe::reg##i,                   \
                                    lldb_private::registers::x86_64::dwarf::reg##i,                   \
                                    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,  \
                                    lldb_private::registers::x86_64::id::reg##i },                  \
                                    nullptr, nullptr, nullptr,                 \
  }

#define DEFINE_BNDC(name, i)                                                   \
  {                                                                            \
    #name, nullptr, BNDC_SIZE,                                                 \
        LLVM_EXTENSION BNDC_OFFSET(i), lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  \
        {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,        \
         LLDB_INVALID_REGNUM, lldb_private::registers::x86_64::id::name },                          \
         nullptr, nullptr, nullptr,                                            \
  }

#define DEFINE_DR(reg, i)                                                      \
  {                                                                            \
    #reg #i, nullptr, DR_SIZE,                                                 \
        DR_OFFSET(i), lldb::eEncodingUint, lldb::eFormatHex,                               \
                  {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                   \
                   LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                   \
                   lldb_private::registers::x86_64::id::reg##i },                                   \
                   nullptr, nullptr, nullptr,                                  \
  }

#define DEFINE_GPR_PSEUDO_32(reg32, reg64)                                     \
  {                                                                            \
    #reg32, nullptr, 4,                                                        \
        GPR_OFFSET(reg64), lldb::eEncodingUint, lldb::eFormatHex,                          \
                   {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                  \
                    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                  \
                    lldb_private::registers::x86_64::id::reg32 },                                   \
                    x86_register_info::g_contained_##reg64,                    \
                    x86_register_info::g_invalidate_##reg64,                   \
                    nullptr,                                                   \
  }

#define DEFINE_GPR_PSEUDO_16(reg16, reg64)                                     \
  {                                                                            \
    #reg16, nullptr, 2,                                                        \
        GPR_OFFSET(reg64), lldb::eEncodingUint, lldb::eFormatHex,                          \
                   {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                  \
                    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                  \
                    lldb_private::registers::x86_64::id::reg16 },                                   \
                    x86_register_info::g_contained_##reg64,                    \
                    x86_register_info::g_invalidate_##reg64,                   \
                    nullptr,                                                   \
  }

#define DEFINE_GPR_PSEUDO_8H(reg8, reg64)                                      \
  {                                                                            \
    #reg8, nullptr, 1,                                                         \
        GPR_OFFSET(reg64) + 1, lldb::eEncodingUint, lldb::eFormatHex,                      \
                   {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                  \
                    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                  \
                    lldb_private::registers::x86_64::id::reg8 },                                    \
                    x86_register_info::g_contained_##reg64,                    \
                    x86_register_info::g_invalidate_##reg64,                   \
                    nullptr,                                                   \
  }

#define DEFINE_GPR_PSEUDO_8L(reg8, reg64)                                      \
  {                                                                            \
    #reg8, nullptr, 1,                                                         \
        GPR_OFFSET(reg64), lldb::eEncodingUint, lldb::eFormatHex,                          \
                   {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                  \
                    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                  \
                    lldb_private::registers::x86_64::id::reg8 },                                    \
                    x86_register_info::g_contained_##reg64,                    \
                    x86_register_info::g_invalidate_##reg64,                   \
                    nullptr                                                    \
  }

#define DEFINE_FPR_32(name, reg, kind1, kind2, kind3, kind4, reg64)            \
  {                                                                            \
    #name, nullptr, FPR_SIZE(reg), FPR_OFFSET(reg), lldb::eEncodingUint, lldb::eFormatHex, \
    {kind1, kind2, kind3, kind4, lldb_private::registers::x86_64::id::name },                       \
    x86_register_info::g_contained_##reg64,                                    \
    x86_register_info::g_invalidate_##reg64,                                   \
    nullptr,                                                                   \
  }

// clang-format off
static RegisterInfo g_register_infos_x86_64[] = {
// General purpose registers     EH_Frame              DWARF                 Generic                     Process Plugin
//  ===========================  ==================    ================      =========================   ====================
    DEFINE_GPR(rax,    nullptr,  lldb_private::registers::x86_64::ehframe::rax,     lldb_private::registers::x86_64::dwarf::rax,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(rbx,    nullptr,  lldb_private::registers::x86_64::ehframe::rbx,     lldb_private::registers::x86_64::dwarf::rbx,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(rcx,    nullptr,  lldb_private::registers::x86_64::ehframe::rcx,     lldb_private::registers::x86_64::dwarf::rcx,     LLDB_REGNUM_GENERIC_ARG4,   LLDB_INVALID_REGNUM),
    DEFINE_GPR(rdx,    nullptr,  lldb_private::registers::x86_64::ehframe::rdx,     lldb_private::registers::x86_64::dwarf::rdx,     LLDB_REGNUM_GENERIC_ARG3,   LLDB_INVALID_REGNUM),
    DEFINE_GPR(rdi,    nullptr,  lldb_private::registers::x86_64::ehframe::rdi,     lldb_private::registers::x86_64::dwarf::rdi,     LLDB_REGNUM_GENERIC_ARG1,   LLDB_INVALID_REGNUM),
    DEFINE_GPR(rsi,    nullptr,  lldb_private::registers::x86_64::ehframe::rsi,     lldb_private::registers::x86_64::dwarf::rsi,     LLDB_REGNUM_GENERIC_ARG2,   LLDB_INVALID_REGNUM),
    DEFINE_GPR(rbp,    nullptr,  lldb_private::registers::x86_64::ehframe::rbp,     lldb_private::registers::x86_64::dwarf::rbp,     LLDB_REGNUM_GENERIC_FP,     LLDB_INVALID_REGNUM),
    DEFINE_GPR(rsp,    nullptr,  lldb_private::registers::x86_64::ehframe::rsp,     lldb_private::registers::x86_64::dwarf::rsp,     LLDB_REGNUM_GENERIC_SP,     LLDB_INVALID_REGNUM),
    DEFINE_GPR(r8,     nullptr,  lldb_private::registers::x86_64::ehframe::r8,      lldb_private::registers::x86_64::dwarf::r8,      LLDB_REGNUM_GENERIC_ARG5,   LLDB_INVALID_REGNUM),
    DEFINE_GPR(r9,     nullptr,  lldb_private::registers::x86_64::ehframe::r9,      lldb_private::registers::x86_64::dwarf::r9,      LLDB_REGNUM_GENERIC_ARG6,   LLDB_INVALID_REGNUM),
    DEFINE_GPR(r10,    nullptr,  lldb_private::registers::x86_64::ehframe::r10,     lldb_private::registers::x86_64::dwarf::r10,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(r11,    nullptr,  lldb_private::registers::x86_64::ehframe::r11,     lldb_private::registers::x86_64::dwarf::r11,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(r12,    nullptr,  lldb_private::registers::x86_64::ehframe::r12,     lldb_private::registers::x86_64::dwarf::r12,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(r13,    nullptr,  lldb_private::registers::x86_64::ehframe::r13,     lldb_private::registers::x86_64::dwarf::r13,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(r14,    nullptr,  lldb_private::registers::x86_64::ehframe::r14,     lldb_private::registers::x86_64::dwarf::r14,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(r15,    nullptr,  lldb_private::registers::x86_64::ehframe::r15,     lldb_private::registers::x86_64::dwarf::r15,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(rip,    nullptr,  lldb_private::registers::x86_64::ehframe::rip,     lldb_private::registers::x86_64::dwarf::rip,     LLDB_REGNUM_GENERIC_PC,     LLDB_INVALID_REGNUM),
    DEFINE_GPR(rflags, nullptr,  lldb_private::registers::x86_64::ehframe::rflags,  lldb_private::registers::x86_64::dwarf::rflags,  LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM),
    DEFINE_GPR(cs,     nullptr,  lldb_private::registers::x86_64::ehframe::cs,      lldb_private::registers::x86_64::dwarf::cs,      LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(fs,     nullptr,  lldb_private::registers::x86_64::ehframe::fs,      lldb_private::registers::x86_64::dwarf::fs,      LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(gs,     nullptr,  lldb_private::registers::x86_64::ehframe::gs,      lldb_private::registers::x86_64::dwarf::gs,      LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(ss,     nullptr,  lldb_private::registers::x86_64::ehframe::ss,      lldb_private::registers::x86_64::dwarf::ss,      LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(ds,     nullptr,  lldb_private::registers::x86_64::ehframe::ds,      lldb_private::registers::x86_64::dwarf::ds,      LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(es,     nullptr,  lldb_private::registers::x86_64::ehframe::es,      lldb_private::registers::x86_64::dwarf::es,      LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),

    DEFINE_GPR_PSEUDO_32(eax, rax), DEFINE_GPR_PSEUDO_32(ebx, rbx),
    DEFINE_GPR_PSEUDO_32(ecx, rcx), DEFINE_GPR_PSEUDO_32(edx, rdx),
    DEFINE_GPR_PSEUDO_32(edi, rdi), DEFINE_GPR_PSEUDO_32(esi, rsi),
    DEFINE_GPR_PSEUDO_32(ebp, rbp), DEFINE_GPR_PSEUDO_32(esp, rsp),
    DEFINE_GPR_PSEUDO_32(r8d, r8), DEFINE_GPR_PSEUDO_32(r9d, r9),
    DEFINE_GPR_PSEUDO_32(r10d, r10), DEFINE_GPR_PSEUDO_32(r11d, r11),
    DEFINE_GPR_PSEUDO_32(r12d, r12), DEFINE_GPR_PSEUDO_32(r13d, r13),
    DEFINE_GPR_PSEUDO_32(r14d, r14), DEFINE_GPR_PSEUDO_32(r15d, r15),
    DEFINE_GPR_PSEUDO_16(ax, rax), DEFINE_GPR_PSEUDO_16(bx, rbx),
    DEFINE_GPR_PSEUDO_16(cx, rcx), DEFINE_GPR_PSEUDO_16(dx, rdx),
    DEFINE_GPR_PSEUDO_16(di, rdi), DEFINE_GPR_PSEUDO_16(si, rsi),
    DEFINE_GPR_PSEUDO_16(bp, rbp), DEFINE_GPR_PSEUDO_16(sp, rsp),
    DEFINE_GPR_PSEUDO_16(r8w, r8), DEFINE_GPR_PSEUDO_16(r9w, r9),
    DEFINE_GPR_PSEUDO_16(r10w, r10), DEFINE_GPR_PSEUDO_16(r11w, r11),
    DEFINE_GPR_PSEUDO_16(r12w, r12), DEFINE_GPR_PSEUDO_16(r13w, r13),
    DEFINE_GPR_PSEUDO_16(r14w, r14), DEFINE_GPR_PSEUDO_16(r15w, r15),
    DEFINE_GPR_PSEUDO_8H(ah, rax), DEFINE_GPR_PSEUDO_8H(bh, rbx),
    DEFINE_GPR_PSEUDO_8H(ch, rcx), DEFINE_GPR_PSEUDO_8H(dh, rdx),
    DEFINE_GPR_PSEUDO_8L(al, rax), DEFINE_GPR_PSEUDO_8L(bl, rbx),
    DEFINE_GPR_PSEUDO_8L(cl, rcx), DEFINE_GPR_PSEUDO_8L(dl, rdx),
    DEFINE_GPR_PSEUDO_8L(dil, rdi), DEFINE_GPR_PSEUDO_8L(sil, rsi),
    DEFINE_GPR_PSEUDO_8L(bpl, rbp), DEFINE_GPR_PSEUDO_8L(spl, rsp),
    DEFINE_GPR_PSEUDO_8L(r8l, r8), DEFINE_GPR_PSEUDO_8L(r9l, r9),
    DEFINE_GPR_PSEUDO_8L(r10l, r10), DEFINE_GPR_PSEUDO_8L(r11l, r11),
    DEFINE_GPR_PSEUDO_8L(r12l, r12), DEFINE_GPR_PSEUDO_8L(r13l, r13),
    DEFINE_GPR_PSEUDO_8L(r14l, r14), DEFINE_GPR_PSEUDO_8L(r15l, r15),

//  i387 Floating point registers.         EH_frame             DWARF                Generic              Process Plugin       reg64
//  ====================================== ===============      ==================   ===================  ==================== =====
    DEFINE_FPR(fctrl,     fctrl,           lldb_private::registers::x86_64::ehframe::fctrl,  lldb_private::registers::x86_64::dwarf::fctrl,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(fstat,     fstat,           lldb_private::registers::x86_64::ehframe::fstat,  lldb_private::registers::x86_64::dwarf::fstat,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(ftag,      ftag,            LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(fop,       fop,             LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR_32(fiseg,  ptr.i386_.fiseg, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fip),
    DEFINE_FPR_32(fioff,  ptr.i386_.fioff, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fip),
    DEFINE_FPR(fip,       ptr.x86_64.fip,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR_32(foseg,  ptr.i386_.foseg, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fdp),
    DEFINE_FPR_32(fooff,  ptr.i386_.fooff, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fdp),
    DEFINE_FPR(fdp,       ptr.x86_64.fdp,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(mxcsr,     mxcsr,           lldb_private::registers::x86_64::ehframe::mxcsr,  lldb_private::registers::x86_64::dwarf::mxcsr,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(mxcsrmask, mxcsrmask,       LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),

    // FP registers.
    DEFINE_FP_ST(st, 0), DEFINE_FP_ST(st, 1), DEFINE_FP_ST(st, 2),
    DEFINE_FP_ST(st, 3), DEFINE_FP_ST(st, 4), DEFINE_FP_ST(st, 5),
    DEFINE_FP_ST(st, 6), DEFINE_FP_ST(st, 7),

    DEFINE_FP_MM(mm, 0, st0), DEFINE_FP_MM(mm, 1, st1),
    DEFINE_FP_MM(mm, 2, st2), DEFINE_FP_MM(mm, 3, st3),
    DEFINE_FP_MM(mm, 4, st4), DEFINE_FP_MM(mm, 5, st5),
    DEFINE_FP_MM(mm, 6, st6), DEFINE_FP_MM(mm, 7, st7),

    // XMM registers
    DEFINE_XMM(xmm, 0), DEFINE_XMM(xmm, 1), DEFINE_XMM(xmm, 2),
    DEFINE_XMM(xmm, 3), DEFINE_XMM(xmm, 4), DEFINE_XMM(xmm, 5),
    DEFINE_XMM(xmm, 6), DEFINE_XMM(xmm, 7), DEFINE_XMM(xmm, 8),
    DEFINE_XMM(xmm, 9), DEFINE_XMM(xmm, 10), DEFINE_XMM(xmm, 11),
    DEFINE_XMM(xmm, 12), DEFINE_XMM(xmm, 13), DEFINE_XMM(xmm, 14),
    DEFINE_XMM(xmm, 15),

    // Copy of YMM registers assembled from xmm and ymmh
    DEFINE_YMM(ymm, 0), DEFINE_YMM(ymm, 1), DEFINE_YMM(ymm, 2),
    DEFINE_YMM(ymm, 3), DEFINE_YMM(ymm, 4), DEFINE_YMM(ymm, 5),
    DEFINE_YMM(ymm, 6), DEFINE_YMM(ymm, 7), DEFINE_YMM(ymm, 8),
    DEFINE_YMM(ymm, 9), DEFINE_YMM(ymm, 10), DEFINE_YMM(ymm, 11),
    DEFINE_YMM(ymm, 12), DEFINE_YMM(ymm, 13), DEFINE_YMM(ymm, 14),
    DEFINE_YMM(ymm, 15),

    // MPX registers
    DEFINE_BNDR(bnd, 0),
    DEFINE_BNDR(bnd, 1),
    DEFINE_BNDR(bnd, 2),
    DEFINE_BNDR(bnd, 3),

    DEFINE_BNDC(bndcfgu, 0),
    DEFINE_BNDC(bndstatus, 1),

    // Debug registers for lldb internal use
    DEFINE_DR(dr, 0), DEFINE_DR(dr, 1), DEFINE_DR(dr, 2), DEFINE_DR(dr, 3),
    DEFINE_DR(dr, 4), DEFINE_DR(dr, 5), DEFINE_DR(dr, 6), DEFINE_DR(dr, 7)};

// clang-format on

static_assert((sizeof(g_register_infos_x86_64) /
               sizeof(g_register_infos_x86_64[0])) ==
                  lldb_private::registers::x86_64::id::k_num_registers,
              "g_register_infos_x86_64 has wrong number of register infos");

#endif // DECLARE_REGISTER_INFOS_X86_64_STRUCT

#ifdef UPDATE_REGISTER_INFOS_I386_STRUCT_WITH_X86_64_OFFSETS

#define UPDATE_GPR_INFO(reg, reg64)                                            \
  do {                                                                         \
    g_register_infos[lldb_private::registers::x86::id::reg].byte_offset =      \
        GPR_OFFSET(reg64);                                                     \
  } while (false);

#define UPDATE_GPR_INFO_8H(reg, reg64)                                         \
  do {                                                                         \
    g_register_infos[lldb_private::registers::x86::id::reg].byte_offset =      \
        GPR_OFFSET(reg64) + 1;                                                 \
  } while (false);

#define UPDATE_FPR_INFO(reg, reg64)                                            \
  do {                                                                         \
    g_register_infos[lldb_private::registers::x86::id::reg].byte_offset =      \
        FPR_OFFSET(reg64);                                                     \
  } while (false);

#define UPDATE_FP_INFO(reg, i)                                                 \
  do {                                                                         \
    g_register_infos[lldb_private::registers::x86::id::reg##i].byte_offset =   \
        FPR_OFFSET(stmm[i]);                                                   \
  } while (false);

#define UPDATE_XMM_INFO(reg, i)                                                \
  do {                                                                         \
    g_register_infos[lldb_private::registers::x86::id::reg##i].byte_offset =   \
        FPR_OFFSET(reg[i]);                                                    \
  } while (false);

#define UPDATE_YMM_INFO(reg, i)                                                \
  do {                                                                         \
    g_register_infos[lldb_private::registers::x86::id::reg##i].byte_offset =   \
        YMM_OFFSET(i);                                                         \
  } while (false);

#define UPDATE_DR_INFO(reg_index)                                              \
  do {                                                                         \
    g_register_infos[lldb_private::registers::x86::id::dr##reg_index].byte_offset =                  \
        DR_OFFSET(reg_index);                                                  \
  } while (false);

// Update the register offsets
UPDATE_GPR_INFO(eax, rax);
UPDATE_GPR_INFO(ebx, rbx);
UPDATE_GPR_INFO(ecx, rcx);
UPDATE_GPR_INFO(edx, rdx);
UPDATE_GPR_INFO(edi, rdi);
UPDATE_GPR_INFO(esi, rsi);
UPDATE_GPR_INFO(ebp, rbp);
UPDATE_GPR_INFO(esp, rsp);
UPDATE_GPR_INFO(eip, rip);
UPDATE_GPR_INFO(eflags, rflags);
UPDATE_GPR_INFO(cs, cs);
UPDATE_GPR_INFO(fs, fs);
UPDATE_GPR_INFO(gs, gs);
UPDATE_GPR_INFO(ss, ss);
UPDATE_GPR_INFO(ds, ds);
UPDATE_GPR_INFO(es, es);

UPDATE_GPR_INFO(ax, rax);
UPDATE_GPR_INFO(bx, rbx);
UPDATE_GPR_INFO(cx, rcx);
UPDATE_GPR_INFO(dx, rdx);
UPDATE_GPR_INFO(di, rdi);
UPDATE_GPR_INFO(si, rsi);
UPDATE_GPR_INFO(bp, rbp);
UPDATE_GPR_INFO(sp, rsp);
UPDATE_GPR_INFO_8H(ah, rax);
UPDATE_GPR_INFO_8H(bh, rbx);
UPDATE_GPR_INFO_8H(ch, rcx);
UPDATE_GPR_INFO_8H(dh, rdx);
UPDATE_GPR_INFO(al, rax);
UPDATE_GPR_INFO(bl, rbx);
UPDATE_GPR_INFO(cl, rcx);
UPDATE_GPR_INFO(dl, rdx);

UPDATE_FPR_INFO(fctrl, fctrl);
UPDATE_FPR_INFO(fstat, fstat);
UPDATE_FPR_INFO(ftag, ftag);
UPDATE_FPR_INFO(fop, fop);
UPDATE_FPR_INFO(fiseg, ptr.i386_.fiseg);
UPDATE_FPR_INFO(fioff, ptr.i386_.fioff);
UPDATE_FPR_INFO(fooff, ptr.i386_.fooff);
UPDATE_FPR_INFO(foseg, ptr.i386_.foseg);
UPDATE_FPR_INFO(mxcsr, mxcsr);
UPDATE_FPR_INFO(mxcsrmask, mxcsrmask);

UPDATE_FP_INFO(st, 0);
UPDATE_FP_INFO(st, 1);
UPDATE_FP_INFO(st, 2);
UPDATE_FP_INFO(st, 3);
UPDATE_FP_INFO(st, 4);
UPDATE_FP_INFO(st, 5);
UPDATE_FP_INFO(st, 6);
UPDATE_FP_INFO(st, 7);
UPDATE_FP_INFO(mm, 0);
UPDATE_FP_INFO(mm, 1);
UPDATE_FP_INFO(mm, 2);
UPDATE_FP_INFO(mm, 3);
UPDATE_FP_INFO(mm, 4);
UPDATE_FP_INFO(mm, 5);
UPDATE_FP_INFO(mm, 6);
UPDATE_FP_INFO(mm, 7);

UPDATE_XMM_INFO(xmm, 0);
UPDATE_XMM_INFO(xmm, 1);
UPDATE_XMM_INFO(xmm, 2);
UPDATE_XMM_INFO(xmm, 3);
UPDATE_XMM_INFO(xmm, 4);
UPDATE_XMM_INFO(xmm, 5);
UPDATE_XMM_INFO(xmm, 6);
UPDATE_XMM_INFO(xmm, 7);

UPDATE_YMM_INFO(ymm, 0);
UPDATE_YMM_INFO(ymm, 1);
UPDATE_YMM_INFO(ymm, 2);
UPDATE_YMM_INFO(ymm, 3);
UPDATE_YMM_INFO(ymm, 4);
UPDATE_YMM_INFO(ymm, 5);
UPDATE_YMM_INFO(ymm, 6);
UPDATE_YMM_INFO(ymm, 7);

UPDATE_DR_INFO(0);
UPDATE_DR_INFO(1);
UPDATE_DR_INFO(2);
UPDATE_DR_INFO(3);
UPDATE_DR_INFO(4);
UPDATE_DR_INFO(5);
UPDATE_DR_INFO(6);
UPDATE_DR_INFO(7);

#endif // UPDATE_REGISTER_INFOS_I386_STRUCT_WITH_X86_64_OFFSETS

#undef GPR_OFFSET
#undef FPR_OFFSET
#undef YMM_OFFSET
#undef GDB_REMOTE_OFFSET
#undef BNDR_OFFSET
#undef BNDC_OFFSET
#undef FPR_SIZE
#undef FP_SIZE
#undef XMM_SIZE
#undef YMM_SIZE
#undef BNDR_SIZE
#undef BNDC_SIZE
#undef DR_SIZE
#undef DEFINE_GPR
#undef DEFINE_FPR
#undef DEFINE_FP_ST
#undef DEFINE_FP_MM
#undef DEFINE_XMM
#undef DEFINE_YMM
#undef DEFINE_BNDR
#undef DEFINE_BNDC
#undef DEFINE_DR
#undef DEFINE_GPR_PSEUDO_32
#undef DEFINE_GPR_PSEUDO_16
#undef DEFINE_GPR_PSEUDO_8H
#undef DEFINE_GPR_PSEUDO_8L
#undef DEFINE_FPR_32
#undef UPDATE_GPR_INFO
#undef UPDATE_GPR_INFO_8H
#undef UPDATE_FPR_INFO
#undef UPDATE_FP_INFO
#undef UPDATE_XMM_INFO
#undef UPDATE_YMM_INFO
#undef UPDATE_DR_INFO
