//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Private architecture-description generator. No native layout inputs.
#define DR_SIZE word_size
// Number of bytes needed to represent a FPR.
#if !defined(FPR_SIZE)
#define FPR_SIZE(reg) FPRValueByteSize(#reg)
#endif

// Number of bytes needed to represent the i'th FP register.
#define FP_SIZE 10

// Number of bytes needed to represent an XMM register.
#define XMM_SIZE 16

// Number of bytes needed to represent a YMM register.
#define YMM_SIZE 32

// Number of bytes needed to represent MPX registers.
#define BNDR_SIZE 16
#define BNDC_SIZE 8

// Value widths and aliases are independent of storage slots.
#define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)                       \
  RegisterDescriptor(lldb_private::registers::x86::id::reg, #reg, alt, word_size,                  \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {kind1, kind2, kind3, kind4, lldb_private::registers::x86::id::reg}, {}, {},  \
                     nullptr)

#define DEFINE_FPR(name, reg, kind1, kind2, kind3, kind4)                      \
  RegisterDescriptor(lldb_private::registers::x86::id::name, #name, nullptr, FPR_SIZE(reg),        \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {kind1, kind2, kind3, kind4, lldb_private::registers::x86::id::name}, {}, {}, \
                     nullptr)

// RegisterKind: EHFrame, DWARF, Generic, Process Plugin, LLDB

#define DEFINE_FP_ST(reg, i)                                                   \
  RegisterDescriptor(lldb_private::registers::x86::id::st##i, #reg #i, nullptr, FP_SIZE,             \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     {lldb_private::registers::x86::ehframe::st##i, lldb_private::registers::x86::dwarf::st##i,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86::id::st##i},                                      \
                     {}, {}, nullptr)

#define DEFINE_FP_MM(reg, i, streg)                                            \
  RegisterDescriptor(lldb_private::registers::x86::id::mm##i, #reg #i, nullptr, 8,                   \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {lldb_private::registers::x86::ehframe::mm##i, lldb_private::registers::x86::dwarf::mm##i,                  \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86::id::mm##i},                \
                     x86_register_info::g_contained_##streg##_32,              \
                     x86_register_info::g_invalidate_##streg##_32, nullptr)

#define DEFINE_XMM(reg, i)                                                     \
  RegisterDescriptor(lldb_private::registers::x86::id::reg##i, #reg #i, nullptr, XMM_SIZE,         \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     {lldb_private::registers::x86::ehframe::reg##i, lldb_private::registers::x86::dwarf::reg##i,          \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86::id::reg##i},                                   \
                     {}, {}, nullptr)

// I believe the YMM registers use dwarf_xmm_%_i386 register numbers and then
// differentiate based on register size.
#define DEFINE_YMM(reg, i)                                                     \
  RegisterDescriptor(lldb_private::registers::x86::id::reg##i, #reg #i, nullptr, YMM_SIZE,         \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     {LLDB_INVALID_REGNUM, lldb_private::registers::x86::dwarf::xmm##i,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86::id::reg##i},                                   \
                     {}, {}, nullptr)

#define DEFINE_BNDR(reg, i)                                                    \
  RegisterDescriptor(lldb_private::registers::x86::id::reg##i, #reg #i, nullptr, BNDR_SIZE,        \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt64,       \
                     {lldb_private::registers::x86::ehframe::reg##i, lldb_private::registers::x86::dwarf::reg##i,            \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86::id::reg##i},                                   \
                     {}, {}, nullptr)

#define DEFINE_BNDC(name, i)                                                   \
  RegisterDescriptor(lldb_private::registers::x86::id::name, #name, nullptr, BNDC_SIZE,            \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86::id::name},                                     \
                     {}, {}, nullptr)

#define DEFINE_DR(reg, i)                                                      \
  RegisterDescriptor(lldb_private::registers::x86::id::reg##i, #reg #i, nullptr, DR_SIZE,          \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86::id::reg##i},                                   \
                     {}, {}, nullptr)

#define DEFINE_GPR_PSEUDO_16(reg16, reg32)                                     \
  RegisterDescriptor(lldb_private::registers::x86::id::reg16, #reg16, nullptr, 2,                  \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86::id::reg16},                \
                     x86_register_info::g_contained_##reg32,                   \
                     x86_register_info::g_invalidate_##reg32, nullptr)

#define DEFINE_GPR_PSEUDO_8H(reg8, reg32)                                      \
  RegisterDescriptor(lldb_private::registers::x86::id::reg8, #reg8, nullptr, 1,                    \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86::id::reg8},                 \
                     x86_register_info::g_contained_##reg32,                   \
                     x86_register_info::g_invalidate_##reg32, nullptr)

#define DEFINE_GPR_PSEUDO_8L(reg8, reg32)                                      \
  RegisterDescriptor(lldb_private::registers::x86::id::reg8, #reg8, nullptr, 1,                    \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86::id::reg8},                 \
                     x86_register_info::g_contained_##reg32,                   \
                     x86_register_info::g_invalidate_##reg32, nullptr)

static const RegisterDescriptor g_register_infos_i386[] = {
    // General purpose registers.
    DEFINE_GPR(eax, nullptr, lldb_private::registers::x86::ehframe::eax, lldb_private::registers::x86::dwarf::eax,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(ebx, nullptr, lldb_private::registers::x86::ehframe::ebx, lldb_private::registers::x86::dwarf::ebx,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(ecx, nullptr, lldb_private::registers::x86::ehframe::ecx, lldb_private::registers::x86::dwarf::ecx,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(edx, nullptr, lldb_private::registers::x86::ehframe::edx, lldb_private::registers::x86::dwarf::edx,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(edi, nullptr, lldb_private::registers::x86::ehframe::edi, lldb_private::registers::x86::dwarf::edi,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(esi, nullptr, lldb_private::registers::x86::ehframe::esi, lldb_private::registers::x86::dwarf::esi,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(ebp, "fp", lldb_private::registers::x86::ehframe::ebp, lldb_private::registers::x86::dwarf::ebp,
               LLDB_REGNUM_GENERIC_FP, LLDB_INVALID_REGNUM),
    DEFINE_GPR(esp, "sp", lldb_private::registers::x86::ehframe::esp, lldb_private::registers::x86::dwarf::esp,
               LLDB_REGNUM_GENERIC_SP, LLDB_INVALID_REGNUM),
    DEFINE_GPR(eip, "pc", lldb_private::registers::x86::ehframe::eip, lldb_private::registers::x86::dwarf::eip,
               LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_REGNUM),
    DEFINE_GPR(eflags, "flags", lldb_private::registers::x86::ehframe::eflags, lldb_private::registers::x86::dwarf::eflags,
               LLDB_REGNUM_GENERIC_FLAGS, LLDB_INVALID_REGNUM),
    DEFINE_GPR(cs, nullptr, LLDB_INVALID_REGNUM, lldb_private::registers::x86::dwarf::cs,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(fs, nullptr, LLDB_INVALID_REGNUM, lldb_private::registers::x86::dwarf::fs,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(gs, nullptr, LLDB_INVALID_REGNUM, lldb_private::registers::x86::dwarf::gs,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(ss, nullptr, LLDB_INVALID_REGNUM, lldb_private::registers::x86::dwarf::ss,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(ds, nullptr, LLDB_INVALID_REGNUM, lldb_private::registers::x86::dwarf::ds,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(es, nullptr, LLDB_INVALID_REGNUM, lldb_private::registers::x86::dwarf::es,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),

    DEFINE_GPR_PSEUDO_16(ax, eax), DEFINE_GPR_PSEUDO_16(bx, ebx),
    DEFINE_GPR_PSEUDO_16(cx, ecx), DEFINE_GPR_PSEUDO_16(dx, edx),
    DEFINE_GPR_PSEUDO_16(di, edi), DEFINE_GPR_PSEUDO_16(si, esi),
    DEFINE_GPR_PSEUDO_16(bp, ebp), DEFINE_GPR_PSEUDO_16(sp, esp),
    DEFINE_GPR_PSEUDO_8H(ah, eax), DEFINE_GPR_PSEUDO_8H(bh, ebx),
    DEFINE_GPR_PSEUDO_8H(ch, ecx), DEFINE_GPR_PSEUDO_8H(dh, edx),
    DEFINE_GPR_PSEUDO_8L(al, eax), DEFINE_GPR_PSEUDO_8L(bl, ebx),
    DEFINE_GPR_PSEUDO_8L(cl, ecx), DEFINE_GPR_PSEUDO_8L(dl, edx),

    // i387 Floating point registers.
    DEFINE_FPR(fctrl, fctrl, LLDB_INVALID_REGNUM, lldb_private::registers::x86::dwarf::fctrl,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(fstat, fstat, LLDB_INVALID_REGNUM, lldb_private::registers::x86::dwarf::fstat,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(ftag, ftag, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(fop, fop, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(fiseg, ptr.i386_.fiseg, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(fioff, ptr.i386_.fioff, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(foseg, ptr.i386_.foseg, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(fooff, ptr.i386_.fooff, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(mxcsr, mxcsr, LLDB_INVALID_REGNUM, lldb_private::registers::x86::dwarf::mxcsr,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(mxcsrmask, mxcsrmask, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),

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
    DEFINE_XMM(xmm, 6), DEFINE_XMM(xmm, 7),

    // Copy of YMM registers assembled from xmm and ymmh
    DEFINE_YMM(ymm, 0), DEFINE_YMM(ymm, 1), DEFINE_YMM(ymm, 2),
    DEFINE_YMM(ymm, 3), DEFINE_YMM(ymm, 4), DEFINE_YMM(ymm, 5),
    DEFINE_YMM(ymm, 6), DEFINE_YMM(ymm, 7),

    // MPX registers
    DEFINE_BNDR(bnd, 0), DEFINE_BNDR(bnd, 1), DEFINE_BNDR(bnd, 2),
    DEFINE_BNDR(bnd, 3),

    DEFINE_BNDC(bndcfgu, 0), DEFINE_BNDC(bndstatus, 1),

    // Debug registers for lldb internal use
    DEFINE_DR(dr, 0), DEFINE_DR(dr, 1), DEFINE_DR(dr, 2), DEFINE_DR(dr, 3),
    DEFINE_DR(dr, 4), DEFINE_DR(dr, 5), DEFINE_DR(dr, 6), DEFINE_DR(dr, 7)};
#undef BNDC_SIZE
#undef BNDR_SIZE
#undef DEFINE_BNDC
#undef DEFINE_BNDR
#undef DEFINE_DR
#undef DEFINE_FPR
#undef DEFINE_FP_MM
#undef DEFINE_FP_ST
#undef DEFINE_GPR
#undef DEFINE_GPR_PSEUDO_16
#undef DEFINE_GPR_PSEUDO_8H
#undef DEFINE_GPR_PSEUDO_8L
#undef DEFINE_XMM
#undef DEFINE_YMM
#undef DR_SIZE
#undef FPR_SIZE
#undef FP_SIZE
#undef XMM_SIZE
#undef YMM_SIZE
