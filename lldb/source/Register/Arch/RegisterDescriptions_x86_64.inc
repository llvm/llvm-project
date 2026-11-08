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
#define FPR_SIZE(reg) FPRValueByteSize(#reg)

// Number of bytes needed to represent the i'th FP register.
#define FP_SIZE 10

// Number of bytes needed to represent an XMM register.
#define XMM_SIZE 16

// Number of bytes needed to represent a YMM register.
#define YMM_SIZE 32

// Number of bytes needed to represent MPX registers.
#define BNDR_SIZE 16
#define BNDC_SIZE 8

// RegisterKind: EHFrame, DWARF, Generic, Process Plugin, LLDB

// Value widths and aliases are independent of storage slots.
#define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)                       \
  RegisterDescriptor(lldb_private::registers::x86_64::id::reg, #reg, alt, word_size,                \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {kind1, kind2, kind3, kind4, lldb_private::registers::x86_64::id::reg}, {},    \
                     {}, nullptr)

#define DEFINE_FPR(name, reg, kind1, kind2, kind3, kind4)                      \
  RegisterDescriptor(lldb_private::registers::x86_64::id::name, #name, nullptr, FPR_SIZE(reg),      \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {kind1, kind2, kind3, kind4, lldb_private::registers::x86_64::id::name}, {},   \
                     {}, nullptr)

#define DEFINE_FP_ST(reg, i)                                                   \
  RegisterDescriptor(lldb_private::registers::x86_64::id::st##i, #reg #i, nullptr, FP_SIZE,           \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     {lldb_private::registers::x86_64::ehframe::st##i, lldb_private::registers::x86_64::dwarf::st##i,              \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86_64::id::st##i},                                    \
                     {}, {}, nullptr)

#define DEFINE_FP_MM(reg, i, streg)                                            \
  RegisterDescriptor(lldb_private::registers::x86_64::id::mm##i, #reg #i, nullptr, 8,                 \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {lldb_private::registers::x86_64::ehframe::mm##i, lldb_private::registers::x86_64::dwarf::mm##i,              \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86_64::id::mm##i},             \
                     x86_register_info::g_contained_##streg##_64,              \
                     x86_register_info::g_invalidate_##streg##_64, nullptr)

#define DEFINE_XMM(reg, i)                                                     \
  RegisterDescriptor(lldb_private::registers::x86_64::id::reg##i, #reg #i, nullptr, XMM_SIZE,       \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     {lldb_private::registers::x86_64::ehframe::reg##i, lldb_private::registers::x86_64::dwarf::reg##i,        \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86_64::id::reg##i},                                 \
                     {}, {}, nullptr)

#define DEFINE_YMM(reg, i)                                                     \
  RegisterDescriptor(lldb_private::registers::x86_64::id::reg##i, #reg #i, nullptr, YMM_SIZE,       \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     {lldb_private::registers::x86_64::ehframe::reg##i##h, lldb_private::registers::x86_64::dwarf::reg##i##h,      \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86_64::id::reg##i},                                 \
                     {}, {}, nullptr)

#define DEFINE_BNDR(reg, i)                                                    \
  RegisterDescriptor(lldb_private::registers::x86_64::id::reg##i, #reg #i, nullptr, BNDR_SIZE,      \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt64,       \
                     {lldb_private::registers::x86_64::ehframe::reg##i, lldb_private::registers::x86_64::dwarf::reg##i,        \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86_64::id::reg##i},                                 \
                     {}, {}, nullptr)

#define DEFINE_BNDC(name, i)                                                   \
  RegisterDescriptor(lldb_private::registers::x86_64::id::name, #name, nullptr, BNDC_SIZE,          \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86_64::id::name},                                   \
                     {}, {}, nullptr)

#define DEFINE_DR(reg, i)                                                      \
  RegisterDescriptor(lldb_private::registers::x86_64::id::reg##i, #reg #i, nullptr, DR_SIZE,        \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86_64::id::reg##i},                                 \
                     {}, {}, nullptr)

#define DEFINE_GPR_PSEUDO_32(reg32, reg64)                                     \
  RegisterDescriptor(lldb_private::registers::x86_64::id::reg32, #reg32, nullptr, 4,                \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86_64::id::reg32},             \
                     x86_register_info::g_contained_##reg64,                   \
                     x86_register_info::g_invalidate_##reg64, nullptr)

#define DEFINE_GPR_PSEUDO_16(reg16, reg64)                                     \
  RegisterDescriptor(lldb_private::registers::x86_64::id::reg16, #reg16, nullptr, 2,                \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86_64::id::reg16},             \
                     x86_register_info::g_contained_##reg64,                   \
                     x86_register_info::g_invalidate_##reg64, nullptr)

#define DEFINE_GPR_PSEUDO_8H(reg8, reg64)                                      \
  RegisterDescriptor(lldb_private::registers::x86_64::id::reg8, #reg8, nullptr, 1,                  \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86_64::id::reg8},              \
                     x86_register_info::g_contained_##reg64,                   \
                     x86_register_info::g_invalidate_##reg64, nullptr)

#define DEFINE_GPR_PSEUDO_8L(reg8, reg64)                                      \
  RegisterDescriptor(lldb_private::registers::x86_64::id::reg8, #reg8, nullptr, 1,                  \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::x86_64::id::reg8},              \
                     x86_register_info::g_contained_##reg64,                   \
                     x86_register_info::g_invalidate_##reg64, nullptr)

#define DEFINE_FPR_32(name, reg, kind1, kind2, kind3, kind4, reg64)            \
  RegisterDescriptor(lldb_private::registers::x86_64::id::name, #name, nullptr, FPR_SIZE(reg),      \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {kind1, kind2, kind3, kind4, lldb_private::registers::x86_64::id::name},       \
                     x86_register_info::g_contained_##reg64,                   \
                     x86_register_info::g_invalidate_##reg64, nullptr)

static const RegisterDescriptor g_register_infos_x86_64[] = {
    // General purpose registers     EH_Frame              DWARF Generic Process
    // Plugin
    //  ===========================  ==================    ================
    //  =========================   ====================
    DEFINE_GPR(rax, nullptr, lldb_private::registers::x86_64::ehframe::rax, lldb_private::registers::x86_64::dwarf::rax,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(rbx, nullptr, lldb_private::registers::x86_64::ehframe::rbx, lldb_private::registers::x86_64::dwarf::rbx,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(rcx, nullptr, lldb_private::registers::x86_64::ehframe::rcx, lldb_private::registers::x86_64::dwarf::rcx,
               LLDB_REGNUM_GENERIC_ARG4, LLDB_INVALID_REGNUM),
    DEFINE_GPR(rdx, nullptr, lldb_private::registers::x86_64::ehframe::rdx,
               lldb_private::registers::x86_64::dwarf::rdx,
               LLDB_REGNUM_GENERIC_ARG3, LLDB_INVALID_REGNUM),
    DEFINE_GPR(rdi, nullptr, lldb_private::registers::x86_64::ehframe::rdi,
               lldb_private::registers::x86_64::dwarf::rdi,
               LLDB_REGNUM_GENERIC_ARG1, LLDB_INVALID_REGNUM),
    DEFINE_GPR(rsi, nullptr, lldb_private::registers::x86_64::ehframe::rsi,
               lldb_private::registers::x86_64::dwarf::rsi,
               LLDB_REGNUM_GENERIC_ARG2, LLDB_INVALID_REGNUM),
    DEFINE_GPR(rbp, nullptr, lldb_private::registers::x86_64::ehframe::rbp,
               lldb_private::registers::x86_64::dwarf::rbp,
               LLDB_REGNUM_GENERIC_FP, LLDB_INVALID_REGNUM),
    DEFINE_GPR(rsp, nullptr, lldb_private::registers::x86_64::ehframe::rsp,
               lldb_private::registers::x86_64::dwarf::rsp,
               LLDB_REGNUM_GENERIC_SP, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r8, nullptr, lldb_private::registers::x86_64::ehframe::r8,
               lldb_private::registers::x86_64::dwarf::r8,
               LLDB_REGNUM_GENERIC_ARG5, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r9, nullptr, lldb_private::registers::x86_64::ehframe::r9,
               lldb_private::registers::x86_64::dwarf::r9,
               LLDB_REGNUM_GENERIC_ARG6, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r10, nullptr, lldb_private::registers::x86_64::ehframe::r10, lldb_private::registers::x86_64::dwarf::r10,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r11, nullptr, lldb_private::registers::x86_64::ehframe::r11, lldb_private::registers::x86_64::dwarf::r11,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r12, nullptr, lldb_private::registers::x86_64::ehframe::r12, lldb_private::registers::x86_64::dwarf::r12,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r13, nullptr, lldb_private::registers::x86_64::ehframe::r13, lldb_private::registers::x86_64::dwarf::r13,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r14, nullptr, lldb_private::registers::x86_64::ehframe::r14, lldb_private::registers::x86_64::dwarf::r14,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r15, nullptr, lldb_private::registers::x86_64::ehframe::r15, lldb_private::registers::x86_64::dwarf::r15,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(rip, nullptr, lldb_private::registers::x86_64::ehframe::rip, lldb_private::registers::x86_64::dwarf::rip,
               LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_REGNUM),
    DEFINE_GPR(rflags, nullptr,
               lldb_private::registers::x86_64::ehframe::rflags,
               lldb_private::registers::x86_64::dwarf::rflags,
               LLDB_REGNUM_GENERIC_FLAGS, LLDB_INVALID_REGNUM),
    DEFINE_GPR(cs, nullptr, lldb_private::registers::x86_64::ehframe::cs, lldb_private::registers::x86_64::dwarf::cs,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(fs, nullptr, lldb_private::registers::x86_64::ehframe::fs, lldb_private::registers::x86_64::dwarf::fs,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(gs, nullptr, lldb_private::registers::x86_64::ehframe::gs, lldb_private::registers::x86_64::dwarf::gs,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(ss, nullptr, lldb_private::registers::x86_64::ehframe::ss, lldb_private::registers::x86_64::dwarf::ss,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(ds, nullptr, lldb_private::registers::x86_64::ehframe::ds, lldb_private::registers::x86_64::dwarf::ds,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(es, nullptr, lldb_private::registers::x86_64::ehframe::es, lldb_private::registers::x86_64::dwarf::es,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),

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

    //  i387 Floating point registers.         EH_frame             DWARF
    //  Generic              Process Plugin       reg64
    //  ====================================== ===============
    //  ==================   ===================  ==================== =====
    DEFINE_FPR(fctrl, fctrl, lldb_private::registers::x86_64::ehframe::fctrl,
               lldb_private::registers::x86_64::dwarf::fctrl,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(fstat, fstat, lldb_private::registers::x86_64::ehframe::fstat,
               lldb_private::registers::x86_64::dwarf::fstat,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(ftag, ftag, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(fop, fop, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR_32(fiseg, ptr.i386_.fiseg, LLDB_INVALID_REGNUM,
                  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                  fip),
    DEFINE_FPR_32(fioff, ptr.i386_.fioff, LLDB_INVALID_REGNUM,
                  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                  fip),
    DEFINE_FPR(fip, ptr.x86_64.fip, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR_32(foseg, ptr.i386_.foseg, LLDB_INVALID_REGNUM,
                  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                  fdp),
    DEFINE_FPR_32(fooff, ptr.i386_.fooff, LLDB_INVALID_REGNUM,
                  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                  fdp),
    DEFINE_FPR(fdp, ptr.x86_64.fdp, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_FPR(mxcsr, mxcsr, lldb_private::registers::x86_64::ehframe::mxcsr,
               lldb_private::registers::x86_64::dwarf::mxcsr,
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
#undef DEFINE_FPR_32
#undef DEFINE_FP_MM
#undef DEFINE_FP_ST
#undef DEFINE_GPR
#undef DEFINE_GPR_PSEUDO_16
#undef DEFINE_GPR_PSEUDO_32
#undef DEFINE_GPR_PSEUDO_8H
#undef DEFINE_GPR_PSEUDO_8L
#undef DEFINE_XMM
#undef DEFINE_YMM
#undef DR_SIZE
#undef FPR_SIZE
#undef FP_SIZE
#undef XMM_SIZE
#undef YMM_SIZE
