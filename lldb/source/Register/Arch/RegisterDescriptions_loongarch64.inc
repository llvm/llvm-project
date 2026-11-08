//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Textual register-table fragment; intentionally has no include guard.
// Only the corresponding common CPU builder includes this fragment.
// Include its ordinary dependencies at file scope.

#if !defined(DECLARE_REGISTER_INFOS_LOONGARCH64_STRUCT)
#error Select a register-table generation mode before including this fragment
#endif

#ifdef DECLARE_REGISTER_INFOS_LOONGARCH64_STRUCT

// I suppose EHFrame and DWARF are the same.
#define KIND_HELPER(reg, generic_kind)                                         \
  {                                                                            \
    lldb_private::registers::loongarch::ehframe::reg, lldb_private::registers::loongarch::dwarf::reg, generic_kind,  \
    LLDB_INVALID_REGNUM, lldb_private::registers::loongarch::id::reg                                       \
  }

// Generates register kinds array for generic purpose registers
#define GPR64_KIND(reg, generic_kind) KIND_HELPER(reg, generic_kind)

// Generates register kinds array for floating point registers
#define FPR64_KIND(reg, generic_kind) KIND_HELPER(reg, generic_kind)

// Defines a 64-bit general purpose register
#define DEFINE_GPR64(reg, generic_kind) DEFINE_GPR64_ALT(reg, reg, generic_kind)
#define DEFINE_GPR64_ALT(reg, alt, generic_kind)                               \
  RegisterDescriptor(lldb_private::registers::loongarch::id::gpr_##reg, #reg, #alt, 8,                     \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     GPR64_KIND(gpr_##reg, generic_kind), {}, {}, nullptr)

// Defines a 64-bit floating point register
#define DEFINE_FPR64(reg, generic_kind) DEFINE_FPR64_ALT(reg, reg, generic_kind)
#define DEFINE_FPR64_ALT(reg, alt, generic_kind)                               \
  RegisterDescriptor(lldb_private::registers::loongarch::id::fpr_##reg, #reg, #alt, 8,                     \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     FPR64_KIND(fpr_##reg, generic_kind), {}, {}, nullptr)

#define DEFINE_FCC(reg, generic_kind)                                          \
  RegisterDescriptor(lldb_private::registers::loongarch::id::fpr_##reg, #reg, nullptr, 1,                  \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     FPR64_KIND(fpr_##reg, generic_kind), {}, {}, nullptr)

#define DEFINE_FCSR(reg, generic_kind)                                         \
  RegisterDescriptor(lldb_private::registers::loongarch::id::fpr_##reg, #reg, nullptr, 4,                  \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     FPR64_KIND(fpr_##reg, generic_kind), {}, {}, nullptr)

#define DEFINE_LSX(reg, generic_kind) \
  RegisterDescriptor(lldb_private::registers::loongarch::id::lsx_##reg, #reg, nullptr, 16,                 \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     KIND_HELPER(lsx_##reg, generic_kind), {}, {}, nullptr)

#define DEFINE_LASX(reg, generic_kind) \
  RegisterDescriptor(lldb_private::registers::loongarch::id::lasx_##reg, #reg, nullptr, 32,                \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     KIND_HELPER(lasx_##reg, generic_kind), {}, {}, nullptr)

static const RegisterDescriptor g_register_infos_loongarch64[] = {
    DEFINE_GPR64_ALT(r0, zero, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r1, ra, LLDB_REGNUM_GENERIC_RA),
    DEFINE_GPR64_ALT(r2, tp, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r3, sp, LLDB_REGNUM_GENERIC_SP),
    DEFINE_GPR64_ALT(r4, a0, LLDB_REGNUM_GENERIC_ARG1),
    DEFINE_GPR64_ALT(r5, a1, LLDB_REGNUM_GENERIC_ARG2),
    DEFINE_GPR64_ALT(r6, a2, LLDB_REGNUM_GENERIC_ARG3),
    DEFINE_GPR64_ALT(r7, a3, LLDB_REGNUM_GENERIC_ARG4),
    DEFINE_GPR64_ALT(r8, a4, LLDB_REGNUM_GENERIC_ARG5),
    DEFINE_GPR64_ALT(r9, a5, LLDB_REGNUM_GENERIC_ARG6),
    DEFINE_GPR64_ALT(r10, a6, LLDB_REGNUM_GENERIC_ARG7),
    DEFINE_GPR64_ALT(r11, a7, LLDB_REGNUM_GENERIC_ARG8),
    DEFINE_GPR64_ALT(r12, t0, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r13, t1, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r14, t2, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r15, t3, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r16, t4, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r17, t5, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r18, t6, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r19, t7, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r20, t8, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(r21, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r22, fp, LLDB_REGNUM_GENERIC_FP),
    DEFINE_GPR64_ALT(r23, s0, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r24, s1, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r25, s2, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r26, s3, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r27, s4, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r28, s5, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r29, s6, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r30, s7, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(r31, s8, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(pc, LLDB_REGNUM_GENERIC_PC),
    DEFINE_GPR64(badv, LLDB_INVALID_REGNUM),

    DEFINE_FPR64_ALT(f0, fa0, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f1, fa1, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f2, fa2, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f3, fa3, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f4, fa4, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f5, fa5, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f6, fa6, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f7, fa7, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f8, ft0, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f9, ft1, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f10, ft2, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f11, ft3, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f12, ft4, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f13, ft5, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f14, ft6, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f15, ft7, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f16, ft8, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f17, ft9, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f18, ft10, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f19, ft11, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f20, ft12, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f21, ft13, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f22, ft14, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f23, ft15, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f24, fs0, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f25, fs1, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f26, fs2, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f27, fs3, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f28, fs4, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f29, fs5, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f30, fs6, LLDB_INVALID_REGNUM),
    DEFINE_FPR64_ALT(f31, fs7, LLDB_INVALID_REGNUM),

    DEFINE_FCC(fcc0, LLDB_INVALID_REGNUM),
    DEFINE_FCC(fcc1, LLDB_INVALID_REGNUM),
    DEFINE_FCC(fcc2, LLDB_INVALID_REGNUM),
    DEFINE_FCC(fcc3, LLDB_INVALID_REGNUM),
    DEFINE_FCC(fcc4, LLDB_INVALID_REGNUM),
    DEFINE_FCC(fcc5, LLDB_INVALID_REGNUM),
    DEFINE_FCC(fcc6, LLDB_INVALID_REGNUM),
    DEFINE_FCC(fcc7, LLDB_INVALID_REGNUM),
    DEFINE_FCSR(fcsr, LLDB_INVALID_REGNUM),

    DEFINE_LSX(vr0, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr1, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr2, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr3, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr4, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr5, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr6, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr7, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr8, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr9, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr10, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr11, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr12, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr13, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr14, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr15, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr16, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr17, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr18, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr19, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr20, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr21, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr22, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr23, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr24, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr25, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr26, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr27, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr28, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr29, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr30, LLDB_INVALID_REGNUM),
    DEFINE_LSX(vr31, LLDB_INVALID_REGNUM),

    DEFINE_LASX(xr0, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr1, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr2, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr3, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr4, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr5, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr6, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr7, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr8, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr9, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr10, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr11, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr12, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr13, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr14, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr15, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr16, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr17, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr18, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr19, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr20, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr21, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr22, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr23, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr24, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr25, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr26, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr27, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr28, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr29, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr30, LLDB_INVALID_REGNUM),
    DEFINE_LASX(xr31, LLDB_INVALID_REGNUM),
};

#endif // DECLARE_REGISTER_INFOS_LOONGARCH64_STRUCT

#undef KIND_HELPER
#undef GPR64_KIND
#undef FPR64_KIND
#undef DEFINE_GPR64
#undef DEFINE_GPR64_ALT
#undef DEFINE_FPR64
#undef DEFINE_FPR64_ALT
#undef DEFINE_FCC
#undef DEFINE_FCSR
#undef DEFINE_LSX
#undef DEFINE_LASX
