//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Architecture metadata only. word_size selects the exposed GPR value width.
#define GPR_SIZE(...) word_size
#define DEFINE_GPR(reg, alt, lldb_kind)                                        \
  RegisterDescriptor(lldb_private::registers::ppc64le::id::gpr_##reg, #reg, alt, GPR_SIZE(reg),            \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {lldb_private::registers::ppc64le::ehframe::reg,                    \
                      lldb_private::registers::ppc64le::dwarf::reg, lldb_kind,         \
                      LLDB_INVALID_REGNUM, lldb_private::registers::ppc64le::id::gpr_##reg},               \
                     {}, {}, NULL)
#define DEFINE_FPR(reg, alt, lldb_kind)                                        \
  RegisterDescriptor(lldb_private::registers::ppc64le::id::fpr_##reg, #reg, alt, 8,                        \
                     lldb::eEncodingIEEE754, lldb::eFormatFloat,               \
                     {lldb_private::registers::ppc64le::ehframe::reg,                    \
                      lldb_private::registers::ppc64le::dwarf::reg, lldb_kind,         \
                      LLDB_INVALID_REGNUM, lldb_private::registers::ppc64le::id::fpr_##reg},               \
                     {}, {}, NULL)
#define DEFINE_VMX(reg, lldb_kind)                                             \
  RegisterDescriptor(lldb_private::registers::ppc64le::id::vmx_##reg, #reg, NULL, 16,                      \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt32,       \
                     {lldb_private::registers::ppc64le::ehframe::reg,                    \
                      lldb_private::registers::ppc64le::dwarf::reg, lldb_kind,         \
                      LLDB_INVALID_REGNUM, lldb_private::registers::ppc64le::id::vmx_##reg},               \
                     {}, {}, NULL)
#define DEFINE_VSX(reg, lldb_kind)                                             \
  RegisterDescriptor(lldb_private::registers::ppc64le::id::vsx_##reg, #reg, NULL, 16,                      \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt32,       \
                     {lldb_private::registers::ppc64le::ehframe::reg,                    \
                      lldb_private::registers::ppc64le::dwarf::reg, lldb_kind,         \
                      LLDB_INVALID_REGNUM, lldb_private::registers::ppc64le::id::vsx_##reg},               \
                     {}, {}, NULL)

// General purpose registers.
// EH_Frame, Generic, Process Plugin
#define POWERPC_REGS                                                           \
  DEFINE_GPR(r0, NULL, LLDB_INVALID_REGNUM)                                    \
  , DEFINE_GPR(r1, NULL, LLDB_REGNUM_GENERIC_SP),                              \
      DEFINE_GPR(r2, NULL, LLDB_INVALID_REGNUM),                               \
      DEFINE_GPR(r3, NULL, LLDB_REGNUM_GENERIC_ARG1),                          \
      DEFINE_GPR(r4, NULL, LLDB_REGNUM_GENERIC_ARG2),                          \
      DEFINE_GPR(r5, NULL, LLDB_REGNUM_GENERIC_ARG3),                          \
      DEFINE_GPR(r6, NULL, LLDB_REGNUM_GENERIC_ARG4),                          \
      DEFINE_GPR(r7, NULL, LLDB_REGNUM_GENERIC_ARG5),                          \
      DEFINE_GPR(r8, NULL, LLDB_REGNUM_GENERIC_ARG6),                          \
      DEFINE_GPR(r9, NULL, LLDB_REGNUM_GENERIC_ARG7),                          \
      DEFINE_GPR(r10, NULL, LLDB_REGNUM_GENERIC_ARG8),                         \
      DEFINE_GPR(r11, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r12, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r13, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r14, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r15, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r16, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r17, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r18, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r19, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r20, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r21, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r22, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r23, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r24, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r25, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r26, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r27, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r28, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r29, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r30, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r31, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(pc, NULL, LLDB_REGNUM_GENERIC_PC),                            \
      DEFINE_GPR(msr, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(ctr, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(lr, NULL, LLDB_REGNUM_GENERIC_RA),                            \
      DEFINE_GPR(xer, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(cr, NULL, LLDB_REGNUM_GENERIC_FLAGS),                         \
      DEFINE_FPR(f0, NULL, LLDB_INVALID_REGNUM),                               \
      DEFINE_FPR(f1, NULL, LLDB_INVALID_REGNUM),                               \
      DEFINE_FPR(f2, NULL, LLDB_INVALID_REGNUM),                               \
      DEFINE_FPR(f3, NULL, LLDB_INVALID_REGNUM),                               \
      DEFINE_FPR(f4, NULL, LLDB_INVALID_REGNUM),                               \
      DEFINE_FPR(f5, NULL, LLDB_INVALID_REGNUM),                               \
      DEFINE_FPR(f6, NULL, LLDB_INVALID_REGNUM),                               \
      DEFINE_FPR(f7, NULL, LLDB_INVALID_REGNUM),                               \
      DEFINE_FPR(f8, NULL, LLDB_INVALID_REGNUM),                               \
      DEFINE_FPR(f9, NULL, LLDB_INVALID_REGNUM),                               \
      DEFINE_FPR(f10, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f11, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f12, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f13, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f14, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f15, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f16, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f17, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f18, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f19, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f20, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f21, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f22, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f23, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f24, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f25, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f26, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f27, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f28, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f29, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f30, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_FPR(f31, NULL, LLDB_INVALID_REGNUM),                              \
      RegisterDescriptor(lldb_private::registers::ppc64le::id::fpr_fpscr, "fpscr", NULL, 8,                  \
                         lldb::eEncodingUint, lldb::eFormatHex,                \
                         {lldb_private::registers::ppc64le::ehframe::fpscr,                  \
                          lldb_private::registers::ppc64le::dwarf::fpscr,                  \
                          LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,            \
                          lldb_private::registers::ppc64le::id::fpr_fpscr},    \
                         {}, {}, NULL),                                        \
      DEFINE_VMX(vr0, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VMX(vr1, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VMX(vr2, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VMX(vr3, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VMX(vr4, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VMX(vr5, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VMX(vr6, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VMX(vr7, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VMX(vr8, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VMX(vr9, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VMX(vr10, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr11, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr12, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr13, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr14, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr15, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr16, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr17, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr18, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr19, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr20, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr21, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr22, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr23, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr24, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr25, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr26, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr27, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr28, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr29, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr30, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VMX(vr31, LLDB_INVALID_REGNUM),                                   \
      RegisterDescriptor(lldb_private::registers::ppc64le::id::vmx_vscr, "vscr", NULL, 4,                    \
                         lldb::eEncodingUint, lldb::eFormatHex,                \
                         {lldb_private::registers::ppc64le::ehframe::vscr,                   \
                          lldb_private::registers::ppc64le::dwarf::vscr,                   \
                          LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,            \
                          lldb_private::registers::ppc64le::id::vmx_vscr},     \
                         {}, {}, NULL),                                        \
      RegisterDescriptor(lldb_private::registers::ppc64le::id::vmx_vrsave, "vrsave", NULL, 4,                \
                         lldb::eEncodingUint, lldb::eFormatHex,                \
                         {lldb_private::registers::ppc64le::ehframe::vrsave,                 \
                          lldb_private::registers::ppc64le::dwarf::vrsave,                 \
                          LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,            \
                          lldb_private::registers::ppc64le::id::vmx_vrsave},   \
                         {}, {}, NULL),                                        \
      DEFINE_VSX(vs0, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VSX(vs1, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VSX(vs2, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VSX(vs3, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VSX(vs4, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VSX(vs5, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VSX(vs6, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VSX(vs7, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VSX(vs8, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VSX(vs9, LLDB_INVALID_REGNUM),                                    \
      DEFINE_VSX(vs10, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs11, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs12, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs13, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs14, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs15, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs16, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs17, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs18, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs19, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs20, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs21, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs22, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs23, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs24, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs25, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs26, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs27, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs28, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs29, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs30, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs31, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs32, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs33, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs34, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs35, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs36, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs37, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs38, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs39, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs40, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs41, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs42, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs43, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs44, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs45, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs46, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs47, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs48, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs49, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs50, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs51, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs52, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs53, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs54, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs55, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs56, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs57, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs58, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs59, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs60, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs61, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs62, LLDB_INVALID_REGNUM),                                   \
      DEFINE_VSX(vs63, LLDB_INVALID_REGNUM), /* */

static const RegisterDescriptor values[] = {POWERPC_REGS};
#undef DEFINE_FPR
#undef DEFINE_GPR
#undef DEFINE_VMX
#undef DEFINE_VSX
#undef GPR_SIZE
#undef POWERPC_REGS
