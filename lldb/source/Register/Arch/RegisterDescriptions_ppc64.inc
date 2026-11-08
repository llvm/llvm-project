//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Architecture metadata only. word_size selects the exposed GPR value width.
#define GPR_PPC64_SIZE(...) word_size
#define DEFINE_GPR_PPC64(reg, alt, lldb_kind)                                  \
  RegisterDescriptor(lldb_private::registers::ppc64::id::gpr_##reg, #reg, alt, GPR_PPC64_SIZE(reg),        \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {lldb_private::registers::ppc64::ehframe::reg,                        \
                      lldb_private::registers::ppc64::dwarf::reg, lldb_kind,             \
                      LLDB_INVALID_REGNUM, lldb_private::registers::ppc64::id::gpr_##reg},                 \
                     {}, {}, NULL)
#define DEFINE_FPR_PPC64(reg, alt, lldb_kind)                                  \
  RegisterDescriptor(lldb_private::registers::ppc64::id::fpr_##reg, #reg, alt, 8, lldb::eEncodingIEEE754,  \
                     lldb::eFormatFloat,                                       \
                     {lldb_private::registers::ppc64::ehframe::reg,                        \
                      lldb_private::registers::ppc64::dwarf::reg, lldb_kind,             \
                      LLDB_INVALID_REGNUM, lldb_private::registers::ppc64::id::fpr_##reg},                 \
                     {}, {}, NULL)
#define DEFINE_VMX_PPC64(reg, lldb_kind)                                       \
  RegisterDescriptor(lldb_private::registers::ppc64::id::vmx_##reg, #reg, NULL, 16, lldb::eEncodingVector, \
                     lldb::eFormatVectorOfUInt32,                              \
                     {lldb_private::registers::ppc64::ehframe::reg,                        \
                      lldb_private::registers::ppc64::dwarf::reg, lldb_kind,             \
                      LLDB_INVALID_REGNUM, lldb_private::registers::ppc64::id::vmx_##reg},                 \
                     {}, {}, NULL)

#define DEFINE_VSX_PPC64(reg, lldb_kind)                                       \
  RegisterDescriptor(lldb_private::registers::ppc64::id::vsx_##reg, #reg, NULL, 16, lldb::eEncodingVector, \
                     lldb::eFormatVectorOfUInt32,                              \
                     {lldb_private::registers::ppc64::ehframe::reg,                        \
                      lldb_private::registers::ppc64::dwarf::reg, lldb_kind,             \
                      LLDB_INVALID_REGNUM, lldb_private::registers::ppc64::id::vsx_##reg},                 \
                     {}, {}, NULL)

// General purpose registers.
// EH_Frame, Generic, Process Plugin
#define PPC64_REGS                                                             \
  DEFINE_GPR_PPC64(r0, NULL, LLDB_INVALID_REGNUM)                              \
  , DEFINE_GPR_PPC64(r1, NULL, LLDB_REGNUM_GENERIC_SP),                        \
      DEFINE_GPR_PPC64(r2, NULL, LLDB_INVALID_REGNUM),                         \
      DEFINE_GPR_PPC64(r3, NULL, LLDB_REGNUM_GENERIC_ARG1),                    \
      DEFINE_GPR_PPC64(r4, NULL, LLDB_REGNUM_GENERIC_ARG2),                    \
      DEFINE_GPR_PPC64(r5, NULL, LLDB_REGNUM_GENERIC_ARG3),                    \
      DEFINE_GPR_PPC64(r6, NULL, LLDB_REGNUM_GENERIC_ARG4),                    \
      DEFINE_GPR_PPC64(r7, NULL, LLDB_REGNUM_GENERIC_ARG5),                    \
      DEFINE_GPR_PPC64(r8, NULL, LLDB_REGNUM_GENERIC_ARG6),                    \
      DEFINE_GPR_PPC64(r9, NULL, LLDB_REGNUM_GENERIC_ARG7),                    \
      DEFINE_GPR_PPC64(r10, NULL, LLDB_REGNUM_GENERIC_ARG8),                   \
      DEFINE_GPR_PPC64(r11, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r12, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r13, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r14, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r15, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r16, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r17, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r18, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r19, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r20, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r21, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r22, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r23, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r24, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r25, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r26, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r27, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r28, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r29, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r30, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(r31, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(cr, NULL, LLDB_REGNUM_GENERIC_FLAGS),                   \
      DEFINE_GPR_PPC64(msr, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(xer, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(lr, NULL, LLDB_REGNUM_GENERIC_RA),                      \
      DEFINE_GPR_PPC64(ctr, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_GPR_PPC64(pc, NULL, LLDB_REGNUM_GENERIC_PC),                      \
      DEFINE_FPR_PPC64(f0, NULL, LLDB_INVALID_REGNUM),                         \
      DEFINE_FPR_PPC64(f1, NULL, LLDB_INVALID_REGNUM),                         \
      DEFINE_FPR_PPC64(f2, NULL, LLDB_INVALID_REGNUM),                         \
      DEFINE_FPR_PPC64(f3, NULL, LLDB_INVALID_REGNUM),                         \
      DEFINE_FPR_PPC64(f4, NULL, LLDB_INVALID_REGNUM),                         \
      DEFINE_FPR_PPC64(f5, NULL, LLDB_INVALID_REGNUM),                         \
      DEFINE_FPR_PPC64(f6, NULL, LLDB_INVALID_REGNUM),                         \
      DEFINE_FPR_PPC64(f7, NULL, LLDB_INVALID_REGNUM),                         \
      DEFINE_FPR_PPC64(f8, NULL, LLDB_INVALID_REGNUM),                         \
      DEFINE_FPR_PPC64(f9, NULL, LLDB_INVALID_REGNUM),                         \
      DEFINE_FPR_PPC64(f10, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f11, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f12, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f13, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f14, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f15, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f16, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f17, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f18, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f19, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f20, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f21, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f22, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f23, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f24, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f25, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f26, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f27, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f28, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f29, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f30, NULL, LLDB_INVALID_REGNUM),                        \
      DEFINE_FPR_PPC64(f31, NULL, LLDB_INVALID_REGNUM),                        \
      RegisterDescriptor(lldb_private::registers::ppc64::id::fpr_fpscr, "fpscr", NULL, 8,                    \
                         lldb::eEncodingUint, lldb::eFormatHex,                \
                         {lldb_private::registers::ppc64::ehframe::fpscr,                      \
                          lldb_private::registers::ppc64::dwarf::fpscr, LLDB_INVALID_REGNUM, \
                          LLDB_INVALID_REGNUM, lldb_private::registers::ppc64::id::fpr_fpscr},               \
                         {}, {}, NULL),                                        \
      DEFINE_VMX_PPC64(vr0, LLDB_INVALID_REGNUM),                              \
      DEFINE_VMX_PPC64(vr1, LLDB_INVALID_REGNUM),                              \
      DEFINE_VMX_PPC64(vr2, LLDB_INVALID_REGNUM),                              \
      DEFINE_VMX_PPC64(vr3, LLDB_INVALID_REGNUM),                              \
      DEFINE_VMX_PPC64(vr4, LLDB_INVALID_REGNUM),                              \
      DEFINE_VMX_PPC64(vr5, LLDB_INVALID_REGNUM),                              \
      DEFINE_VMX_PPC64(vr6, LLDB_INVALID_REGNUM),                              \
      DEFINE_VMX_PPC64(vr7, LLDB_INVALID_REGNUM),                              \
      DEFINE_VMX_PPC64(vr8, LLDB_INVALID_REGNUM),                              \
      DEFINE_VMX_PPC64(vr9, LLDB_INVALID_REGNUM),                              \
      DEFINE_VMX_PPC64(vr10, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr11, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr12, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr13, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr14, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr15, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr16, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr17, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr18, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr19, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr20, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr21, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr22, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr23, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr24, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr25, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr26, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr27, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr28, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr29, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr30, LLDB_INVALID_REGNUM),                             \
      DEFINE_VMX_PPC64(vr31, LLDB_INVALID_REGNUM),                             \
      RegisterDescriptor(lldb_private::registers::ppc64::id::vmx_vscr, "vscr", \
                         NULL, 4, lldb::eEncodingUint, lldb::eFormatHex,       \
                         {lldb_private::registers::ppc64::ehframe::vscr,       \
                          lldb_private::registers::ppc64::dwarf::vscr,         \
                          LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,            \
                          lldb_private::registers::ppc64::id::vmx_vscr},       \
                         {}, {}, NULL),                                        \
      RegisterDescriptor(                                                      \
          lldb_private::registers::ppc64::id::vmx_vrsave, "vrsave", NULL, 4, lldb::eEncodingUint,            \
          lldb::eFormatHex,                                                    \
          {lldb_private::registers::ppc64::ehframe::vrsave, lldb_private::registers::ppc64::dwarf::vrsave,   \
           LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, lldb_private::registers::ppc64::id::vmx_vrsave},        \
          {}, {}, NULL),                                                       \
      DEFINE_VSX_PPC64(vs0, LLDB_INVALID_REGNUM),                              \
      DEFINE_VSX_PPC64(vs1, LLDB_INVALID_REGNUM),                              \
      DEFINE_VSX_PPC64(vs2, LLDB_INVALID_REGNUM),                              \
      DEFINE_VSX_PPC64(vs3, LLDB_INVALID_REGNUM),                              \
      DEFINE_VSX_PPC64(vs4, LLDB_INVALID_REGNUM),                              \
      DEFINE_VSX_PPC64(vs5, LLDB_INVALID_REGNUM),                              \
      DEFINE_VSX_PPC64(vs6, LLDB_INVALID_REGNUM),                              \
      DEFINE_VSX_PPC64(vs7, LLDB_INVALID_REGNUM),                              \
      DEFINE_VSX_PPC64(vs8, LLDB_INVALID_REGNUM),                              \
      DEFINE_VSX_PPC64(vs9, LLDB_INVALID_REGNUM),                              \
      DEFINE_VSX_PPC64(vs10, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs11, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs12, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs13, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs14, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs15, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs16, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs17, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs18, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs19, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs20, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs21, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs22, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs23, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs24, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs25, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs26, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs27, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs28, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs29, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs30, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs31, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs32, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs33, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs34, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs35, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs36, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs37, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs38, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs39, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs40, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs41, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs42, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs43, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs44, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs45, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs46, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs47, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs48, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs49, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs50, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs51, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs52, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs53, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs54, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs55, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs56, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs57, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs58, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs59, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs60, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs61, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs62, LLDB_INVALID_REGNUM),                             \
      DEFINE_VSX_PPC64(vs63, LLDB_INVALID_REGNUM), /* */

static const RegisterDescriptor values[] = {PPC64_REGS};
#undef DEFINE_FPR_PPC64
#undef DEFINE_GPR_PPC64
#undef DEFINE_VMX_PPC64
#undef DEFINE_VSX_PPC64
#undef GPR_PPC64_SIZE
#undef PPC64_REGS
