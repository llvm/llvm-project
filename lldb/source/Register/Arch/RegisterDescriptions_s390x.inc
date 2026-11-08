//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define DEFINE_GPR(name, size, alt, generic)                                   \
  RegisterDescriptor(lldb_private::registers::s390x::id::name, #name, alt, size,                    \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {lldb_private::registers::s390x::ehframe::name,                       \
                      lldb_private::registers::s390x::dwarf::name, generic,              \
                      LLDB_INVALID_REGNUM, lldb_private::registers::s390x::id::name},               \
                     {}, {}, NULL)

#define DEFINE_GPR_NODWARF(name, size, alt, generic)                           \
  RegisterDescriptor(lldb_private::registers::s390x::id::name, #name, alt, size,                    \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, generic,       \
                      LLDB_INVALID_REGNUM,                                     \
                      lldb_private::registers::s390x::id::name},               \
                     {}, {}, NULL)

#define DEFINE_FPR(name, size)                                                 \
  RegisterDescriptor(lldb_private::registers::s390x::id::name, #name, NULL, size,                   \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {lldb_private::registers::s390x::ehframe::name,                       \
                      lldb_private::registers::s390x::dwarf::name, LLDB_INVALID_REGNUM,  \
                      LLDB_INVALID_REGNUM, lldb_private::registers::s390x::id::name},               \
                     {}, {}, NULL)

#define DEFINE_FPR_NODWARF(name, size)                                         \
  RegisterDescriptor(lldb_private::registers::s390x::id::name, #name, NULL, size,                   \
                     lldb::eEncodingUint, lldb::eFormatHex,                    \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,                \
                      lldb_private::registers::s390x::id::name},               \
                     {}, {}, NULL)

static const RegisterDescriptor values[] = {
    // General purpose registers.
    DEFINE_GPR(r0, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r1, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r2, 8, nullptr, LLDB_REGNUM_GENERIC_ARG1),
    DEFINE_GPR(r3, 8, nullptr, LLDB_REGNUM_GENERIC_ARG2),
    DEFINE_GPR(r4, 8, nullptr, LLDB_REGNUM_GENERIC_ARG3),
    DEFINE_GPR(r5, 8, nullptr, LLDB_REGNUM_GENERIC_ARG4),
    DEFINE_GPR(r6, 8, nullptr, LLDB_REGNUM_GENERIC_ARG5),
    DEFINE_GPR(r7, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r8, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r9, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r10, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r11, 8, nullptr, LLDB_REGNUM_GENERIC_FP),
    DEFINE_GPR(r12, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r13, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r14, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r15, 8, nullptr, LLDB_REGNUM_GENERIC_SP),
    DEFINE_GPR(acr0, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr1, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr2, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr3, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr4, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr5, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr6, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr7, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr8, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr9, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr10, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr11, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr12, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr13, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr14, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(acr15, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_GPR(pswm, 8, nullptr, LLDB_REGNUM_GENERIC_FLAGS),
    DEFINE_GPR(pswa, 8, nullptr, LLDB_REGNUM_GENERIC_PC),

    // Floating point registers.
    DEFINE_FPR(f0, 8),
    DEFINE_FPR(f1, 8),
    DEFINE_FPR(f2, 8),
    DEFINE_FPR(f3, 8),
    DEFINE_FPR(f4, 8),
    DEFINE_FPR(f5, 8),
    DEFINE_FPR(f6, 8),
    DEFINE_FPR(f7, 8),
    DEFINE_FPR(f8, 8),
    DEFINE_FPR(f9, 8),
    DEFINE_FPR(f10, 8),
    DEFINE_FPR(f11, 8),
    DEFINE_FPR(f12, 8),
    DEFINE_FPR(f13, 8),
    DEFINE_FPR(f14, 8),
    DEFINE_FPR(f15, 8),
    DEFINE_FPR_NODWARF(fpc, 4),

};
#undef DEFINE_FPR
#undef DEFINE_FPR_NODWARF
#undef DEFINE_GPR
#undef DEFINE_GPR_NODWARF
