//===-- RegisterInfos_riscv64.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef DECLARE_REGISTER_INFOS_RISCV64_STRUCT

#include <stddef.h>

#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private.h"

#include "Utility/RISCV_DWARF_Registers.h"
#include "lldb-riscv-register-enums.h"

#ifndef GPR_OFFSET
#error GPR_OFFSET must be defined before including this header file
#endif

#ifndef FPR_OFFSET
#error FPR_OFFSET must be defined before including this header file
#endif

using namespace riscv_dwarf;

// clang-format off

// I suppose EHFrame and DWARF are the same.
#define KIND_HELPER(reg, generic_kind)                                         \
  {                                                                            \
    riscv_dwarf::dwarf_##reg, riscv_dwarf::dwarf_##reg, generic_kind,          \
    LLDB_INVALID_REGNUM, reg##_riscv                                           \
  }

// Generates register kinds array for vector registers
#define GPR64_KIND(reg, generic_kind) KIND_HELPER(reg, generic_kind)

// FPR register kinds array for vector registers
#define FPR64_KIND(reg, generic_kind) KIND_HELPER(reg, generic_kind)

// Defines a 64-bit general purpose register
#define DEFINE_GPR64(reg, generic_kind) DEFINE_GPR64_ALT(reg, reg, generic_kind)

// Defines a 64-bit general purpose register
#define DEFINE_GPR64_ALT(reg, alt, generic_kind)                               \
  {                                                                            \
    #reg, #alt, 8, GPR_OFFSET(gpr_##reg##_riscv - gpr_first_riscv),            \
    lldb::eEncodingUint, lldb::eFormatHex,                                     \
    GPR64_KIND(gpr_##reg, generic_kind), nullptr, nullptr                      \
  }

#define DEFINE_FPR64(reg, generic_kind)                                        \
  {                                                                            \
    #reg, nullptr, 8, FPR_OFFSET(fpr_##reg##_riscv - fpr_first_riscv),         \
    lldb::eEncodingUint, lldb::eFormatHex,                                     \
    FPR64_KIND(fpr_##reg, generic_kind), nullptr, nullptr                      \
  }

// clang-format on

static lldb_private::RegisterInfo g_register_infos_riscv64_le[] = {
    // DEFINE_GPR64(name, GENERIC KIND)
    DEFINE_GPR64(pc, LLDB_REGNUM_GENERIC_PC),
    DEFINE_GPR64_ALT(ra, x1, LLDB_REGNUM_GENERIC_RA),
    DEFINE_GPR64_ALT(sp, x2, LLDB_REGNUM_GENERIC_SP),
    DEFINE_GPR64_ALT(gp, x3, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(tp, x4, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(t0, x5, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(t1, x6, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(t2, x7, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(fp, x8, LLDB_REGNUM_GENERIC_FP),
    DEFINE_GPR64_ALT(s1, x9, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(a0, x10, LLDB_REGNUM_GENERIC_ARG1),
    DEFINE_GPR64_ALT(a1, x11, LLDB_REGNUM_GENERIC_ARG2),
    DEFINE_GPR64_ALT(a2, x12, LLDB_REGNUM_GENERIC_ARG3),
    DEFINE_GPR64_ALT(a3, x13, LLDB_REGNUM_GENERIC_ARG4),
    DEFINE_GPR64_ALT(a4, x14, LLDB_REGNUM_GENERIC_ARG5),
    DEFINE_GPR64_ALT(a5, x15, LLDB_REGNUM_GENERIC_ARG6),
    DEFINE_GPR64_ALT(a6, x16, LLDB_REGNUM_GENERIC_ARG7),
    DEFINE_GPR64_ALT(a7, x17, LLDB_REGNUM_GENERIC_ARG8),
    DEFINE_GPR64_ALT(s2, x18, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(s3, x19, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(s4, x20, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(s5, x21, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(s6, x22, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(s7, x23, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(s8, x24, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(s9, x25, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(s10, x26, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(s11, x27, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(t3, x28, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(t4, x29, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(t5, x30, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(t6, x31, LLDB_INVALID_REGNUM),
    DEFINE_GPR64_ALT(zero, x0, LLDB_INVALID_REGNUM),

    DEFINE_FPR64(f0, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f1, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f2, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f3, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f4, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f5, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f6, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f7, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f8, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f9, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f10, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f11, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f12, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f13, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f14, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f15, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f16, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f17, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f18, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f19, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f20, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f21, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f22, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f23, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f24, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f25, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f26, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f27, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f28, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f29, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f30, LLDB_INVALID_REGNUM),
    DEFINE_FPR64(f31, LLDB_INVALID_REGNUM),
};

#endif // DECLARE_REGISTER_INFOS_RISCV64_STRUCT
