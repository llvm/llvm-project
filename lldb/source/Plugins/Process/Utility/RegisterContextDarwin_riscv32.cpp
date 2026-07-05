//===-- RegisterContextDarwin_riscv32.cpp
//------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"

#include <cstddef>

#include <memory>

#include "RegisterContextDarwin_riscv32.h"
#include "Utility/RISCV_DWARF_Registers.h"

using namespace lldb;
using namespace lldb_private;

enum {
  gpr_x0 = 0,
  gpr_x1,
  gpr_x2,
  gpr_x3,
  gpr_x4,
  gpr_x5,
  gpr_x6,
  gpr_x7,
  gpr_x8,
  gpr_x9,
  gpr_x10,
  gpr_x11,
  gpr_x12,
  gpr_x13,
  gpr_x14,
  gpr_x15,
  gpr_x16,
  gpr_x17,
  gpr_x18,
  gpr_x19,
  gpr_x20,
  gpr_x21,
  gpr_x22,
  gpr_x23,
  gpr_x24,
  gpr_x25,
  gpr_x26,
  gpr_x27,
  gpr_x28,
  gpr_x29,
  gpr_x30,
  gpr_x31,
  gpr_pc,

  fpr_f0,
  fpr_f1,
  fpr_f2,
  fpr_f3,
  fpr_f4,
  fpr_f5,
  fpr_f6,
  fpr_f7,
  fpr_f8,
  fpr_f9,
  fpr_f10,
  fpr_f11,
  fpr_f12,
  fpr_f13,
  fpr_f14,
  fpr_f15,
  fpr_f16,
  fpr_f17,
  fpr_f18,
  fpr_f19,
  fpr_f20,
  fpr_f21,
  fpr_f22,
  fpr_f23,
  fpr_f24,
  fpr_f25,
  fpr_f26,
  fpr_f27,
  fpr_f28,
  fpr_f29,
  fpr_f30,
  fpr_f31,
  fpr_fcsr,

  exc_exception,
  exc_fsr,
  exc_far,

  csr_bank,

  k_num_registers
};

/* clang-format off */
#define GPR_OFFSET(reg)                                                        \
  (LLVM_EXTENSION offsetof(RegisterContextDarwin_riscv32::GPR, reg))
#define FPU_OFFSET(reg)                                                        \
  (LLVM_EXTENSION offsetof(RegisterContextDarwin_riscv32::FPU, reg) +          \
   sizeof(RegisterContextDarwin_riscv32::GPR))
#define EXC_OFFSET(reg)                                                        \
  (LLVM_EXTENSION offsetof(RegisterContextDarwin_riscv32::EXC, reg) +          \
   sizeof(RegisterContextDarwin_riscv32::GPR) +                                \
   sizeof(RegisterContextDarwin_riscv32::FPU))

// These macros will auto define the register name, alt name, register size,
// register offset, encoding, format and native register. This ensures that the
// register state structures are defined correctly and have the correct sizes
// and offsets.
#define DEFINE_GPR_ABI(reg, canon)                                             \
      #reg, #canon,                                                            \
      sizeof(((RegisterContextDarwin_riscv32::GPR *)nullptr)->canon),          \
      GPR_OFFSET(canon), eEncodingUint, eFormatHex
#define DEFINE_GPR(reg)                                                        \
      #reg, nullptr,                                                           \
      sizeof(((RegisterContextDarwin_riscv32::GPR *)nullptr)->reg),            \
      GPR_OFFSET(reg), eEncodingUint, eFormatHex
#define DEFINE_FPU_ABI(reg, canon)                                             \
      #reg, #canon,                                                            \
      sizeof(((RegisterContextDarwin_riscv32::FPU *)nullptr)->canon),          \
      FPU_OFFSET(canon), eEncodingUint, eFormatHex
#define DEFINE_FPU(reg)                                                        \
      #reg, nullptr,                                                           \
      sizeof(((RegisterContextDarwin_riscv32::FPU *)nullptr)->reg),            \
      FPU_OFFSET(reg), eEncodingUint, eFormatHex
#define DEFINE_EXC(reg)                                                        \
      #reg, nullptr,                                                           \
      sizeof(((RegisterContextDarwin_riscv32::EXC *)nullptr)->reg),            \
      EXC_OFFSET(reg), eEncodingUint, eFormatHex
#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterContextDarwin_riscv32::GPR) +                                \
   sizeof(RegisterContextDarwin_riscv32::FPU) +                                \
   sizeof(RegisterContextDarwin_riscv32::EXC) +                                \
   sizeof(RegisterContextDarwin_riscv32::CSR))
/* clang-format on */

static RegisterInfo g_register_infos[] = {
    {
        DEFINE_GPR_ABI(zero, x0),
        {riscv_dwarf::dwarf_gpr_x0, riscv_dwarf::dwarf_gpr_x0,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x0},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(ra, x1),
        {riscv_dwarf::dwarf_gpr_x1, riscv_dwarf::dwarf_gpr_x1,
         LLDB_REGNUM_GENERIC_RA, LLDB_INVALID_REGNUM, gpr_x1},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(sp, x2),
        {riscv_dwarf::dwarf_gpr_x2, riscv_dwarf::dwarf_gpr_x2,
         LLDB_REGNUM_GENERIC_SP, LLDB_INVALID_REGNUM, gpr_x2},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(gp, x3),
        {riscv_dwarf::dwarf_gpr_x3, riscv_dwarf::dwarf_gpr_x3,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x3},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(tp, x4),
        {riscv_dwarf::dwarf_gpr_x4, riscv_dwarf::dwarf_gpr_x4,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x4},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(t0, x5),
        {riscv_dwarf::dwarf_gpr_x5, riscv_dwarf::dwarf_gpr_x5,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x5},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(t1, x6),
        {riscv_dwarf::dwarf_gpr_x6, riscv_dwarf::dwarf_gpr_x6,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x6},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(t2, x7),
        {riscv_dwarf::dwarf_gpr_x7, riscv_dwarf::dwarf_gpr_x7,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x7},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(fp, x8),
        {riscv_dwarf::dwarf_gpr_x8, riscv_dwarf::dwarf_gpr_x8,
         LLDB_REGNUM_GENERIC_FP, LLDB_INVALID_REGNUM, gpr_x8},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(s1, x9),
        {riscv_dwarf::dwarf_gpr_x9, riscv_dwarf::dwarf_gpr_x9,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x9},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(a0, x10),
        {riscv_dwarf::dwarf_gpr_x10, riscv_dwarf::dwarf_gpr_x10,
         LLDB_REGNUM_GENERIC_ARG1, LLDB_INVALID_REGNUM, gpr_x10},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(a1, x11),
        {riscv_dwarf::dwarf_gpr_x11, riscv_dwarf::dwarf_gpr_x11,
         LLDB_REGNUM_GENERIC_ARG2, LLDB_INVALID_REGNUM, gpr_x11},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(a2, x12),
        {riscv_dwarf::dwarf_gpr_x12, riscv_dwarf::dwarf_gpr_x12,
         LLDB_REGNUM_GENERIC_ARG3, LLDB_INVALID_REGNUM, gpr_x12},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(a3, x13),
        {riscv_dwarf::dwarf_gpr_x13, riscv_dwarf::dwarf_gpr_x13,
         LLDB_REGNUM_GENERIC_ARG4, LLDB_INVALID_REGNUM, gpr_x13},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(a4, x14),
        {riscv_dwarf::dwarf_gpr_x14, riscv_dwarf::dwarf_gpr_x14,
         LLDB_REGNUM_GENERIC_ARG5, LLDB_INVALID_REGNUM, gpr_x14},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(a5, x15),
        {riscv_dwarf::dwarf_gpr_x15, riscv_dwarf::dwarf_gpr_x15,
         LLDB_REGNUM_GENERIC_ARG6, LLDB_INVALID_REGNUM, gpr_x15},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(a6, x16),
        {riscv_dwarf::dwarf_gpr_x16, riscv_dwarf::dwarf_gpr_x16,
         LLDB_REGNUM_GENERIC_ARG7, LLDB_INVALID_REGNUM, gpr_x16},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(a7, x17),
        {riscv_dwarf::dwarf_gpr_x17, riscv_dwarf::dwarf_gpr_x17,
         LLDB_REGNUM_GENERIC_ARG8, LLDB_INVALID_REGNUM, gpr_x17},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(s2, x18),
        {riscv_dwarf::dwarf_gpr_x18, riscv_dwarf::dwarf_gpr_x18,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x18},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(s3, x19),
        {riscv_dwarf::dwarf_gpr_x19, riscv_dwarf::dwarf_gpr_x19,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x19},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(s4, x20),
        {riscv_dwarf::dwarf_gpr_x20, riscv_dwarf::dwarf_gpr_x20,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x20},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(s5, x21),
        {riscv_dwarf::dwarf_gpr_x21, riscv_dwarf::dwarf_gpr_x21,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x21},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(s6, x22),
        {riscv_dwarf::dwarf_gpr_x22, riscv_dwarf::dwarf_gpr_x22,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x22},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(s7, x23),
        {riscv_dwarf::dwarf_gpr_x23, riscv_dwarf::dwarf_gpr_x23,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x23},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(s8, x24),
        {riscv_dwarf::dwarf_gpr_x24, riscv_dwarf::dwarf_gpr_x24,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x24},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(s9, x25),
        {riscv_dwarf::dwarf_gpr_x25, riscv_dwarf::dwarf_gpr_x25,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x25},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(s10, x26),
        {riscv_dwarf::dwarf_gpr_x26, riscv_dwarf::dwarf_gpr_x26,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x26},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(s11, x27),
        {riscv_dwarf::dwarf_gpr_x27, riscv_dwarf::dwarf_gpr_x27,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x27},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(t3, x28),
        {riscv_dwarf::dwarf_gpr_x28, riscv_dwarf::dwarf_gpr_x28,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x28},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(t4, x29),
        {riscv_dwarf::dwarf_gpr_x29, riscv_dwarf::dwarf_gpr_x29,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x29},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(t5, x30),
        {riscv_dwarf::dwarf_gpr_x30, riscv_dwarf::dwarf_gpr_x30,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x30},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR_ABI(t6, x31),
        {riscv_dwarf::dwarf_gpr_x31, riscv_dwarf::dwarf_gpr_x31,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gpr_x31},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_GPR(pc),
        {riscv_dwarf::dwarf_gpr_pc, riscv_dwarf::dwarf_gpr_pc,
         LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_REGNUM, gpr_pc},
        nullptr,
        nullptr,
        nullptr,
    },

    {
        DEFINE_FPU_ABI(ft0, f0),
        {riscv_dwarf::dwarf_fpr_f0, riscv_dwarf::dwarf_fpr_f0,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f0},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(ft1, f1),
        {riscv_dwarf::dwarf_fpr_f1, riscv_dwarf::dwarf_fpr_f1,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f1},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(ft2, f2),
        {riscv_dwarf::dwarf_fpr_f2, riscv_dwarf::dwarf_fpr_f2,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f2},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(ft3, f3),
        {riscv_dwarf::dwarf_fpr_f3, riscv_dwarf::dwarf_fpr_f3,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f3},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(ft4, f4),
        {riscv_dwarf::dwarf_fpr_f4, riscv_dwarf::dwarf_fpr_f4,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f4},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(ft5, f5),
        {riscv_dwarf::dwarf_fpr_f5, riscv_dwarf::dwarf_fpr_f5,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f5},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(ft6, f6),
        {riscv_dwarf::dwarf_fpr_f6, riscv_dwarf::dwarf_fpr_f6,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f6},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(ft7, f7),
        {riscv_dwarf::dwarf_fpr_f7, riscv_dwarf::dwarf_fpr_f7,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f7},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fs0, f8),
        {riscv_dwarf::dwarf_fpr_f8, riscv_dwarf::dwarf_fpr_f8,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f8},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fs1, f9),
        {riscv_dwarf::dwarf_fpr_f9, riscv_dwarf::dwarf_fpr_f9,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f9},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fa0, f10),
        {riscv_dwarf::dwarf_fpr_f10, riscv_dwarf::dwarf_fpr_f10,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f10},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fa1, f11),
        {riscv_dwarf::dwarf_fpr_f11, riscv_dwarf::dwarf_fpr_f11,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f11},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fa2, f12),
        {riscv_dwarf::dwarf_fpr_f12, riscv_dwarf::dwarf_fpr_f12,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f12},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fa3, f13),
        {riscv_dwarf::dwarf_fpr_f13, riscv_dwarf::dwarf_fpr_f13,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f13},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fa4, f14),
        {riscv_dwarf::dwarf_fpr_f14, riscv_dwarf::dwarf_fpr_f14,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f14},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fa5, f15),
        {riscv_dwarf::dwarf_fpr_f15, riscv_dwarf::dwarf_fpr_f15,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f15},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fa6, f16),
        {riscv_dwarf::dwarf_fpr_f16, riscv_dwarf::dwarf_fpr_f16,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f16},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fa7, f17),
        {riscv_dwarf::dwarf_fpr_f17, riscv_dwarf::dwarf_fpr_f17,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f17},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fs2, f18),
        {riscv_dwarf::dwarf_fpr_f18, riscv_dwarf::dwarf_fpr_f18,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f18},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fs3, f19),
        {riscv_dwarf::dwarf_fpr_f19, riscv_dwarf::dwarf_fpr_f19,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f19},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fs4, f20),
        {riscv_dwarf::dwarf_fpr_f20, riscv_dwarf::dwarf_fpr_f20,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f20},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fs5, f21),
        {riscv_dwarf::dwarf_fpr_f21, riscv_dwarf::dwarf_fpr_f21,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f21},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fs6, f22),
        {riscv_dwarf::dwarf_fpr_f22, riscv_dwarf::dwarf_fpr_f22,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f22},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fs7, f23),
        {riscv_dwarf::dwarf_fpr_f23, riscv_dwarf::dwarf_fpr_f23,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f23},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fs8, f24),
        {riscv_dwarf::dwarf_fpr_f24, riscv_dwarf::dwarf_fpr_f24,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f24},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fs9, f25),
        {riscv_dwarf::dwarf_fpr_f25, riscv_dwarf::dwarf_fpr_f25,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f25},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fs10, f26),
        {riscv_dwarf::dwarf_fpr_f26, riscv_dwarf::dwarf_fpr_f26,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f26},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(fs11, f27),
        {riscv_dwarf::dwarf_fpr_f27, riscv_dwarf::dwarf_fpr_f27,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f27},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(ft8, f28),
        {riscv_dwarf::dwarf_fpr_f28, riscv_dwarf::dwarf_fpr_f28,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f28},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(ft9, f29),
        {riscv_dwarf::dwarf_fpr_f29, riscv_dwarf::dwarf_fpr_f29,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f29},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(ft10, f30),
        {riscv_dwarf::dwarf_fpr_f30, riscv_dwarf::dwarf_fpr_f30,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f30},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU_ABI(ft11, f31),
        {riscv_dwarf::dwarf_fpr_f31, riscv_dwarf::dwarf_fpr_f31,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, fpr_f31},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_FPU(fcsr),
        {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, fpr_fcsr},
        nullptr,
        nullptr,
        nullptr,
    },

    {
        DEFINE_EXC(exception),
        {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, exc_exception},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_EXC(fsr),
        {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, exc_fsr},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        DEFINE_EXC(far),
        {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, exc_far},
        nullptr,
        nullptr,
        nullptr,
    },
    {"csr",
     nullptr,
     1024 * sizeof(uint32_t),
     0,
     eEncodingVector,
     eFormatVectorOfUInt32,
     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, csr_bank},
     nullptr,
     nullptr,
     nullptr}};

static size_t k_num_register_infos = std::size(g_register_infos);

RegisterContextDarwin_riscv32::RegisterContextDarwin_riscv32(
    Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContext(thread, concrete_frame_idx), gpr(), fpr(), exc() {
  uint32_t i;
  for (i = 0; i < kNumErrors; i++) {
    gpr_errs[i] = -1;
    fpr_errs[i] = -1;
    exc_errs[i] = -1;
    csr_errs[i] = -1;
  }
}

RegisterContextDarwin_riscv32::~RegisterContextDarwin_riscv32() = default;

void RegisterContextDarwin_riscv32::InvalidateAllRegisters() {
  InvalidateAllRegisterStates();
}

size_t RegisterContextDarwin_riscv32::GetRegisterCount() {
  assert(k_num_register_infos == k_num_registers);
  return k_num_registers;
}

const RegisterInfo *
RegisterContextDarwin_riscv32::GetRegisterInfoAtIndex(size_t reg) {
  assert(k_num_register_infos == k_num_registers);
  if (reg < k_num_registers)
    return &g_register_infos[reg];
  return nullptr;
}

size_t RegisterContextDarwin_riscv32::GetRegisterInfosCount() {
  return k_num_register_infos;
}

const RegisterInfo *RegisterContextDarwin_riscv32::GetRegisterInfos() {
  return g_register_infos;
}

// General purpose registers
static uint32_t g_gpr_regnums[] = {
    gpr_x0,  gpr_x1,  gpr_x2,  gpr_x3,  gpr_x4,  gpr_x5,  gpr_x6,
    gpr_x7,  gpr_x8,  gpr_x9,  gpr_x10, gpr_x11, gpr_x12, gpr_x13,
    gpr_x14, gpr_x15, gpr_x16, gpr_x17, gpr_x18, gpr_x19, gpr_x20,
    gpr_x21, gpr_x22, gpr_x23, gpr_x24, gpr_x25, gpr_x26, gpr_x27,
    gpr_x28, gpr_x29, gpr_x30, gpr_x31, gpr_pc};

// Floating point registers
static uint32_t g_fpr_regnums[] = {
    fpr_f0,  fpr_f1,  fpr_f2,  fpr_f3,  fpr_f4,  fpr_f5,  fpr_f6,
    fpr_f7,  fpr_f8,  fpr_f9,  fpr_f10, fpr_f11, fpr_f12, fpr_f13,
    fpr_f14, fpr_f15, fpr_f16, fpr_f17, fpr_f18, fpr_f19, fpr_f20,
    fpr_f21, fpr_f22, fpr_f23, fpr_f24, fpr_f25, fpr_f26, fpr_f27,
    fpr_f28, fpr_f29, fpr_f30, fpr_f31, fpr_fcsr};

// Exception registers

static uint32_t g_exc_regnums[] = {exc_exception, exc_fsr, exc_far};

// CSR bank registers
static uint32_t g_csr_regnums[] = {csr_bank};

// Number of registers in each register set
const size_t k_num_gpr_registers = std::size(g_gpr_regnums);
const size_t k_num_fpr_registers = std::size(g_fpr_regnums);
const size_t k_num_exc_registers = std::size(g_exc_regnums);
const size_t k_num_csr_registers = std::size(g_csr_regnums);

// Register set definitions. The first definitions at register set index of
// zero is for all registers, followed by other registers sets. The register
// information for the all register set need not be filled in.
static const RegisterSet g_reg_sets[] = {
    {
        "General Purpose Registers",
        "gpr",
        k_num_gpr_registers,
        g_gpr_regnums,
    },
    {"Floating Point Registers", "fpr", k_num_fpr_registers, g_fpr_regnums},
    {"Exception State Registers", "exc", k_num_exc_registers, g_exc_regnums},
    {"CSR register bank", "csr", k_num_csr_registers, g_csr_regnums}};

const size_t k_num_regsets = std::size(g_reg_sets);

size_t RegisterContextDarwin_riscv32::GetRegisterSetCount() {
  return k_num_regsets;
}

const RegisterSet *
RegisterContextDarwin_riscv32::GetRegisterSet(size_t reg_set) {
  if (reg_set < k_num_regsets)
    return &g_reg_sets[reg_set];
  return nullptr;
}

// Register information definitions for 32 bit riscv32.
int RegisterContextDarwin_riscv32::GetSetForNativeRegNum(int reg_num) {
  if (reg_num < fpr_f0)
    return GPRRegSet;
  else if (reg_num < exc_exception)
    return FPURegSet;
  else if (reg_num < csr_bank)
    return EXCRegSet;
  else if (reg_num < k_num_registers)
    return CSRRegSet;
  return -1;
}

void RegisterContextDarwin_riscv32::LogGPR(Log *log, const char *title) {
  if (log) {
    if (title)
      LLDB_LOGF(log, "%s", title);
    for (uint32_t i = 0; i < k_num_gpr_registers; i++) {
      uint32_t reg = gpr_x0 + i;
      LLDB_LOGF(log, "%12s = 0x%4.4x", g_register_infos[reg].name,
                (&gpr.x0)[reg]);
    }
  }
}

int RegisterContextDarwin_riscv32::ReadGPR(bool force) {
  int set = GPRRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadGPR(GetThreadID(), set, gpr));
  }
  return GetError(set, Read);
}

int RegisterContextDarwin_riscv32::ReadFPU(bool force) {
  int set = FPURegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadFPU(GetThreadID(), set, fpr));
  }
  return GetError(set, Read);
}

int RegisterContextDarwin_riscv32::ReadEXC(bool force) {
  int set = EXCRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadEXC(GetThreadID(), set, exc));
  }
  return GetError(set, Read);
}

int RegisterContextDarwin_riscv32::ReadCSR(bool force) {
  int set = CSRRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadCSR(GetThreadID(), set, csr));
  }
  return GetError(set, Read);
}

int RegisterContextDarwin_riscv32::WriteGPR() {
  int set = GPRRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteGPR(GetThreadID(), set, gpr));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_riscv32::WriteFPU() {
  int set = FPURegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteFPU(GetThreadID(), set, fpr));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_riscv32::WriteEXC() {
  int set = EXCRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteEXC(GetThreadID(), set, exc));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_riscv32::WriteCSR() {
  int set = CSRRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteCSR(GetThreadID(), set, csr));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_riscv32::ReadRegisterSet(uint32_t set, bool force) {
  switch (set) {
  case GPRRegSet:
    return ReadGPR(force);
  case FPURegSet:
    return ReadFPU(force);
  case EXCRegSet:
    return ReadEXC(force);
  case CSRRegSet:
    return ReadCSR(force);
  default:
    break;
  }
  return -1;
}

int RegisterContextDarwin_riscv32::WriteRegisterSet(uint32_t set) {
  // Make sure we have a valid context to set.
  if (RegisterSetIsCached(set)) {
    switch (set) {
    case GPRRegSet:
      return WriteGPR();
    case FPURegSet:
      return WriteFPU();
    case EXCRegSet:
      return WriteEXC();
    case CSRRegSet:
      return WriteCSR();
    default:
      break;
    }
  }
  return -1;
}

bool RegisterContextDarwin_riscv32::ReadRegister(const RegisterInfo *reg_info,
                                                 RegisterValue &value) {
  const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
  int set = RegisterContextDarwin_riscv32::GetSetForNativeRegNum(reg);

  if (set == -1)
    return false;

  if (ReadRegisterSet(set, false) != 0)
    return false;

  switch (reg) {
  case gpr_x0:
  case gpr_x1:
  case gpr_x2:
  case gpr_x3:
  case gpr_x4:
  case gpr_x5:
  case gpr_x6:
  case gpr_x7:
  case gpr_x8:
  case gpr_x9:
  case gpr_x10:
  case gpr_x11:
  case gpr_x12:
  case gpr_x13:
  case gpr_x14:
  case gpr_x15:
  case gpr_x16:
  case gpr_x17:
  case gpr_x18:
  case gpr_x19:
  case gpr_x20:
  case gpr_x21:
  case gpr_x22:
  case gpr_x23:
  case gpr_x24:
  case gpr_x25:
  case gpr_x26:
  case gpr_x27:
  case gpr_x28:
  case gpr_x29:
  case gpr_x30:
  case gpr_x31:
  case gpr_pc:
    value = (&gpr.x0)[reg - gpr_x0];
    break;

  case fpr_f0:
  case fpr_f1:
  case fpr_f2:
  case fpr_f3:
  case fpr_f4:
  case fpr_f5:
  case fpr_f6:
  case fpr_f7:
  case fpr_f8:
  case fpr_f9:
  case fpr_f10:
  case fpr_f11:
  case fpr_f12:
  case fpr_f13:
  case fpr_f14:
  case fpr_f15:
  case fpr_f16:
  case fpr_f17:
  case fpr_f18:
  case fpr_f19:
  case fpr_f20:
  case fpr_f21:
  case fpr_f22:
  case fpr_f23:
  case fpr_f24:
  case fpr_f25:
  case fpr_f26:
  case fpr_f27:
  case fpr_f28:
  case fpr_f29:
  case fpr_f30:
  case fpr_f31:
  case fpr_fcsr:
    value = (&fpr.f0)[reg - fpr_f0];
    break;

  case exc_exception:
    value = exc.exception;
    break;

  case exc_fsr:
    value = exc.fsr;
    break;

  case exc_far:
    value = exc.far;
    break;

  case csr_bank:
    // These values don't fit into scalar types,
    // RegisterContext::ReadRegisterBytes() must be used for these registers
    //::memcpy (reg_value.value.vector.uint8, fpu.stmm[reg - fpu_stmm0].bytes,
    // 10);

    // AArch64 copies NEON registers with
    // value.SetBytes(csr.bytes, reg_info->byte_size,
    //                endian::InlHostByteOrder());
    return false;

  default:
    return false;
  }
  return true;
}

bool RegisterContextDarwin_riscv32::WriteRegister(const RegisterInfo *reg_info,
                                                  const RegisterValue &value) {
  const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
  int set = GetSetForNativeRegNum(reg);

  if (set == -1)
    return false;

  if (ReadRegisterSet(set, false) != 0)
    return false;

  switch (reg) {
  case gpr_x0:
  case gpr_x1:
  case gpr_x2:
  case gpr_x3:
  case gpr_x4:
  case gpr_x5:
  case gpr_x6:
  case gpr_x7:
  case gpr_x8:
  case gpr_x9:
  case gpr_x10:
  case gpr_x11:
  case gpr_x12:
  case gpr_x13:
  case gpr_x14:
  case gpr_x15:
  case gpr_x16:
  case gpr_x17:
  case gpr_x18:
  case gpr_x19:
  case gpr_x20:
  case gpr_x21:
  case gpr_x22:
  case gpr_x23:
  case gpr_x24:
  case gpr_x25:
  case gpr_x26:
  case gpr_x27:
  case gpr_x28:
  case gpr_x29:
  case gpr_x30:
  case gpr_x31:
  case gpr_pc:
    (&gpr.x0)[reg - gpr_x0] = value.GetAsUInt32();
    break;

  case fpr_f0:
  case fpr_f1:
  case fpr_f2:
  case fpr_f3:
  case fpr_f4:
  case fpr_f5:
  case fpr_f6:
  case fpr_f7:
  case fpr_f8:
  case fpr_f9:
  case fpr_f10:
  case fpr_f11:
  case fpr_f12:
  case fpr_f13:
  case fpr_f14:
  case fpr_f15:
  case fpr_f16:
  case fpr_f17:
  case fpr_f18:
  case fpr_f19:
  case fpr_f20:
  case fpr_f21:
  case fpr_f22:
  case fpr_f23:
  case fpr_f24:
  case fpr_f25:
  case fpr_f26:
  case fpr_f27:
  case fpr_f28:
  case fpr_f29:
  case fpr_f30:
  case fpr_f31:
  case fpr_fcsr:
    (&fpr.f0)[reg - fpr_f0] = value.GetAsUInt32();
    break;

  case exc_exception:
    exc.exception = value.GetAsUInt32();
    break;

  case exc_fsr:
    exc.fsr = value.GetAsUInt32();
    break;

  case exc_far:
    exc.far = value.GetAsUInt32();
    break;

  case csr_bank:
    // These values don't fit into scalar types,
    // RegisterContext::ReadRegisterBytes() must be used for these registers
    //::memcpy(csr.bytes, value.GetBytes(),
    // value.GetByteSize());
    return false;

  default:
    return false;
  }
  return WriteRegisterSet(set) == 0;
}

bool RegisterContextDarwin_riscv32::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  data_sp = std::make_shared<DataBufferHeap>(REG_CONTEXT_SIZE, 0);
  if (ReadGPR(false) == 0 && ReadFPU(false) == 0 && ReadEXC(false) == 0 &&
      ReadCSR(false) == 0) {
    uint8_t *dst = data_sp->GetBytes();
    ::memcpy(dst, &gpr, sizeof(gpr));
    dst += sizeof(gpr);

    ::memcpy(dst, &fpr, sizeof(fpr));
    dst += sizeof(gpr);

    ::memcpy(dst, &exc, sizeof(exc));
    return true;

    ::memcpy(dst, &csr, sizeof(csr));
    return true;
  }
  return false;
}

bool RegisterContextDarwin_riscv32::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (data_sp && data_sp->GetByteSize() == REG_CONTEXT_SIZE) {
    const uint8_t *src = data_sp->GetBytes();
    ::memcpy(&gpr, src, sizeof(gpr));
    src += sizeof(gpr);

    ::memcpy(&fpr, src, sizeof(fpr));
    src += sizeof(fpr);

    ::memcpy(&exc, src, sizeof(exc));
    src += sizeof(exc);

    ::memcpy(&csr, src, sizeof(csr));
    uint32_t success_count = 0;

    if (WriteGPR() == 0)
      ++success_count;
    if (WriteFPU() == 0)
      ++success_count;
    if (WriteEXC() == 0)
      ++success_count;
    if (WriteCSR() == 0)
      ++success_count;
    return success_count == 3;
  }
  return false;
}

uint32_t RegisterContextDarwin_riscv32::ConvertRegisterKindToRegisterNumber(
    lldb::RegisterKind kind, uint32_t reg) {
  if (kind == eRegisterKindGeneric) {
    switch (reg) {
    case LLDB_REGNUM_GENERIC_PC:
      return gpr_pc;
    case LLDB_REGNUM_GENERIC_SP:
      return gpr_x2;
    case LLDB_REGNUM_GENERIC_FP:
      return gpr_x8;
    case LLDB_REGNUM_GENERIC_RA:
      return gpr_x1;
    default:
      break;
    }
  } else if (kind == eRegisterKindEHFrame || kind == eRegisterKindDWARF) {
    switch (reg) {
    case riscv_dwarf::dwarf_gpr_x0:
    case riscv_dwarf::dwarf_gpr_x1:
    case riscv_dwarf::dwarf_gpr_x2:
    case riscv_dwarf::dwarf_gpr_x3:
    case riscv_dwarf::dwarf_gpr_x4:
    case riscv_dwarf::dwarf_gpr_x5:
    case riscv_dwarf::dwarf_gpr_x6:
    case riscv_dwarf::dwarf_gpr_x7:
    case riscv_dwarf::dwarf_gpr_x8:
    case riscv_dwarf::dwarf_gpr_x9:
    case riscv_dwarf::dwarf_gpr_x10:
    case riscv_dwarf::dwarf_gpr_x11:
    case riscv_dwarf::dwarf_gpr_x12:
    case riscv_dwarf::dwarf_gpr_x13:
    case riscv_dwarf::dwarf_gpr_x14:
    case riscv_dwarf::dwarf_gpr_x15:
    case riscv_dwarf::dwarf_gpr_x16:
    case riscv_dwarf::dwarf_gpr_x17:
    case riscv_dwarf::dwarf_gpr_x18:
    case riscv_dwarf::dwarf_gpr_x19:
    case riscv_dwarf::dwarf_gpr_x20:
    case riscv_dwarf::dwarf_gpr_x21:
    case riscv_dwarf::dwarf_gpr_x22:
    case riscv_dwarf::dwarf_gpr_x23:
    case riscv_dwarf::dwarf_gpr_x24:
    case riscv_dwarf::dwarf_gpr_x25:
    case riscv_dwarf::dwarf_gpr_x26:
    case riscv_dwarf::dwarf_gpr_x27:
    case riscv_dwarf::dwarf_gpr_x28:
    case riscv_dwarf::dwarf_gpr_x29:
    case riscv_dwarf::dwarf_gpr_x30:
    case riscv_dwarf::dwarf_gpr_x31:
      return gpr_x0 + (reg - riscv_dwarf::dwarf_gpr_x0);

    case riscv_dwarf::dwarf_fpr_f0:
    case riscv_dwarf::dwarf_fpr_f1:
    case riscv_dwarf::dwarf_fpr_f2:
    case riscv_dwarf::dwarf_fpr_f3:
    case riscv_dwarf::dwarf_fpr_f4:
    case riscv_dwarf::dwarf_fpr_f5:
    case riscv_dwarf::dwarf_fpr_f6:
    case riscv_dwarf::dwarf_fpr_f7:
    case riscv_dwarf::dwarf_fpr_f8:
    case riscv_dwarf::dwarf_fpr_f9:
    case riscv_dwarf::dwarf_fpr_f10:
    case riscv_dwarf::dwarf_fpr_f11:
    case riscv_dwarf::dwarf_fpr_f12:
    case riscv_dwarf::dwarf_fpr_f13:
    case riscv_dwarf::dwarf_fpr_f14:
    case riscv_dwarf::dwarf_fpr_f15:
    case riscv_dwarf::dwarf_fpr_f16:
    case riscv_dwarf::dwarf_fpr_f17:
    case riscv_dwarf::dwarf_fpr_f18:
    case riscv_dwarf::dwarf_fpr_f19:
    case riscv_dwarf::dwarf_fpr_f20:
    case riscv_dwarf::dwarf_fpr_f21:
    case riscv_dwarf::dwarf_fpr_f22:
    case riscv_dwarf::dwarf_fpr_f23:
    case riscv_dwarf::dwarf_fpr_f24:
    case riscv_dwarf::dwarf_fpr_f25:
    case riscv_dwarf::dwarf_fpr_f26:
    case riscv_dwarf::dwarf_fpr_f27:
    case riscv_dwarf::dwarf_fpr_f28:
    case riscv_dwarf::dwarf_fpr_f29:
    case riscv_dwarf::dwarf_fpr_f30:
    case riscv_dwarf::dwarf_fpr_f31:
      return fpr_f0 + (reg - riscv_dwarf::dwarf_fpr_f0);

    default:
      break;
    }
  } else if (kind == eRegisterKindLLDB) {
    return reg;
  }
  return LLDB_INVALID_REGNUM;
}
