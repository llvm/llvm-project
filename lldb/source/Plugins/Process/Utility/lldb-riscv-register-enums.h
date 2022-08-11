//===-- lldb-riscv-register-enums.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_LLDB_RISCV_REGISTER_ENUMS_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_LLDB_RISCV_REGISTER_ENUMS_H

// LLDB register codes (e.g. RegisterKind == eRegisterKindLLDB)

// Internal codes for all riscv registers.
enum {
  // The same order as user_regs_struct in <asm/ptrace.h>
  // note: these enum values are used as byte_offset
  gpr_first_riscv = 0,
  gpr_pc_riscv = gpr_first_riscv,
  gpr_x1_riscv,
  gpr_x2_riscv,
  gpr_x3_riscv,
  gpr_x4_riscv,
  gpr_x5_riscv,
  gpr_x6_riscv,
  gpr_x7_riscv,
  gpr_x8_riscv,
  gpr_x9_riscv,
  gpr_x10_riscv,
  gpr_x11_riscv,
  gpr_x12_riscv,
  gpr_x13_riscv,
  gpr_x14_riscv,
  gpr_x15_riscv,
  gpr_x16_riscv,
  gpr_x17_riscv,
  gpr_x18_riscv,
  gpr_x19_riscv,
  gpr_x20_riscv,
  gpr_x21_riscv,
  gpr_x22_riscv,
  gpr_x23_riscv,
  gpr_x24_riscv,
  gpr_x25_riscv,
  gpr_x26_riscv,
  gpr_x27_riscv,
  gpr_x28_riscv,
  gpr_x29_riscv,
  gpr_x30_riscv,
  gpr_x31_riscv,
  gpr_x0_riscv,
  gpr_last_riscv = gpr_x0_riscv,
  gpr_ra_riscv = gpr_x1_riscv,
  gpr_sp_riscv = gpr_x2_riscv,
  gpr_fp_riscv = gpr_x8_riscv,

  fpr_first_riscv = 33,
  fpr_f0_riscv = fpr_first_riscv,
  fpr_f1_riscv,
  fpr_f2_riscv,
  fpr_f3_riscv,
  fpr_f4_riscv,
  fpr_f5_riscv,
  fpr_f6_riscv,
  fpr_f7_riscv,
  fpr_f8_riscv,
  fpr_f9_riscv,
  fpr_f10_riscv,
  fpr_f11_riscv,
  fpr_f12_riscv,
  fpr_f13_riscv,
  fpr_f14_riscv,
  fpr_f15_riscv,
  fpr_f16_riscv,
  fpr_f17_riscv,
  fpr_f18_riscv,
  fpr_f19_riscv,
  fpr_f20_riscv,
  fpr_f21_riscv,
  fpr_f22_riscv,
  fpr_f23_riscv,
  fpr_f24_riscv,
  fpr_f25_riscv,
  fpr_f26_riscv,
  fpr_f27_riscv,
  fpr_f28_riscv,
  fpr_f29_riscv,
  fpr_f30_riscv,
  fpr_f31_riscv,

  fpr_fcsr_riscv,
  fpr_last_riscv = fpr_fcsr_riscv,

  k_num_registers_riscv
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_LLDB_RISCV_REGISTER_ENUMS_H
