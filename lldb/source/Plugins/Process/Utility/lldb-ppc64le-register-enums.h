//===-- lldb-ppc64le-register-enums.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_ppc64le_register_enums_h
#define lldb_ppc64le_register_enums_h

// LLDB register codes (e.g. RegisterKind == eRegisterKindLLDB)

// ---------------------------------------------------------------------------
// Internal codes for all ppc64le registers.
// ---------------------------------------------------------------------------
enum {
  k_first_gpr_ppc64le,
  gpr_r0_ppc64le = k_first_gpr_ppc64le,
  gpr_r1_ppc64le,
  gpr_r2_ppc64le,
  gpr_r3_ppc64le,
  gpr_r4_ppc64le,
  gpr_r5_ppc64le,
  gpr_r6_ppc64le,
  gpr_r7_ppc64le,
  gpr_r8_ppc64le,
  gpr_r9_ppc64le,
  gpr_r10_ppc64le,
  gpr_r11_ppc64le,
  gpr_r12_ppc64le,
  gpr_r13_ppc64le,
  gpr_r14_ppc64le,
  gpr_r15_ppc64le,
  gpr_r16_ppc64le,
  gpr_r17_ppc64le,
  gpr_r18_ppc64le,
  gpr_r19_ppc64le,
  gpr_r20_ppc64le,
  gpr_r21_ppc64le,
  gpr_r22_ppc64le,
  gpr_r23_ppc64le,
  gpr_r24_ppc64le,
  gpr_r25_ppc64le,
  gpr_r26_ppc64le,
  gpr_r27_ppc64le,
  gpr_r28_ppc64le,
  gpr_r29_ppc64le,
  gpr_r30_ppc64le,
  gpr_r31_ppc64le,
  gpr_pc_ppc64le,
  gpr_msr_ppc64le,
  gpr_origr3_ppc64le,
  gpr_ctr_ppc64le,
  gpr_lr_ppc64le,
  gpr_xer_ppc64le,
  gpr_cr_ppc64le,
  gpr_softe_ppc64le,
  gpr_trap_ppc64le,
  k_last_gpr_ppc64le = gpr_trap_ppc64le,

  k_num_registers_ppc64le,
  k_num_gpr_registers_ppc64le = k_last_gpr_ppc64le - k_first_gpr_ppc64le + 1,
};

#endif // #ifndef lldb_ppc64le_register_enums_h
