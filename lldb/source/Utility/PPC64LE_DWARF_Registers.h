//===-- PPC64LE_DWARF_Registers.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_PPC64LE_DWARF_Registers_h_
#define utility_PPC64LE_DWARF_Registers_h_

#include "lldb/lldb-private.h"

namespace ppc64le_dwarf {

enum {
  dwarf_r0_ppc64le = 0,
  dwarf_r1_ppc64le,
  dwarf_r2_ppc64le,
  dwarf_r3_ppc64le,
  dwarf_r4_ppc64le,
  dwarf_r5_ppc64le,
  dwarf_r6_ppc64le,
  dwarf_r7_ppc64le,
  dwarf_r8_ppc64le,
  dwarf_r9_ppc64le,
  dwarf_r10_ppc64le,
  dwarf_r11_ppc64le,
  dwarf_r12_ppc64le,
  dwarf_r13_ppc64le,
  dwarf_r14_ppc64le,
  dwarf_r15_ppc64le,
  dwarf_r16_ppc64le,
  dwarf_r17_ppc64le,
  dwarf_r18_ppc64le,
  dwarf_r19_ppc64le,
  dwarf_r20_ppc64le,
  dwarf_r21_ppc64le,
  dwarf_r22_ppc64le,
  dwarf_r23_ppc64le,
  dwarf_r24_ppc64le,
  dwarf_r25_ppc64le,
  dwarf_r26_ppc64le,
  dwarf_r27_ppc64le,
  dwarf_r28_ppc64le,
  dwarf_r29_ppc64le,
  dwarf_r30_ppc64le,
  dwarf_r31_ppc64le,
  dwarf_lr_ppc64le = 65,
  dwarf_ctr_ppc64le,
  dwarf_cr_ppc64le = 68,
  dwarf_xer_ppc64le = 76,
  dwarf_pc_ppc64le,
  dwarf_softe_ppc64le,
  dwarf_trap_ppc64le,
  dwarf_origr3_ppc64le,
  dwarf_msr_ppc64le,
};

} // namespace ppc64le_dwarf

#endif // utility_PPC64LE_DWARF_Registers_h_
