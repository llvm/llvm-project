//===-- PPC64LE_ehframe_Registers.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_PPC64LE_ehframe_Registers_h_
#define utility_PPC64LE_ehframe_Registers_h_

// The register numbers used in the eh_frame unwind information.
// Should be the same as DWARF register numbers.

namespace ppc64le_ehframe {

enum {
  r0 = 0,
  r1,
  r2,
  r3,
  r4,
  r5,
  r6,
  r7,
  r8,
  r9,
  r10,
  r11,
  r12,
  r13,
  r14,
  r15,
  r16,
  r17,
  r18,
  r19,
  r20,
  r21,
  r22,
  r23,
  r24,
  r25,
  r26,
  r27,
  r28,
  r29,
  r30,
  r31,
  lr = 65,
  ctr,
  cr = 68,
  xer = 76,
  pc,
  softe,
  trap,
  origr3,
  msr,
};
}

#endif // utility_PPC64LE_ehframe_Registers_h_
