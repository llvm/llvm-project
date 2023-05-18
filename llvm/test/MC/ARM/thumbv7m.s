@ RUN: llvm-mc -triple=thumbv7m-apple-darwin -show-encoding < %s | FileCheck %s
@ RUN: not llvm-mc -triple=thumbv6 -show-encoding 2>&1 < %s | FileCheck %s --check-prefix=CHECK-V6M
  .syntax unified
  .globl _func

@ Check that the assembler can handle the documented syntax from the ARM ARM.
@ These tests test instruction encodings specific to ARMv7m.

@------------------------------------------------------------------------------
@ MRS
@------------------------------------------------------------------------------

        mrs  r0, basepri
        mrs  r0, basepri_max
        mrs  r0, faultmask

@ CHECK: mrs	r0, basepri             @ encoding: [0xef,0xf3,0x11,0x80]
@ CHECK: mrs	r0, basepri_max         @ encoding: [0xef,0xf3,0x12,0x80]
@ CHECK: mrs	r0, faultmask           @ encoding: [0xef,0xf3,0x13,0x80]

@------------------------------------------------------------------------------
@ MSR
@------------------------------------------------------------------------------

        msr  basepri, r0
        msr  basepri_max, r0
        msr  faultmask, r0

@ CHECK: msr	basepri, r0             @ encoding: [0x80,0xf3,0x11,0x88]
@ CHECK: msr	basepri_max, r0         @ encoding: [0x80,0xf3,0x12,0x88]
@ CHECK: msr	faultmask, r0           @ encoding: [0x80,0xf3,0x13,0x88]

@ CHECK-V6M: error: invalid instruction
@ CHECK-V6M-NEXT: mrs r0, basepri
@ CHECK-V6M: error: invalid instruction
@ CHECK-V6M-NEXT: mrs r0, basepri_max
@ CHECK-V6M: error: invalid instruction
@ CHECK-V6M-NEXT: mrs r0, faultmask
@ CHECK-V6M: error: invalid instruction
@ CHECK-V6M-NEXT: msr basepri, r0
@ CHECK-V6M: error: invalid instruction
@ CHECK-V6M-NEXT: msr basepri_max, r0
@ CHECK-V6M: error: invalid instruction
@ CHECK-V6M-NEXT: msr faultmask, r0

