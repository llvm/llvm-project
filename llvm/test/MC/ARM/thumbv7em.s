@ RUN: llvm-mc -triple=thumbv7em -show-encoding < %s | FileCheck %s
@ RUN: not llvm-mc -triple=thumbv7m -show-encoding 2>&1 < %s | FileCheck --check-prefix=CHECK-V7M %s

  .syntax unified

@ Check that the assembler can handle the documented syntax from the ARM ARM.
@ These tests test instruction encodings specific to ARMv7E-M.

@------------------------------------------------------------------------------
@ MSR
@------------------------------------------------------------------------------

        msr  apsr_g, r0
        msr  apsr_nzcvqg, r0
        msr  iapsr_g, r0
        msr  iapsr_nzcvqg, r0
        msr  eapsr_g, r0
        msr  eapsr_nzcvqg, r0
        msr  xpsr_g, r0
        msr  xpsr_nzcvqg, r0

@ CHECK: msr	apsr_g, r0              @ encoding: [0x80,0xf3,0x00,0x84]
@ CHECK: msr	apsr_nzcvqg, r0         @ encoding: [0x80,0xf3,0x00,0x8c]
@ CHECK: msr	iapsr_g, r0             @ encoding: [0x80,0xf3,0x01,0x84]
@ CHECK: msr	iapsr_nzcvqg, r0        @ encoding: [0x80,0xf3,0x01,0x8c]
@ CHECK: msr	eapsr_g, r0             @ encoding: [0x80,0xf3,0x02,0x84]
@ CHECK: msr	eapsr_nzcvqg, r0        @ encoding: [0x80,0xf3,0x02,0x8c]
@ CHECK: msr	xpsr_g, r0              @ encoding: [0x80,0xf3,0x03,0x84]
@ CHECK: msr	xpsr_nzcvqg, r0         @ encoding: [0x80,0xf3,0x03,0x8c]
@ CHECK-V7M: error: invalid operand for instruction
@ CHECK-V7M-NEXT:         msr  apsr_g, r0
@ CHECK-V7M-NEXT:              ^
@ CHECK-V7M: error: invalid operand for instruction
@ CHECK-V7M-NEXT:         msr  apsr_nzcvqg, r0
@ CHECK-V7M-NEXT:              ^
@ CHECK-V7M: error: invalid operand for instruction
@ CHECK-V7M-NEXT:         msr  iapsr_g, r0
@ CHECK-V7M-NEXT:              ^
@ CHECK-V7M: error: invalid operand for instruction
@ CHECK-V7M-NEXT:         msr  iapsr_nzcvqg, r0
@ CHECK-V7M-NEXT:              ^
@ CHECK-V7M: error: invalid operand for instruction
@ CHECK-V7M-NEXT:         msr  eapsr_g, r0
@ CHECK-V7M-NEXT:              ^
@ CHECK-V7M: error: invalid operand for instruction
@ CHECK-V7M-NEXT:         msr  eapsr_nzcvqg, r0
@ CHECK-V7M-NEXT:              ^
@ CHECK-V7M: error: invalid operand for instruction
@ CHECK-V7M-NEXT:         msr  xpsr_g, r0
@ CHECK-V7M-NEXT:              ^
@ CHECK-V7M: error: invalid operand for instruction
@ CHECK-V7M-NEXT:         msr  xpsr_nzcvqg, r0
@ CHECK-V7M-NEXT:              ^
