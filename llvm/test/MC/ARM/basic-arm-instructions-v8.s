@ New ARMv8 A32 encodings

@ RUN: llvm-mc -triple armv8 -show-encoding < %s | FileCheck %s --check-prefix=CHECK-V8
@ RUN: not llvm-mc -triple armv7 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V7

@ HLT
        hlt  #0
        hlt  #65535
@ CHECK-V8: hlt  #0                       @ encoding: [0x70,0x00,0x00,0xe1]
@ CHECK-V8: hlt  #65535                   @ encoding: [0x7f,0xff,0x0f,0xe1]
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8

@ AL condition code allowable
        hltal  #0
@ CHECK-V8: hlt  #0                       @ encoding: [0x70,0x00,0x00,0xe1]
@ CHECK-V7: error: instruction requires: armv8

@------------------------------------------------------------------------------
@ DMB (v8 barriers)
@------------------------------------------------------------------------------
        dmb ishld
        dmb oshld
        dmb nshld
        dmb ld

@ CHECK-V8: dmb ishld @ encoding: [0x59,0xf0,0x7f,0xf5]
@ CHECK-V8: dmb oshld @ encoding: [0x51,0xf0,0x7f,0xf5]
@ CHECK-V8: dmb nshld @ encoding: [0x55,0xf0,0x7f,0xf5]
@ CHECK-V8: dmb ld @ encoding: [0x5d,0xf0,0x7f,0xf5]
@ CHECK-V7: error: invalid operand for instruction
@ CHECK-V7: error: invalid operand for instruction
@ CHECK-V7: error: invalid operand for instruction
@ CHECK-V7: error: invalid operand for instruction

@------------------------------------------------------------------------------
@ DSB (v8 barriers)
@------------------------------------------------------------------------------
        dsb ishld
        dsb oshld
        dsb nshld
        dsb ld

@ CHECK-V8: dsb ishld @ encoding: [0x49,0xf0,0x7f,0xf5]
@ CHECK-V8: dsb oshld @ encoding: [0x41,0xf0,0x7f,0xf5]
@ CHECK-V8: dsb nshld @ encoding: [0x45,0xf0,0x7f,0xf5]
@ CHECK-V8: dsb ld @ encoding: [0x4d,0xf0,0x7f,0xf5]
@ CHECK-V7: error: invalid operand for instruction
@ CHECK-V7: error: invalid operand for instruction
@ CHECK-V7: error: invalid operand for instruction
@ CHECK-V7: error: invalid operand for instruction

@------------------------------------------------------------------------------
@ SEVL
@------------------------------------------------------------------------------
        sevl

@ CHECK-V8: sevl @ encoding: [0x05,0xf0,0x20,0xe3]
@ CHECK-V7: instruction requires: armv8
