@ RUN: llvm-mc -triple=armv7-apple-darwin -show-encoding < %s | FileCheck %s
  .syntax unified
  .globl _func

_func:
@ CHECK: _func:
        it eq
        moveq r2, r3
@ 'it' is parsed but not encoded.
@ CHECK-NOT: it
@ CHECK: moveq	r2, r3          @ encoding: [0x03,0x20,0xa0,0x01]
