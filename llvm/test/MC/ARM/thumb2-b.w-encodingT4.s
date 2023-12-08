@ RUN: llvm-mc -triple=thumbv7-apple-darwin -mcpu=cortex-a8 -show-encoding < %s | FileCheck %s
  .syntax unified
  .globl _func
.thumb_func _foo
.space 0x37c6
_foo:
@------------------------------------------------------------------------------
@ B (thumb2 b.w encoding T4) rdar://12585795
@------------------------------------------------------------------------------
        b.w   0x3680c

@ CHECK: b.w	#223244                    @ encoding: [0x36,0xf0,0x06,0xbc]
