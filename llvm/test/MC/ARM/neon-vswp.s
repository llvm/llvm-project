@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7-apple-darwin -show-encoding < %s | FileCheck %s

vswp d1, d2
vswp q1, q2

@ CHECK: vswp	d1, d2                  @ encoding: [0x02,0x10,0xb2,0xf3]
@ CHECK: vswp	q1, q2                  @ encoding: [0x44,0x20,0xb2,0xf3]
