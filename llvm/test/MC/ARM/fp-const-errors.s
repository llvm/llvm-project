@ RUN: not llvm-mc -mcpu=cortex-a8 -triple armv7-none-linux-gnueabi < %s 2>&1 | FileCheck %s

@ Test for floating point constants that are out of the 8-bit encoded value range
vmov.f32 s2, #32.0
@ CHECK: error: invalid operand for instruction

vmov.f64 d2, #32.0
@ CHECK: error: invalid operand for instruction

@ Test that vmov.f instructions do not accept an 8-bit encoded float as an operand
vmov.f32 s1, #0x70
@ CHECK: error: invalid floating point immediate

vmov.f64 d2, #0x70
@ CHECK: error: invalid floating point immediate

@ Test that fconst instructions do not accept a float constant as an operand
fconsts s1, #1.0
@ CHECK: error: invalid floating point immediate

fconstd d2, #1.0
@ CHECK: error: invalid floating point immediate

vmov.i64 d0, 0x8000000000000000
@ CHECK: error: invalid operand for instruction
