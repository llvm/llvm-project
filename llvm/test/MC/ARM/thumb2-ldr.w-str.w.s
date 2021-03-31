@ RUN: not llvm-mc -triple=thumbv7-unknown-linux-gnueabi -arm-implicit-it=thumb -show-encoding < %s 2>&1 | FileCheck %s
.syntax unified

@ Note: The error stream for XFAIL needs to get checked first.

ldr.w r1, [r1, #-4]!
ldr.w r1, [r0, #256]!
ldr.w r1, [r0, #-256]!
ldr.w r1, [pc, #-4]!
ldr.w r1, [r1], #4
ldr.w r0, [r0], #4
ldr.w r0, [r1], #256
ldr.w r0, [r1], #-256
str.w r0, [r0, #-4]!
str.w pc, [r0, #-4]!
str.w r1, [pc, #-4]!
str.w r1, [r2, #256]!
str.w r1, [r2, #-256]!
str.w r0, [r0], #4
str.w pc, [r0], #4
str.w r1, [r0], #256
str.w r1, [r0], #-256

@@ XFAIL

@ CHECK: error: destination register and base register can't be identical
@ CHECK-NEXT: ldr.w r1, [r1, #-4]!
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK-NEXT: ldr.w r1, [r0, #256]!
@ CHECK: note: invalid operand for instruction
@ CHECK: note: too many operands for instruction
@ CHECK: error: invalid operand for instruction
@ CHECK-NEXT: ldr.w r1, [r0, #-256]!
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK-NEXT: ldr.w r1, [pc, #-4]!
@ CHECK: note: invalid operand for instruction
@ CHECK: note: too many operands for instruction
@ CHECK: error: destination register and base register can't be identical
@ CHECK-NEXT: ldr.w r1, [r1], #4
@ CHECK: error: destination register and base register can't be identical
@ CHECK-NEXT: ldr.w r0, [r0], #4
@ CHECK: error: operand must be in range [-255, 255]
@ CHECK-NEXT: ldr.w r0, [r1], #256
@ CHECK: error: operand must be in range [-255, 255]
@ CHECK-NEXT: ldr.w r0, [r1], #-256
@ CHECK: error: destination register and base register can't be identical
@ CHECK-NEXT: str.w r0, [r0, #-4]!
@ CHECK: error: operand must be a register in range [r0, r14]
@ CHECK-NEXT: str.w pc, [r0, #-4]!
@ CHECK: error: invalid operand for instruction
@ CHECK-NEXT: str.w r1, [pc, #-4]!
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK-NEXT: str.w r1, [r2, #256]!
@ CHECK: note: invalid operand for instruction
@ CHECK: note: too many operands for instruction
@ CHECK: error: invalid operand for instruction
@ CHECK-NEXT: str.w r1, [r2, #-256]!
@ CHECK: error: destination register and base register can't be identical
@ CHECK-NEXT: str.w r0, [r0], #4
@ CHECK: error: operand must be a register in range [r0, r14]
@ CHECK-NEXT: str.w pc, [r0], #4
@ CHECK: error: operand must be in range [-255, 255]
@ CHECK-NEXT: str.w r1, [r0], #256
@ CHECK: error: operand must be in range [-255, 255]
@ CHECK-NEXT: str.w r1, [r0], #-256

@@ XPASS

@ Simple checks that we get the same encoding w/ and w/o the .w suffix.
ldr r3, [r1], #4
ldr.w r3, [r1], #4

str r3, [r0], #4
str.w r3, [r0], #4

ldr r3, [r1, #-4]!
ldr.w r3, [r1, #-4]!

str r3, [r0, #-4]!
str.w r3, [r0, #-4]!

@ CHECK: ldr r3, [r1], #4   @ encoding: [0x51,0xf8,0x04,0x3b]
@ CHECK: ldr r3, [r1], #4   @ encoding: [0x51,0xf8,0x04,0x3b]
@ CHECK: str r3, [r0], #4   @ encoding: [0x40,0xf8,0x04,0x3b]
@ CHECK: str r3, [r0], #4   @ encoding: [0x40,0xf8,0x04,0x3b]
@ CHECK: ldr r3, [r1, #-4]! @ encoding: [0x51,0xf8,0x04,0x3d]
@ CHECK: ldr r3, [r1, #-4]! @ encoding: [0x51,0xf8,0x04,0x3d]
@ CHECK: str r3, [r0, #-4]! @ encoding: [0x40,0xf8,0x04,0x3d]
@ CHECK: str r3, [r0, #-4]! @ encoding: [0x40,0xf8,0x04,0x3d]

@@ LDR pre-increment w/ writeback
@ Vary Rt.
ldr.w r0, [r1, #-4]!
ldr.w sp, [r1, #-4]! @ TODO: GAS warns for this
ldr.w pc, [r1, #-4]!
@ Vary Rn.
ldr.w r1, [r0, #-4]!
ldr.w r1, [sp, #-4]!
@ Vary imm.
ldr.w r1, [r0, #255]!
ldr.w r1, [r0, #-255]!
ldr.w r1, [r0, #0]!
@ Condition codes.
ldreq.w r1, [r0, #255]!
ldrle.w r1, [r0, #255]!

@ CHECK: ldr r0, [r1, #-4]!    @ encoding: [0x51,0xf8,0x04,0x0d]
@ CHECK: ldr sp, [r1, #-4]!    @ encoding: [0x51,0xf8,0x04,0xdd]
@ CHECK: ldr pc, [r1, #-4]!    @ encoding: [0x51,0xf8,0x04,0xfd]
@ CHECK: ldr r1, [r0, #-4]!    @ encoding: [0x50,0xf8,0x04,0x1d]
@ CHECK: ldr r1, [sp, #-4]!    @ encoding: [0x5d,0xf8,0x04,0x1d]
@ CHECK: ldr r1, [r0, #255]!   @ encoding: [0x50,0xf8,0xff,0x1f]
@ CHECK: ldr r1, [r0, #-255]!  @ encoding: [0x50,0xf8,0xff,0x1d]
@ CHECK: ldr r1, [r0, #0]!     @ encoding: [0x50,0xf8,0x00,0x1f]
@ CHECK: it    eq              @ encoding: [0x08,0xbf]
@ CHECK: ldreq r1, [r0, #255]! @ encoding: [0x50,0xf8,0xff,0x1f]
@ CHECK: it    le              @ encoding: [0xd8,0xbf]
@ CHECK: ldrle r1, [r0, #255]! @ encoding: [0x50,0xf8,0xff,0x1f]

@@ LDR post-increment
@ Vary Rt.
ldr.w r0, [r1], #4
ldr.w sp, [r1], #4 @ TODO: GAS warns for this
ldr.w pc, [r1], #4
@ Vary Rn.
ldr.w r0, [r1], #4
ldr.w r0, [sp], #4
ldr.w r0, [pc], #4 @ TODO: GAS warns for this
@ Vary imm.
ldr.w r0, [r1], #255
ldr.w r0, [r1], #0
ldr.w r0, [r1], #-255
@ Condition codes.
ldreq.w r0, [r1], #255
ldrle.w r0, [r1], #255

@ CHECK: ldr r0, [r1], #4     @ encoding: [0x51,0xf8,0x04,0x0b]
@ CHECK: ldr sp, [r1], #4     @ encoding: [0x51,0xf8,0x04,0xdb]
@ CHECK: ldr pc, [r1], #4     @ encoding: [0x51,0xf8,0x04,0xfb]
@ CHECK: ldr r0, [r1], #4     @ encoding: [0x51,0xf8,0x04,0x0b]
@ CHECK: ldr r0, [sp], #4     @ encoding: [0x5d,0xf8,0x04,0x0b]
@ CHECK: ldr r0, [pc], #4     @ encoding: [0x5f,0xf8,0x04,0x0b]
@ CHECK: ldr r0, [r1], #255   @ encoding: [0x51,0xf8,0xff,0x0b]
@ CHECK: ldr r0, [r1], #0     @ encoding: [0x51,0xf8,0x00,0x0b]
@ CHECK: ldr r0, [r1], #-255  @ encoding: [0x51,0xf8,0xff,0x09]
@ CHECK: it    eq             @ encoding: [0x08,0xbf]
@ CHECK: ldreq r0, [r1], #255 @ encoding: [0x51,0xf8,0xff,0x0b]
@ CHECK: it    le             @ encoding: [0xd8,0xbf]
@ CHECK: ldrle r0, [r1], #255 @ encoding: [0x51,0xf8,0xff,0x0b]

@@ STR pre-increment w/ writeback
@ Vary Rt.
str.w r1, [r0, #-4]!
str.w sp, [r0, #-4]!
@ Vary Rn.
str.w r1, [r2, #-4]!
str.w r1, [sp, #-4]!
@ Vary imm.
str.w r1, [r2, #255]!
str.w r1, [r2, #0]!
str.w r1, [r2, #-255]!
@ Condition codes.
streq.w r1, [r2, #255]!
strle.w r1, [r2, #255]!

@ CHECK: str r1, [r0, #-4]!     @ encoding: [0x40,0xf8,0x04,0x1d]
@ CHECK: str sp, [r0, #-4]!     @ encoding: [0x40,0xf8,0x04,0xdd]
@ CHECK: str r1, [r2, #-4]!     @ encoding: [0x42,0xf8,0x04,0x1d]
@ CHECK: str r1, [sp, #-4]!     @ encoding: [0x4d,0xf8,0x04,0x1d]
@ CHECK: str   r1, [r2, #255]!  @ encoding: [0x42,0xf8,0xff,0x1f]
@ CHECK: str   r1, [r2, #0]!    @ encoding: [0x42,0xf8,0x00,0x1f]
@ CHECK: str   r1, [r2, #-255]! @ encoding: [0x42,0xf8,0xff,0x1d]
@ CHECK: it    eq               @ encoding: [0x08,0xbf]
@ CHECK: streq r1, [r2, #255]!  @ encoding: [0x42,0xf8,0xff,0x1f]
@ CHECK: it    le               @ encoding: [0xd8,0xbf]
@ CHECK: strle r1, [r2, #255]!  @ encoding: [0x42,0xf8,0xff,0x1f]

@@ STR post-increment
@ Vary Rt.
str.w r1, [r0], #4
str.w sp, [r0], #4
@ Vary Rn.
str.w r0, [r1], #4
str.w r0, [sp], #4
str.w r0, [pc], #4 @ TODO: GAS warns for this.
@ Vary imm.
str.w r1, [r0], #255
str.w r1, [r0], #0
str.w r1, [r0], #-255
@ Condition codes.
streq.w r1, [r0], #255
strle.w r1, [r0], #255

@ CHECK: str   r1, [r0], #4    @ encoding: [0x40,0xf8,0x04,0x1b]
@ CHECK: str   sp, [r0], #4    @ encoding: [0x40,0xf8,0x04,0xdb]
@ CHECK: str   r0, [r1], #4    @ encoding: [0x41,0xf8,0x04,0x0b]
@ CHECK: str   r0, [sp], #4    @ encoding: [0x4d,0xf8,0x04,0x0b]
@ CHECK: str   r0, [pc], #4    @ encoding: [0x4f,0xf8,0x04,0x0b]
@ CHECK: str   r1, [r0], #255  @ encoding: [0x40,0xf8,0xff,0x1b]
@ CHECK: str   r1, [r0], #0    @ encoding: [0x40,0xf8,0x00,0x1b]
@ CHECK: str   r1, [r0], #-255 @ encoding: [0x40,0xf8,0xff,0x19]
@ CHECK: it    eq              @ encoding: [0x08,0xbf]
@ CHECK: streq r1, [r0], #255  @ encoding: [0x40,0xf8,0xff,0x1b]
@ CHECK: it    le              @ encoding: [0xd8,0xbf]
@ CHECK: strle r1, [r0], #255  @ encoding: [0x40,0xf8,0xff,0x1b]
