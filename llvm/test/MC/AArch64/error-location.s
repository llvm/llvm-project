// RUN: not llvm-mc -triple aarch64--none-eabi -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

// Note: These errors are not always emitted in the order in which the relevant
// source appears, this file is carefully ordered so that that is the case.

  .text
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: symbol 'undef' can not be undefined in a subtraction expression
  .word (0-undef)

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: expected relocatable expression
  .word -undef

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: No relocation available to represent this relative expression
  adr x0, #a-undef

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Cannot represent a difference across sections
  .word x_a - y_a

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 1-byte data relocations not supported
  .byte undef

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 1-byte data relocations not supported
  .byte undef-.

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Unsupported pc-relative fixup kind
  ldr x0, [x1, :lo12:undef-.]

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid fixup for 8-bit load/store instruction
  ldrb w0, [x1, :gottprel_lo12:undef]

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid fixup for 16-bit load/store instruction
  ldrh w0, [x1, :gottprel_lo12:undef]

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: LP64 32-bit load/store relocation not supported (ILP32 eqv: TLSIE_LD32_GOTTPREL_LO12_NC)
  ldr w0, [x1, :gottprel_lo12:undef]



w:
  .word 0
  .weak w


  .section sec_x
x_a:
  .word 0


  .section sec_y
y_a:
  .word 0
