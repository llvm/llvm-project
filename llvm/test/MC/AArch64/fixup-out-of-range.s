// RUN: not llvm-mc -triple aarch64 -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64-windows -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-WIN

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  adr x0, distant

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  ldr x0, distant

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup not sufficiently aligned
  ldr x0, unaligned

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  b.eq distant

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup not sufficiently aligned
  b.eq unaligned

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  ldr x0, [x1, distant-.]

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup must be 8-byte aligned
  ldr x0, [x1, unaligned-.]

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  ldr w0, [x1, distant-.]

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup must be 4-byte aligned
  ldr w0, [x1, unaligned-.]

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  ldrh w0, [x1, distant-.]

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup must be 2-byte aligned
  ldrh w0, [x1, unaligned-.]

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  ldrb w0, [x1, distant-.]

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  ldr q0, [x1, distant-.]

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup must be 16-byte aligned
  ldr q0, [x1, unaligned-.]

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  tbz x0, #1, distant

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup not sufficiently aligned
  tbz x0, #1, unaligned

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  b distant

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup not sufficiently aligned
  b unaligned

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  movz x0, #:abs_g0:value1

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  movz x0, #:abs_g1:value2

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  movz x0, #:abs_g0_s:value1

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  movz x0, #:abs_g1_s:value2

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: relocation for a thread-local variable points to an absolute symbol
  movz x0, #:tprel_g0:value1

// CHECK-WIN: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  adrp x0, external+0x1000000

  .byte 0
unaligned:
  .byte 0

  .space 1<<27
  .balign 8
distant:
  .word 0
value1 = 0x12345678
value2 = 0x123456789
