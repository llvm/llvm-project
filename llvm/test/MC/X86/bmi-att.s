// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: blsmskl  %r11d, %r10d
// CHECK: encoding: [0xc4,0xc2,0x28,0xf3,0xd3]
          blsmskl  %r11d, %r10d

// CHECK: blsmskq  %r11, %r10
// CHECK: encoding: [0xc4,0xc2,0xa8,0xf3,0xd3]
          blsmskq  %r11, %r10

// CHECK: blsmskl  (%rax), %r10d
// CHECK: encoding: [0xc4,0xe2,0x28,0xf3,0x10]
          blsmskl  (%rax), %r10d

// CHECK: blsmskq  (%rax), %r10
// CHECK: encoding: [0xc4,0xe2,0xa8,0xf3,0x10]
          blsmskq  (%rax), %r10

// CHECK: blsil  %r11d, %r10d
// CHECK: encoding: [0xc4,0xc2,0x28,0xf3,0xdb]
          blsil  %r11d, %r10d

// CHECK: blsiq  %r11, %r10
// CHECK: encoding: [0xc4,0xc2,0xa8,0xf3,0xdb]
          blsiq  %r11, %r10

// CHECK: blsil  (%rax), %r10d
// CHECK: encoding: [0xc4,0xe2,0x28,0xf3,0x18]
          blsil  (%rax), %r10d

// CHECK: blsiq  (%rax), %r10
// CHECK: encoding: [0xc4,0xe2,0xa8,0xf3,0x18]
          blsiq  (%rax), %r10

// CHECK: blsrl  %r11d, %r10d
// CHECK: encoding: [0xc4,0xc2,0x28,0xf3,0xcb]
          blsrl  %r11d, %r10d

// CHECK: blsrq  %r11, %r10
// CHECK: encoding: [0xc4,0xc2,0xa8,0xf3,0xcb]
          blsrq  %r11, %r10

// CHECK: blsrl  (%rax), %r10d
// CHECK: encoding: [0xc4,0xe2,0x28,0xf3,0x08]
          blsrl  (%rax), %r10d

// CHECK: blsrq  (%rax), %r10
// CHECK: encoding: [0xc4,0xe2,0xa8,0xf3,0x08]
          blsrq  (%rax), %r10

// CHECK: andnl  (%rax), %r11d, %r10d
// CHECK: encoding: [0xc4,0x62,0x20,0xf2,0x10]
          andnl  (%rax), %r11d, %r10d

// CHECK: andnq  (%rax), %r11, %r10
// CHECK: encoding: [0xc4,0x62,0xa0,0xf2,0x10]
          andnq  (%rax), %r11, %r10

// CHECK: bextrl %r12d, (%rax), %r10d
// CHECK: encoding: [0xc4,0x62,0x18,0xf7,0x10]
          bextrl %r12d, (%rax), %r10d

// CHECK: bextrl %r12d, %r11d, %r10d
// CHECK: encoding: [0xc4,0x42,0x18,0xf7,0xd3]
          bextrl %r12d, %r11d, %r10d

// CHECK: bextrq %r12, (%rax), %r10
// CHECK: encoding: [0xc4,0x62,0x98,0xf7,0x10]
          bextrq %r12, (%rax), %r10

// CHECK: bextrq %r12, %r11, %r10
// CHECK: encoding: [0xc4,0x42,0x98,0xf7,0xd3]
          bextrq %r12, %r11, %r10

// CHECK: bzhil %r12d, (%rax), %r10d
// CHECK: encoding: [0xc4,0x62,0x18,0xf5,0x10]
          bzhil %r12d, (%rax), %r10d

// CHECK: bzhil %r12d, %r11d, %r10d
// CHECK: encoding: [0xc4,0x42,0x18,0xf5,0xd3]
          bzhil %r12d, %r11d, %r10d

// CHECK: bzhiq %r12, (%rax), %r10
// CHECK: encoding: [0xc4,0x62,0x98,0xf5,0x10]
          bzhiq %r12, (%rax), %r10

// CHECK: bzhiq %r12, %r11, %r10
// CHECK: encoding: [0xc4,0x42,0x98,0xf5,0xd3]
          bzhiq %r12, %r11, %r10

// CHECK: pextl %r12d, %r11d, %r10d
// CHECK: encoding: [0xc4,0x42,0x22,0xf5,0xd4]
          pextl %r12d, %r11d, %r10d

// CHECK: pextl (%rax), %r11d, %r10d
// CHECK: encoding: [0xc4,0x62,0x22,0xf5,0x10]
          pextl (%rax), %r11d, %r10d

// CHECK: pextq %r12, %r11, %r10
// CHECK: encoding: [0xc4,0x42,0xa2,0xf5,0xd4]
          pextq %r12, %r11, %r10

// CHECK: pextq (%rax), %r11, %r10
// CHECK: encoding: [0xc4,0x62,0xa2,0xf5,0x10]
          pextq (%rax), %r11, %r10

// CHECK: pdepl %r12d, %r11d, %r10d
// CHECK: encoding: [0xc4,0x42,0x23,0xf5,0xd4]
          pdepl %r12d, %r11d, %r10d

// CHECK: pdepl (%rax), %r11d, %r10d
// CHECK: encoding: [0xc4,0x62,0x23,0xf5,0x10]
          pdepl (%rax), %r11d, %r10d

// CHECK: pdepq %r12, %r11, %r10
// CHECK: encoding: [0xc4,0x42,0xa3,0xf5,0xd4]
          pdepq %r12, %r11, %r10

// CHECK: pdepq (%rax), %r11, %r10
// CHECK: encoding: [0xc4,0x62,0xa3,0xf5,0x10]
          pdepq (%rax), %r11, %r10

// CHECK: mulxl %r12d, %r11d, %r10d
// CHECK: encoding: [0xc4,0x42,0x23,0xf6,0xd4]
          mulxl %r12d, %r11d, %r10d

// CHECK: mulxl (%rax), %r11d, %r10d
// CHECK: encoding: [0xc4,0x62,0x23,0xf6,0x10]
          mulxl (%rax), %r11d, %r10d

// CHECK: mulxq %r12, %r11, %r10
// CHECK: encoding: [0xc4,0x42,0xa3,0xf6,0xd4]
          mulxq %r12, %r11, %r10

// CHECK: mulxq (%rax), %r11, %r10
// CHECK: encoding: [0xc4,0x62,0xa3,0xf6,0x10]
          mulxq (%rax), %r11, %r10

// CHECK: rorxl $10, %r12d, %r10d
// CHECK: encoding: [0xc4,0x43,0x7b,0xf0,0xd4,0x0a]
          rorxl $10, %r12d, %r10d

// CHECK: rorxl $31, (%rax), %r10d
// CHECK: encoding: [0xc4,0x63,0x7b,0xf0,0x10,0x1f]
          rorxl $31, (%rax), %r10d

// CHECK: rorxq $1, %r12, %r10
// CHECK: encoding: [0xc4,0x43,0xfb,0xf0,0xd4,0x01]
          rorxq $1, %r12, %r10

// CHECK: rorxq $63, (%rax), %r10
// CHECK: encoding: [0xc4,0x63,0xfb,0xf0,0x10,0x3f]
          rorxq $63, (%rax), %r10

// CHECK: shlxl %r12d, (%rax), %r10d
// CHECK: encoding: [0xc4,0x62,0x19,0xf7,0x10]
          shlxl %r12d, (%rax), %r10d

// CHECK: shlxl %r12d, %r11d, %r10d
// CHECK: encoding: [0xc4,0x42,0x19,0xf7,0xd3]
          shlxl %r12d, %r11d, %r10d

// CHECK: shlxq %r12, (%rax), %r10
// CHECK: encoding: [0xc4,0x62,0x99,0xf7,0x10]
          shlxq %r12, (%rax), %r10

// CHECK: shlxq %r12, %r11, %r10
// CHECK: encoding: [0xc4,0x42,0x99,0xf7,0xd3]
          shlxq %r12, %r11, %r10

// CHECK: sarxl %r12d, (%rax), %r10d
// CHECK: encoding: [0xc4,0x62,0x1a,0xf7,0x10]
          sarxl %r12d, (%rax), %r10d

// CHECK: sarxl %r12d, %r11d, %r10d
// CHECK: encoding: [0xc4,0x42,0x1a,0xf7,0xd3]
          sarxl %r12d, %r11d, %r10d

// CHECK: sarxq %r12, (%rax), %r10
// CHECK: encoding: [0xc4,0x62,0x9a,0xf7,0x10]
          sarxq %r12, (%rax), %r10

// CHECK: sarxq %r12, %r11, %r10
// CHECK: encoding: [0xc4,0x42,0x9a,0xf7,0xd3]
          sarxq %r12, %r11, %r10

// CHECK: shrxl %r12d, (%rax), %r10d
// CHECK: encoding: [0xc4,0x62,0x1b,0xf7,0x10]
          shrxl %r12d, (%rax), %r10d

// CHECK: shrxl %r12d, %r11d, %r10d
// CHECK: encoding: [0xc4,0x42,0x1b,0xf7,0xd3]
          shrxl %r12d, %r11d, %r10d

// CHECK: shrxq %r12, (%rax), %r10
// CHECK: encoding: [0xc4,0x62,0x9b,0xf7,0x10]
          shrxq %r12, (%rax), %r10

// CHECK: shrxq %r12, %r11, %r10
// CHECK: encoding: [0xc4,0x42,0x9b,0xf7,0xd3]
          shrxq %r12, %r11, %r10

