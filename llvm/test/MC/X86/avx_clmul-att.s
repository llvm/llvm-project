// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vpclmulqdq  $17, %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x43,0x29,0x44,0xdc,0x11]
          vpclmulhqhqdq %xmm12, %xmm10, %xmm11

// CHECK: vpclmulqdq  $17, (%rax), %xmm10, %xmm13
// CHECK: encoding: [0xc4,0x63,0x29,0x44,0x28,0x11]
          vpclmulhqhqdq (%rax), %xmm10, %xmm13

// CHECK: vpclmulqdq  $1, %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x43,0x29,0x44,0xdc,0x01]
          vpclmulhqlqdq %xmm12, %xmm10, %xmm11

// CHECK: vpclmulqdq  $1, (%rax), %xmm10, %xmm13
// CHECK: encoding: [0xc4,0x63,0x29,0x44,0x28,0x01]
          vpclmulhqlqdq (%rax), %xmm10, %xmm13

// CHECK: vpclmulqdq  $16, %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x43,0x29,0x44,0xdc,0x10]
          vpclmullqhqdq %xmm12, %xmm10, %xmm11

// CHECK: vpclmulqdq  $16, (%rax), %xmm10, %xmm13
// CHECK: encoding: [0xc4,0x63,0x29,0x44,0x28,0x10]
          vpclmullqhqdq (%rax), %xmm10, %xmm13

// CHECK: vpclmulqdq  $0, %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x43,0x29,0x44,0xdc,0x00]
          vpclmullqlqdq %xmm12, %xmm10, %xmm11

// CHECK: vpclmulqdq  $0, (%rax), %xmm10, %xmm13
// CHECK: encoding: [0xc4,0x63,0x29,0x44,0x28,0x00]
          vpclmullqlqdq (%rax), %xmm10, %xmm13

// CHECK: vpclmulqdq  $17, %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x43,0x29,0x44,0xdc,0x11]
          vpclmulqdq  $17, %xmm12, %xmm10, %xmm11

// CHECK: vpclmulqdq  $17, (%rax), %xmm10, %xmm13
// CHECK: encoding: [0xc4,0x63,0x29,0x44,0x28,0x11]
          vpclmulqdq  $17, (%rax), %xmm10, %xmm13

