// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vpclmulqdq $17, %ymm3, %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x6d,0x44,0xcb,0x11]
          vpclmulqdq $17, %ymm3, %ymm2, %ymm1

// CHECK: vpclmulqdq  $1, (%rcx), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x6d,0x44,0x09,0x01]
          vpclmulqdq  $1, (%rcx), %ymm2, %ymm1

// CHECK: vpclmulqdq  $1, -4(%rsp), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x6d,0x44,0x4c,0x24,0xfc,0x01]
          vpclmulqdq  $1, -4(%rsp), %ymm2, %ymm1

// CHECK: vpclmulqdq  $1, 4(%rsp), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x6d,0x44,0x4c,0x24,0x04,0x01]
          vpclmulqdq  $1, 4(%rsp), %ymm2, %ymm1

// CHECK: vpclmulqdq  $1, 268435456(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa3,0x6d,0x44,0x8c,0xf1,0x00,0x00,0x00,0x10,0x01]
          vpclmulqdq  $1, 268435456(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vpclmulqdq  $1, -536870912(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa3,0x6d,0x44,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x01]
          vpclmulqdq  $1, -536870912(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vpclmulqdq  $1, -536870910(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa3,0x6d,0x44,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x01]
          vpclmulqdq  $1, -536870910(%rcx,%r14,8), %ymm2, %ymm1

