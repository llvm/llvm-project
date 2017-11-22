//RUN: llvm-mc -triple x86_64-unknown-unknown -mcpu=knl -mattr=+vpclmulqdq --show-encoding < %s  | FileCheck %s

// CHECK: vpclmulqdq $1, %zmm3, %zmm22, %zmm1
// CHECK: encoding: [0x62,0xf3,0x4d,0x40,0x44,0xcb,0x01]
          vpclmulqdq $1, %zmm3, %zmm22, %zmm1

// CHECK: vpclmulqdq  $1, (%rcx), %zmm22, %zmm1
// CHECK: encoding: [0x62,0xf3,0x4d,0x40,0x44,0x09,0x01]
          vpclmulqdq  $1, (%rcx), %zmm22, %zmm1

// CHECK: vpclmulqdq  $1, -256(%rsp), %zmm22, %zmm1
// CHECK: encoding: [0x62,0xf3,0x4d,0x40,0x44,0x4c,0x24,0xfc,0x01]
          vpclmulqdq  $1, -256(%rsp), %zmm22, %zmm1

// CHECK: vpclmulqdq  $1, 256(%rsp), %zmm22, %zmm1
// CHECK: encoding: [0x62,0xf3,0x4d,0x40,0x44,0x4c,0x24,0x04,0x01]
          vpclmulqdq  $1, 256(%rsp), %zmm22, %zmm1

// CHECK: vpclmulqdq  $1, 268435456(%rcx,%r14,8), %zmm22, %zmm1
// CHECK: encoding: [0x62,0xb3,0x4d,0x40,0x44,0x8c,0xf1,0x00,0x00,0x00,0x10,0x01]
          vpclmulqdq  $1, 268435456(%rcx,%r14,8), %zmm22, %zmm1

// CHECK: vpclmulqdq  $1, -536870912(%rcx,%r14,8), %zmm22, %zmm1
// CHECK: encoding: [0x62,0xb3,0x4d,0x40,0x44,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x01]
          vpclmulqdq  $1, -536870912(%rcx,%r14,8), %zmm22, %zmm1

// CHECK: vpclmulqdq  $1, -536870910(%rcx,%r14,8), %zmm22, %zmm1
// CHECK: encoding: [0x62,0xb3,0x4d,0x40,0x44,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x01]
          vpclmulqdq  $1, -536870910(%rcx,%r14,8), %zmm22, %zmm1
