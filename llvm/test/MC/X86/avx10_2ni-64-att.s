// RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s

// VMPSADBW

// CHECK: vmpsadbw $123, %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x83,0x46,0x00,0x42,0xf0,0x7b]
          vmpsadbw $123, %xmm24, %xmm23, %xmm22

// CHECK: vmpsadbw $123, %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x46,0x07,0x42,0xf0,0x7b]
          vmpsadbw $123, %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vmpsadbw $123, %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x46,0x87,0x42,0xf0,0x7b]
          vmpsadbw $123, %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vmpsadbw $123, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x83,0x46,0x20,0x42,0xf0,0x7b]
          vmpsadbw $123, %ymm24, %ymm23, %ymm22

// CHECK: vmpsadbw $123, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x46,0x27,0x42,0xf0,0x7b]
          vmpsadbw $123, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vmpsadbw $123, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x46,0xa7,0x42,0xf0,0x7b]
          vmpsadbw $123, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmpsadbw $123, %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x83,0x46,0x40,0x42,0xf0,0x7b]
          vmpsadbw $123, %zmm24, %zmm23, %zmm22

// CHECK: vmpsadbw $123, %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x46,0x47,0x42,0xf0,0x7b]
          vmpsadbw $123, %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vmpsadbw $123, %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x46,0xc7,0x42,0xf0,0x7b]
          vmpsadbw $123, %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vmpsadbw  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0x46,0x00,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vmpsadbw  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vmpsadbw  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x46,0x07,0x42,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vmpsadbw  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vmpsadbw  $123, (%rip), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0x46,0x00,0x42,0x35,0x00,0x00,0x00,0x00,0x7b]
          vmpsadbw  $123, (%rip), %xmm23, %xmm22

// CHECK: vmpsadbw  $123, -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0x46,0x00,0x42,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vmpsadbw  $123, -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vmpsadbw  $123, 2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x46,0x87,0x42,0x71,0x7f,0x7b]
          vmpsadbw  $123, 2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vmpsadbw  $123, -2048(%rdx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x46,0x87,0x42,0x72,0x80,0x7b]
          vmpsadbw  $123, -2048(%rdx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vmpsadbw  $123, 268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x46,0x20,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vmpsadbw  $123, 268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vmpsadbw  $123, 291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x46,0x27,0x42,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vmpsadbw  $123, 291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vmpsadbw  $123, (%rip), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe3,0x46,0x20,0x42,0x35,0x00,0x00,0x00,0x00,0x7b]
          vmpsadbw  $123, (%rip), %ymm23, %ymm22

// CHECK: vmpsadbw  $123, -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe3,0x46,0x20,0x42,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vmpsadbw  $123, -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vmpsadbw  $123, 4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x46,0xa7,0x42,0x71,0x7f,0x7b]
          vmpsadbw  $123, 4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vmpsadbw  $123, -4096(%rdx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x46,0xa7,0x42,0x72,0x80,0x7b]
          vmpsadbw  $123, -4096(%rdx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vmpsadbw  $123, 268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa3,0x46,0x40,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vmpsadbw  $123, 268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vmpsadbw  $123, 291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x46,0x47,0x42,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vmpsadbw  $123, 291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vmpsadbw  $123, (%rip), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe3,0x46,0x40,0x42,0x35,0x00,0x00,0x00,0x00,0x7b]
          vmpsadbw  $123, (%rip), %zmm23, %zmm22

// CHECK: vmpsadbw  $123, -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe3,0x46,0x40,0x42,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vmpsadbw  $123, -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vmpsadbw  $123, 8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x46,0xc7,0x42,0x71,0x7f,0x7b]
          vmpsadbw  $123, 8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vmpsadbw  $123, -8192(%rdx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x46,0xc7,0x42,0x72,0x80,0x7b]
          vmpsadbw  $123, -8192(%rdx), %zmm23, %zmm22 {%k7} {z}

// YMM Rounding

// CHECK: vaddpd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x81,0xc1,0x10,0x58,0xf0]
          vaddpd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vaddpd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x81,0xc1,0x37,0x58,0xf0]
          vaddpd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vaddpd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x81,0xc1,0xf7,0x58,0xf0]
          vaddpd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vaddph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x40,0x10,0x58,0xf0]
          vaddph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vaddph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x40,0x37,0x58,0xf0]
          vaddph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vaddph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x40,0xf7,0x58,0xf0]
          vaddph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vaddps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x81,0x40,0x10,0x58,0xf0]
          vaddps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vaddps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x81,0x40,0x37,0x58,0xf0]
          vaddps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vaddps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x81,0x40,0xf7,0x58,0xf0]
          vaddps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
