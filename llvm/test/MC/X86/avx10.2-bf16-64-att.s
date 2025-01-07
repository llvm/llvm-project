// RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s

// CHECK: vaddnepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x58,0xf0]
          vaddnepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vaddnepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x58,0xf0]
          vaddnepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vaddnepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x58,0xf0]
          vaddnepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vaddnepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x58,0xf0]
          vaddnepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vaddnepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x58,0xf0]
          vaddnepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vaddnepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x58,0xf0]
          vaddnepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vaddnepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x58,0xf0]
          vaddnepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vaddnepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x58,0xf0]
          vaddnepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vaddnepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x58,0xf0]
          vaddnepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vaddnepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddnepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vaddnepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x58,0xb4,0x80,0x23,0x01,0x00,0x00]
          vaddnepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vaddnepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x58,0x35,0x00,0x00,0x00,0x00]
          vaddnepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vaddnepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x58,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vaddnepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vaddnepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x58,0x71,0x7f]
          vaddnepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vaddnepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x58,0x72,0x80]
          vaddnepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vaddnepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddnepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vaddnepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x58,0xb4,0x80,0x23,0x01,0x00,0x00]
          vaddnepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vaddnepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x58,0x35,0x00,0x00,0x00,0x00]
          vaddnepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vaddnepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x58,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vaddnepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vaddnepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x58,0x71,0x7f]
          vaddnepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vaddnepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x58,0x72,0x80]
          vaddnepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vaddnepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddnepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vaddnepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x58,0xb4,0x80,0x23,0x01,0x00,0x00]
          vaddnepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vaddnepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x58,0x35,0x00,0x00,0x00,0x00]
          vaddnepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vaddnepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x58,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vaddnepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vaddnepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x58,0x71,0x7f]
          vaddnepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vaddnepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x58,0x72,0x80]
          vaddnepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcmppbf16 $123, %ymm24, %ymm23, %k5
// CHECK: encoding: [0x62,0x93,0x47,0x20,0xc2,0xe8,0x7b]
          vcmppbf16 $123, %ymm24, %ymm23, %k5

// CHECK: vcmppbf16 $123, %ymm24, %ymm23, %k5 {%k7}
// CHECK: encoding: [0x62,0x93,0x47,0x27,0xc2,0xe8,0x7b]
          vcmppbf16 $123, %ymm24, %ymm23, %k5 {%k7}

// CHECK: vcmppbf16 $123, %xmm24, %xmm23, %k5
// CHECK: encoding: [0x62,0x93,0x47,0x00,0xc2,0xe8,0x7b]
          vcmppbf16 $123, %xmm24, %xmm23, %k5

// CHECK: vcmppbf16 $123, %xmm24, %xmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0x93,0x47,0x07,0xc2,0xe8,0x7b]
          vcmppbf16 $123, %xmm24, %xmm23, %k5 {%k7}

// CHECK: vcmppbf16 $123, %zmm24, %zmm23, %k5
// CHECK: encoding: [0x62,0x93,0x47,0x40,0xc2,0xe8,0x7b]
          vcmppbf16 $123, %zmm24, %zmm23, %k5

// CHECK: vcmppbf16 $123, %zmm24, %zmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0x93,0x47,0x47,0xc2,0xe8,0x7b]
          vcmppbf16 $123, %zmm24, %zmm23, %k5 {%k7}

// CHECK: vcmppbf16  $123, 268435456(%rbp,%r14,8), %zmm23, %k5
// CHECK: encoding: [0x62,0xb3,0x47,0x40,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vcmppbf16  $123, 268435456(%rbp,%r14,8), %zmm23, %k5

// CHECK: vcmppbf16  $123, 291(%r8,%rax,4), %zmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xd3,0x47,0x47,0xc2,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vcmppbf16  $123, 291(%r8,%rax,4), %zmm23, %k5 {%k7}

// CHECK: vcmppbf16  $123, (%rip){1to32}, %zmm23, %k5
// CHECK: encoding: [0x62,0xf3,0x47,0x50,0xc2,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vcmppbf16  $123, (%rip){1to32}, %zmm23, %k5

// CHECK: vcmppbf16  $123, -2048(,%rbp,2), %zmm23, %k5
// CHECK: encoding: [0x62,0xf3,0x47,0x40,0xc2,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vcmppbf16  $123, -2048(,%rbp,2), %zmm23, %k5

// CHECK: vcmppbf16  $123, 8128(%rcx), %zmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x47,0x47,0xc2,0x69,0x7f,0x7b]
          vcmppbf16  $123, 8128(%rcx), %zmm23, %k5 {%k7}

// CHECK: vcmppbf16  $123, -256(%rdx){1to32}, %zmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x47,0x57,0xc2,0x6a,0x80,0x7b]
          vcmppbf16  $123, -256(%rdx){1to32}, %zmm23, %k5 {%k7}

// CHECK: vcmppbf16  $123, 268435456(%rbp,%r14,8), %xmm23, %k5
// CHECK: encoding: [0x62,0xb3,0x47,0x00,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vcmppbf16  $123, 268435456(%rbp,%r14,8), %xmm23, %k5

// CHECK: vcmppbf16  $123, 291(%r8,%rax,4), %xmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xd3,0x47,0x07,0xc2,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vcmppbf16  $123, 291(%r8,%rax,4), %xmm23, %k5 {%k7}

// CHECK: vcmppbf16  $123, (%rip){1to8}, %xmm23, %k5
// CHECK: encoding: [0x62,0xf3,0x47,0x10,0xc2,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vcmppbf16  $123, (%rip){1to8}, %xmm23, %k5

// CHECK: vcmppbf16  $123, -512(,%rbp,2), %xmm23, %k5
// CHECK: encoding: [0x62,0xf3,0x47,0x00,0xc2,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vcmppbf16  $123, -512(,%rbp,2), %xmm23, %k5

// CHECK: vcmppbf16  $123, 2032(%rcx), %xmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x47,0x07,0xc2,0x69,0x7f,0x7b]
          vcmppbf16  $123, 2032(%rcx), %xmm23, %k5 {%k7}

// CHECK: vcmppbf16  $123, -256(%rdx){1to8}, %xmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x47,0x17,0xc2,0x6a,0x80,0x7b]
          vcmppbf16  $123, -256(%rdx){1to8}, %xmm23, %k5 {%k7}

// CHECK: vcmppbf16  $123, 268435456(%rbp,%r14,8), %ymm23, %k5
// CHECK: encoding: [0x62,0xb3,0x47,0x20,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vcmppbf16  $123, 268435456(%rbp,%r14,8), %ymm23, %k5

// CHECK: vcmppbf16  $123, 291(%r8,%rax,4), %ymm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xd3,0x47,0x27,0xc2,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vcmppbf16  $123, 291(%r8,%rax,4), %ymm23, %k5 {%k7}

// CHECK: vcmppbf16  $123, (%rip){1to16}, %ymm23, %k5
// CHECK: encoding: [0x62,0xf3,0x47,0x30,0xc2,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vcmppbf16  $123, (%rip){1to16}, %ymm23, %k5

// CHECK: vcmppbf16  $123, -1024(,%rbp,2), %ymm23, %k5
// CHECK: encoding: [0x62,0xf3,0x47,0x20,0xc2,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vcmppbf16  $123, -1024(,%rbp,2), %ymm23, %k5

// CHECK: vcmppbf16  $123, 4064(%rcx), %ymm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x47,0x27,0xc2,0x69,0x7f,0x7b]
          vcmppbf16  $123, 4064(%rcx), %ymm23, %k5 {%k7}

// CHECK: vcmppbf16  $123, -256(%rdx){1to16}, %ymm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x47,0x37,0xc2,0x6a,0x80,0x7b]
          vcmppbf16  $123, -256(%rdx){1to16}, %ymm23, %k5 {%k7}

// CHECK: vcomsbf16 %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x2f,0xf7]
          vcomsbf16 %xmm23, %xmm22

// CHECK: vcomsbf16  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x2f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcomsbf16  268435456(%rbp,%r14,8), %xmm22

// CHECK: vcomsbf16  291(%r8,%rax,4), %xmm22
// CHECK: encoding: [0x62,0xc5,0x7d,0x08,0x2f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcomsbf16  291(%r8,%rax,4), %xmm22

// CHECK: vcomsbf16  (%rip), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x35,0x00,0x00,0x00,0x00]
          vcomsbf16  (%rip), %xmm22

// CHECK: vcomsbf16  -64(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x34,0x6d,0xc0,0xff,0xff,0xff]
          vcomsbf16  -64(,%rbp,2), %xmm22

// CHECK: vcomsbf16  254(%rcx), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x71,0x7f]
          vcomsbf16  254(%rcx), %xmm22

// CHECK: vcomsbf16  -256(%rdx), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x72,0x80]
          vcomsbf16  -256(%rdx), %xmm22

// CHECK: vdivnepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5e,0xf0]
          vdivnepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vdivnepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5e,0xf0]
          vdivnepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vdivnepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5e,0xf0]
          vdivnepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vdivnepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5e,0xf0]
          vdivnepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vdivnepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5e,0xf0]
          vdivnepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vdivnepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5e,0xf0]
          vdivnepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vdivnepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5e,0xf0]
          vdivnepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vdivnepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5e,0xf0]
          vdivnepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vdivnepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5e,0xf0]
          vdivnepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vdivnepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivnepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vdivnepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdivnepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vdivnepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5e,0x35,0x00,0x00,0x00,0x00]
          vdivnepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vdivnepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5e,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vdivnepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vdivnepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5e,0x71,0x7f]
          vdivnepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vdivnepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5e,0x72,0x80]
          vdivnepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vdivnepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivnepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vdivnepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdivnepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vdivnepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5e,0x35,0x00,0x00,0x00,0x00]
          vdivnepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vdivnepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vdivnepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vdivnepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5e,0x71,0x7f]
          vdivnepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vdivnepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5e,0x72,0x80]
          vdivnepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vdivnepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivnepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vdivnepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdivnepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vdivnepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5e,0x35,0x00,0x00,0x00,0x00]
          vdivnepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vdivnepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vdivnepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vdivnepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5e,0x71,0x7f]
          vdivnepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vdivnepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5e,0x72,0x80]
          vdivnepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd132nepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x98,0xf0]
          vfmadd132nepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vfmadd132nepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x98,0xf0]
          vfmadd132nepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd132nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x98,0xf0]
          vfmadd132nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd132nepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x98,0xf0]
          vfmadd132nepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vfmadd132nepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x98,0xf0]
          vfmadd132nepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfmadd132nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x98,0xf0]
          vfmadd132nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd132nepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x98,0xf0]
          vfmadd132nepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vfmadd132nepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x98,0xf0]
          vfmadd132nepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfmadd132nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x98,0xf0]
          vfmadd132nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd132nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfmadd132nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x98,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd132nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfmadd132nepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x98,0x35,0x00,0x00,0x00,0x00]
          vfmadd132nepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfmadd132nepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x98,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd132nepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfmadd132nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x98,0x71,0x7f]
          vfmadd132nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd132nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x98,0x72,0x80]
          vfmadd132nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd132nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfmadd132nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x98,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd132nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfmadd132nepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x98,0x35,0x00,0x00,0x00,0x00]
          vfmadd132nepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfmadd132nepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x98,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd132nepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfmadd132nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x98,0x71,0x7f]
          vfmadd132nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd132nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x98,0x72,0x80]
          vfmadd132nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd132nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfmadd132nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x98,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd132nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfmadd132nepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x98,0x35,0x00,0x00,0x00,0x00]
          vfmadd132nepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfmadd132nepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x98,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd132nepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfmadd132nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x98,0x71,0x7f]
          vfmadd132nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd132nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x98,0x72,0x80]
          vfmadd132nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd213nepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xa8,0xf0]
          vfmadd213nepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vfmadd213nepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xa8,0xf0]
          vfmadd213nepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd213nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xa8,0xf0]
          vfmadd213nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd213nepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xa8,0xf0]
          vfmadd213nepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vfmadd213nepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xa8,0xf0]
          vfmadd213nepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfmadd213nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xa8,0xf0]
          vfmadd213nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd213nepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xa8,0xf0]
          vfmadd213nepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vfmadd213nepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xa8,0xf0]
          vfmadd213nepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfmadd213nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xa8,0xf0]
          vfmadd213nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd213nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfmadd213nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xa8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd213nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfmadd213nepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xa8,0x35,0x00,0x00,0x00,0x00]
          vfmadd213nepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfmadd213nepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xa8,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd213nepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfmadd213nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xa8,0x71,0x7f]
          vfmadd213nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd213nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xa8,0x72,0x80]
          vfmadd213nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd213nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfmadd213nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xa8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd213nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfmadd213nepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xa8,0x35,0x00,0x00,0x00,0x00]
          vfmadd213nepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfmadd213nepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xa8,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd213nepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfmadd213nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xa8,0x71,0x7f]
          vfmadd213nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd213nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xa8,0x72,0x80]
          vfmadd213nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd213nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfmadd213nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xa8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd213nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfmadd213nepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xa8,0x35,0x00,0x00,0x00,0x00]
          vfmadd213nepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfmadd213nepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xa8,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd213nepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfmadd213nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xa8,0x71,0x7f]
          vfmadd213nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd213nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xa8,0x72,0x80]
          vfmadd213nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd231nepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xb8,0xf0]
          vfmadd231nepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vfmadd231nepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xb8,0xf0]
          vfmadd231nepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd231nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xb8,0xf0]
          vfmadd231nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd231nepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xb8,0xf0]
          vfmadd231nepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vfmadd231nepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xb8,0xf0]
          vfmadd231nepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfmadd231nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xb8,0xf0]
          vfmadd231nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd231nepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xb8,0xf0]
          vfmadd231nepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vfmadd231nepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xb8,0xf0]
          vfmadd231nepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfmadd231nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xb8,0xf0]
          vfmadd231nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd231nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfmadd231nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xb8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd231nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfmadd231nepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xb8,0x35,0x00,0x00,0x00,0x00]
          vfmadd231nepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfmadd231nepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xb8,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd231nepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfmadd231nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xb8,0x71,0x7f]
          vfmadd231nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd231nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xb8,0x72,0x80]
          vfmadd231nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd231nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfmadd231nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xb8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd231nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfmadd231nepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xb8,0x35,0x00,0x00,0x00,0x00]
          vfmadd231nepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfmadd231nepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xb8,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd231nepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfmadd231nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xb8,0x71,0x7f]
          vfmadd231nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd231nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xb8,0x72,0x80]
          vfmadd231nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd231nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfmadd231nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xb8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd231nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfmadd231nepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xb8,0x35,0x00,0x00,0x00,0x00]
          vfmadd231nepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfmadd231nepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xb8,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd231nepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfmadd231nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xb8,0x71,0x7f]
          vfmadd231nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd231nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xb8,0x72,0x80]
          vfmadd231nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub132nepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x9a,0xf0]
          vfmsub132nepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vfmsub132nepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x9a,0xf0]
          vfmsub132nepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub132nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x9a,0xf0]
          vfmsub132nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub132nepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x9a,0xf0]
          vfmsub132nepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vfmsub132nepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x9a,0xf0]
          vfmsub132nepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfmsub132nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x9a,0xf0]
          vfmsub132nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub132nepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x9a,0xf0]
          vfmsub132nepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vfmsub132nepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x9a,0xf0]
          vfmsub132nepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfmsub132nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x9a,0xf0]
          vfmsub132nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub132nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfmsub132nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x9a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub132nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfmsub132nepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x9a,0x35,0x00,0x00,0x00,0x00]
          vfmsub132nepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfmsub132nepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x9a,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub132nepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfmsub132nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x9a,0x71,0x7f]
          vfmsub132nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub132nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x9a,0x72,0x80]
          vfmsub132nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub132nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfmsub132nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x9a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub132nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfmsub132nepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x9a,0x35,0x00,0x00,0x00,0x00]
          vfmsub132nepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfmsub132nepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x9a,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub132nepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfmsub132nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x9a,0x71,0x7f]
          vfmsub132nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub132nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x9a,0x72,0x80]
          vfmsub132nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub132nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfmsub132nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x9a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub132nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfmsub132nepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x9a,0x35,0x00,0x00,0x00,0x00]
          vfmsub132nepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfmsub132nepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x9a,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub132nepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfmsub132nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x9a,0x71,0x7f]
          vfmsub132nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub132nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x9a,0x72,0x80]
          vfmsub132nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub213nepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xaa,0xf0]
          vfmsub213nepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vfmsub213nepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xaa,0xf0]
          vfmsub213nepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub213nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xaa,0xf0]
          vfmsub213nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub213nepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xaa,0xf0]
          vfmsub213nepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vfmsub213nepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xaa,0xf0]
          vfmsub213nepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfmsub213nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xaa,0xf0]
          vfmsub213nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub213nepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xaa,0xf0]
          vfmsub213nepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vfmsub213nepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xaa,0xf0]
          vfmsub213nepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfmsub213nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xaa,0xf0]
          vfmsub213nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub213nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfmsub213nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xaa,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub213nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfmsub213nepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xaa,0x35,0x00,0x00,0x00,0x00]
          vfmsub213nepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfmsub213nepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xaa,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub213nepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfmsub213nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xaa,0x71,0x7f]
          vfmsub213nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub213nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xaa,0x72,0x80]
          vfmsub213nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub213nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfmsub213nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xaa,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub213nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfmsub213nepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xaa,0x35,0x00,0x00,0x00,0x00]
          vfmsub213nepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfmsub213nepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xaa,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub213nepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfmsub213nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xaa,0x71,0x7f]
          vfmsub213nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub213nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xaa,0x72,0x80]
          vfmsub213nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub213nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfmsub213nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xaa,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub213nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfmsub213nepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xaa,0x35,0x00,0x00,0x00,0x00]
          vfmsub213nepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfmsub213nepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xaa,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub213nepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfmsub213nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xaa,0x71,0x7f]
          vfmsub213nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub213nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xaa,0x72,0x80]
          vfmsub213nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub231nepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xba,0xf0]
          vfmsub231nepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vfmsub231nepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xba,0xf0]
          vfmsub231nepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub231nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xba,0xf0]
          vfmsub231nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub231nepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xba,0xf0]
          vfmsub231nepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vfmsub231nepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xba,0xf0]
          vfmsub231nepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfmsub231nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xba,0xf0]
          vfmsub231nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub231nepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xba,0xf0]
          vfmsub231nepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vfmsub231nepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xba,0xf0]
          vfmsub231nepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfmsub231nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xba,0xf0]
          vfmsub231nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub231nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfmsub231nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xba,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub231nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfmsub231nepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xba,0x35,0x00,0x00,0x00,0x00]
          vfmsub231nepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfmsub231nepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xba,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub231nepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfmsub231nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xba,0x71,0x7f]
          vfmsub231nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub231nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xba,0x72,0x80]
          vfmsub231nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub231nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfmsub231nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xba,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub231nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfmsub231nepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xba,0x35,0x00,0x00,0x00,0x00]
          vfmsub231nepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfmsub231nepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xba,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub231nepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfmsub231nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xba,0x71,0x7f]
          vfmsub231nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub231nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xba,0x72,0x80]
          vfmsub231nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub231nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfmsub231nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xba,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub231nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfmsub231nepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xba,0x35,0x00,0x00,0x00,0x00]
          vfmsub231nepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfmsub231nepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xba,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub231nepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfmsub231nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xba,0x71,0x7f]
          vfmsub231nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub231nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xba,0x72,0x80]
          vfmsub231nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd132nepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x9c,0xf0]
          vfnmadd132nepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd132nepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x9c,0xf0]
          vfnmadd132nepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd132nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x9c,0xf0]
          vfnmadd132nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd132nepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x9c,0xf0]
          vfnmadd132nepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vfnmadd132nepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x9c,0xf0]
          vfnmadd132nepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfnmadd132nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x9c,0xf0]
          vfnmadd132nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd132nepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x9c,0xf0]
          vfnmadd132nepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vfnmadd132nepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x9c,0xf0]
          vfnmadd132nepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfnmadd132nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x9c,0xf0]
          vfnmadd132nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd132nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfnmadd132nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x9c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd132nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfnmadd132nepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x9c,0x35,0x00,0x00,0x00,0x00]
          vfnmadd132nepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfnmadd132nepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x9c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd132nepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfnmadd132nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x9c,0x71,0x7f]
          vfnmadd132nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd132nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x9c,0x72,0x80]
          vfnmadd132nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd132nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfnmadd132nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x9c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd132nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd132nepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x9c,0x35,0x00,0x00,0x00,0x00]
          vfnmadd132nepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfnmadd132nepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x9c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd132nepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfnmadd132nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x9c,0x71,0x7f]
          vfnmadd132nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd132nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x9c,0x72,0x80]
          vfnmadd132nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd132nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfnmadd132nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x9c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd132nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfnmadd132nepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x9c,0x35,0x00,0x00,0x00,0x00]
          vfnmadd132nepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfnmadd132nepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x9c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd132nepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfnmadd132nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x9c,0x71,0x7f]
          vfnmadd132nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd132nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x9c,0x72,0x80]
          vfnmadd132nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd213nepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xac,0xf0]
          vfnmadd213nepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd213nepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xac,0xf0]
          vfnmadd213nepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd213nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xac,0xf0]
          vfnmadd213nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd213nepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xac,0xf0]
          vfnmadd213nepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vfnmadd213nepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xac,0xf0]
          vfnmadd213nepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfnmadd213nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xac,0xf0]
          vfnmadd213nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd213nepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xac,0xf0]
          vfnmadd213nepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vfnmadd213nepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xac,0xf0]
          vfnmadd213nepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfnmadd213nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xac,0xf0]
          vfnmadd213nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd213nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfnmadd213nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xac,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd213nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfnmadd213nepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xac,0x35,0x00,0x00,0x00,0x00]
          vfnmadd213nepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfnmadd213nepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xac,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd213nepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfnmadd213nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xac,0x71,0x7f]
          vfnmadd213nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd213nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xac,0x72,0x80]
          vfnmadd213nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd213nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfnmadd213nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xac,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd213nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd213nepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xac,0x35,0x00,0x00,0x00,0x00]
          vfnmadd213nepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfnmadd213nepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xac,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd213nepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfnmadd213nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xac,0x71,0x7f]
          vfnmadd213nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd213nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xac,0x72,0x80]
          vfnmadd213nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd213nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfnmadd213nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xac,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd213nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfnmadd213nepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xac,0x35,0x00,0x00,0x00,0x00]
          vfnmadd213nepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfnmadd213nepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xac,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd213nepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfnmadd213nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xac,0x71,0x7f]
          vfnmadd213nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd213nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xac,0x72,0x80]
          vfnmadd213nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd231nepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xbc,0xf0]
          vfnmadd231nepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd231nepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xbc,0xf0]
          vfnmadd231nepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd231nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xbc,0xf0]
          vfnmadd231nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd231nepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xbc,0xf0]
          vfnmadd231nepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vfnmadd231nepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xbc,0xf0]
          vfnmadd231nepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfnmadd231nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xbc,0xf0]
          vfnmadd231nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd231nepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xbc,0xf0]
          vfnmadd231nepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vfnmadd231nepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xbc,0xf0]
          vfnmadd231nepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfnmadd231nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xbc,0xf0]
          vfnmadd231nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd231nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfnmadd231nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xbc,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd231nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfnmadd231nepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xbc,0x35,0x00,0x00,0x00,0x00]
          vfnmadd231nepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfnmadd231nepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xbc,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd231nepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfnmadd231nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xbc,0x71,0x7f]
          vfnmadd231nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd231nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xbc,0x72,0x80]
          vfnmadd231nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd231nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfnmadd231nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xbc,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd231nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd231nepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xbc,0x35,0x00,0x00,0x00,0x00]
          vfnmadd231nepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfnmadd231nepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xbc,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd231nepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfnmadd231nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xbc,0x71,0x7f]
          vfnmadd231nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd231nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xbc,0x72,0x80]
          vfnmadd231nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd231nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfnmadd231nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xbc,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd231nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfnmadd231nepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xbc,0x35,0x00,0x00,0x00,0x00]
          vfnmadd231nepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfnmadd231nepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xbc,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd231nepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfnmadd231nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xbc,0x71,0x7f]
          vfnmadd231nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd231nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xbc,0x72,0x80]
          vfnmadd231nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub132nepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x9e,0xf0]
          vfnmsub132nepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub132nepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x9e,0xf0]
          vfnmsub132nepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub132nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x9e,0xf0]
          vfnmsub132nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub132nepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x9e,0xf0]
          vfnmsub132nepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vfnmsub132nepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x9e,0xf0]
          vfnmsub132nepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfnmsub132nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x9e,0xf0]
          vfnmsub132nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub132nepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x9e,0xf0]
          vfnmsub132nepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vfnmsub132nepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x9e,0xf0]
          vfnmsub132nepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfnmsub132nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x9e,0xf0]
          vfnmsub132nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub132nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfnmsub132nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x9e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub132nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfnmsub132nepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x9e,0x35,0x00,0x00,0x00,0x00]
          vfnmsub132nepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfnmsub132nepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x9e,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub132nepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfnmsub132nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x9e,0x71,0x7f]
          vfnmsub132nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub132nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x9e,0x72,0x80]
          vfnmsub132nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub132nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfnmsub132nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x9e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub132nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub132nepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x9e,0x35,0x00,0x00,0x00,0x00]
          vfnmsub132nepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfnmsub132nepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x9e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub132nepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfnmsub132nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x9e,0x71,0x7f]
          vfnmsub132nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub132nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x9e,0x72,0x80]
          vfnmsub132nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub132nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfnmsub132nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x9e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub132nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfnmsub132nepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x9e,0x35,0x00,0x00,0x00,0x00]
          vfnmsub132nepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfnmsub132nepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x9e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub132nepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfnmsub132nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x9e,0x71,0x7f]
          vfnmsub132nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub132nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x9e,0x72,0x80]
          vfnmsub132nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub213nepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xae,0xf0]
          vfnmsub213nepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub213nepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xae,0xf0]
          vfnmsub213nepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub213nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xae,0xf0]
          vfnmsub213nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub213nepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xae,0xf0]
          vfnmsub213nepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vfnmsub213nepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xae,0xf0]
          vfnmsub213nepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfnmsub213nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xae,0xf0]
          vfnmsub213nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub213nepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xae,0xf0]
          vfnmsub213nepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vfnmsub213nepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xae,0xf0]
          vfnmsub213nepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfnmsub213nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xae,0xf0]
          vfnmsub213nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub213nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfnmsub213nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xae,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub213nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfnmsub213nepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xae,0x35,0x00,0x00,0x00,0x00]
          vfnmsub213nepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfnmsub213nepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xae,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub213nepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfnmsub213nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xae,0x71,0x7f]
          vfnmsub213nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub213nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xae,0x72,0x80]
          vfnmsub213nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub213nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfnmsub213nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xae,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub213nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub213nepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xae,0x35,0x00,0x00,0x00,0x00]
          vfnmsub213nepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfnmsub213nepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xae,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub213nepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfnmsub213nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xae,0x71,0x7f]
          vfnmsub213nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub213nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xae,0x72,0x80]
          vfnmsub213nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub213nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfnmsub213nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xae,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub213nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfnmsub213nepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xae,0x35,0x00,0x00,0x00,0x00]
          vfnmsub213nepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfnmsub213nepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xae,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub213nepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfnmsub213nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xae,0x71,0x7f]
          vfnmsub213nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub213nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xae,0x72,0x80]
          vfnmsub213nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub231nepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xbe,0xf0]
          vfnmsub231nepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub231nepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xbe,0xf0]
          vfnmsub231nepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub231nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xbe,0xf0]
          vfnmsub231nepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub231nepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xbe,0xf0]
          vfnmsub231nepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vfnmsub231nepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xbe,0xf0]
          vfnmsub231nepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfnmsub231nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xbe,0xf0]
          vfnmsub231nepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub231nepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xbe,0xf0]
          vfnmsub231nepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vfnmsub231nepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xbe,0xf0]
          vfnmsub231nepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfnmsub231nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xbe,0xf0]
          vfnmsub231nepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub231nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231nepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfnmsub231nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xbe,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub231nepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfnmsub231nepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xbe,0x35,0x00,0x00,0x00,0x00]
          vfnmsub231nepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfnmsub231nepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xbe,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub231nepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfnmsub231nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xbe,0x71,0x7f]
          vfnmsub231nepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub231nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xbe,0x72,0x80]
          vfnmsub231nepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub231nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231nepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfnmsub231nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xbe,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub231nepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub231nepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xbe,0x35,0x00,0x00,0x00,0x00]
          vfnmsub231nepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfnmsub231nepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xbe,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub231nepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfnmsub231nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xbe,0x71,0x7f]
          vfnmsub231nepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub231nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xbe,0x72,0x80]
          vfnmsub231nepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub231nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231nepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfnmsub231nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xbe,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub231nepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfnmsub231nepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xbe,0x35,0x00,0x00,0x00,0x00]
          vfnmsub231nepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfnmsub231nepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xbe,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub231nepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfnmsub231nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xbe,0x71,0x7f]
          vfnmsub231nepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub231nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xbe,0x72,0x80]
          vfnmsub231nepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfpclasspbf16 $123, %zmm23, %k5
// CHECK: encoding: [0x62,0xb3,0x7f,0x48,0x66,0xef,0x7b]
          vfpclasspbf16 $123, %zmm23, %k5

// CHECK: vfpclasspbf16 $123, %zmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xb3,0x7f,0x4f,0x66,0xef,0x7b]
          vfpclasspbf16 $123, %zmm23, %k5 {%k7}

// CHECK: vfpclasspbf16 $123, %ymm23, %k5
// CHECK: encoding: [0x62,0xb3,0x7f,0x28,0x66,0xef,0x7b]
          vfpclasspbf16 $123, %ymm23, %k5

// CHECK: vfpclasspbf16 $123, %ymm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xb3,0x7f,0x2f,0x66,0xef,0x7b]
          vfpclasspbf16 $123, %ymm23, %k5 {%k7}

// CHECK: vfpclasspbf16 $123, %xmm23, %k5
// CHECK: encoding: [0x62,0xb3,0x7f,0x08,0x66,0xef,0x7b]
          vfpclasspbf16 $123, %xmm23, %k5

// CHECK: vfpclasspbf16 $123, %xmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xb3,0x7f,0x0f,0x66,0xef,0x7b]
          vfpclasspbf16 $123, %xmm23, %k5 {%k7}

// CHECK: vfpclasspbf16x  $123, 268435456(%rbp,%r14,8), %k5
// CHECK: encoding: [0x62,0xb3,0x7f,0x08,0x66,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vfpclasspbf16x  $123, 268435456(%rbp,%r14,8), %k5

// CHECK: vfpclasspbf16x  $123, 291(%r8,%rax,4), %k5 {%k7}
// CHECK: encoding: [0x62,0xd3,0x7f,0x0f,0x66,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vfpclasspbf16x  $123, 291(%r8,%rax,4), %k5 {%k7}

// CHECK: vfpclasspbf16  $123, (%rip){1to8}, %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x66,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vfpclasspbf16  $123, (%rip){1to8}, %k5

// CHECK: vfpclasspbf16x  $123, -512(,%rbp,2), %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x66,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vfpclasspbf16x  $123, -512(,%rbp,2), %k5

// CHECK: vfpclasspbf16x  $123, 2032(%rcx), %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x66,0x69,0x7f,0x7b]
          vfpclasspbf16x  $123, 2032(%rcx), %k5 {%k7}

// CHECK: vfpclasspbf16  $123, -256(%rdx){1to8}, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x1f,0x66,0x6a,0x80,0x7b]
          vfpclasspbf16  $123, -256(%rdx){1to8}, %k5 {%k7}

// CHECK: vfpclasspbf16  $123, (%rip){1to16}, %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x66,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vfpclasspbf16  $123, (%rip){1to16}, %k5

// CHECK: vfpclasspbf16y  $123, -1024(,%rbp,2), %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x66,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vfpclasspbf16y  $123, -1024(,%rbp,2), %k5

// CHECK: vfpclasspbf16y  $123, 4064(%rcx), %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x66,0x69,0x7f,0x7b]
          vfpclasspbf16y  $123, 4064(%rcx), %k5 {%k7}

// CHECK: vfpclasspbf16  $123, -256(%rdx){1to16}, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x3f,0x66,0x6a,0x80,0x7b]
          vfpclasspbf16  $123, -256(%rdx){1to16}, %k5 {%k7}

// CHECK: vfpclasspbf16  $123, (%rip){1to32}, %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x66,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vfpclasspbf16  $123, (%rip){1to32}, %k5

// CHECK: vfpclasspbf16z  $123, -2048(,%rbp,2), %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x66,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vfpclasspbf16z  $123, -2048(,%rbp,2), %k5

// CHECK: vfpclasspbf16z  $123, 8128(%rcx), %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x66,0x69,0x7f,0x7b]
          vfpclasspbf16z  $123, 8128(%rcx), %k5 {%k7}

// CHECK: vfpclasspbf16  $123, -256(%rdx){1to32}, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x5f,0x66,0x6a,0x80,0x7b]
          vfpclasspbf16  $123, -256(%rdx){1to32}, %k5 {%k7}

// CHECK: vgetexppbf16 %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x42,0xf7]
          vgetexppbf16 %xmm23, %xmm22

// CHECK: vgetexppbf16 %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x42,0xf7]
          vgetexppbf16 %xmm23, %xmm22 {%k7}

// CHECK: vgetexppbf16 %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x42,0xf7]
          vgetexppbf16 %xmm23, %xmm22 {%k7} {z}

// CHECK: vgetexppbf16 %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x42,0xf7]
          vgetexppbf16 %zmm23, %zmm22

// CHECK: vgetexppbf16 %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x42,0xf7]
          vgetexppbf16 %zmm23, %zmm22 {%k7}

// CHECK: vgetexppbf16 %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7d,0xcf,0x42,0xf7]
          vgetexppbf16 %zmm23, %zmm22 {%k7} {z}

// CHECK: vgetexppbf16 %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x42,0xf7]
          vgetexppbf16 %ymm23, %ymm22

// CHECK: vgetexppbf16 %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x42,0xf7]
          vgetexppbf16 %ymm23, %ymm22 {%k7}

// CHECK: vgetexppbf16 %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7d,0xaf,0x42,0xf7]
          vgetexppbf16 %ymm23, %ymm22 {%k7} {z}

// CHECK: vgetexppbf16  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexppbf16  268435456(%rbp,%r14,8), %xmm22

// CHECK: vgetexppbf16  291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x42,0xb4,0x80,0x23,0x01,0x00,0x00]
          vgetexppbf16  291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vgetexppbf16  (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x42,0x35,0x00,0x00,0x00,0x00]
          vgetexppbf16  (%rip){1to8}, %xmm22

// CHECK: vgetexppbf16  -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x42,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vgetexppbf16  -512(,%rbp,2), %xmm22

// CHECK: vgetexppbf16  2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x42,0x71,0x7f]
          vgetexppbf16  2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vgetexppbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x42,0x72,0x80]
          vgetexppbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vgetexppbf16  268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexppbf16  268435456(%rbp,%r14,8), %ymm22

// CHECK: vgetexppbf16  291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x42,0xb4,0x80,0x23,0x01,0x00,0x00]
          vgetexppbf16  291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vgetexppbf16  (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x42,0x35,0x00,0x00,0x00,0x00]
          vgetexppbf16  (%rip){1to16}, %ymm22

// CHECK: vgetexppbf16  -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x42,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vgetexppbf16  -1024(,%rbp,2), %ymm22

// CHECK: vgetexppbf16  4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x42,0x71,0x7f]
          vgetexppbf16  4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vgetexppbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x42,0x72,0x80]
          vgetexppbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vgetexppbf16  268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexppbf16  268435456(%rbp,%r14,8), %zmm22

// CHECK: vgetexppbf16  291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x42,0xb4,0x80,0x23,0x01,0x00,0x00]
          vgetexppbf16  291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vgetexppbf16  (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x42,0x35,0x00,0x00,0x00,0x00]
          vgetexppbf16  (%rip){1to32}, %zmm22

// CHECK: vgetexppbf16  -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x42,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vgetexppbf16  -2048(,%rbp,2), %zmm22

// CHECK: vgetexppbf16  8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x42,0x71,0x7f]
          vgetexppbf16  8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vgetexppbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x42,0x72,0x80]
          vgetexppbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vgetmantpbf16 $123, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x26,0xf7,0x7b]
          vgetmantpbf16 $123, %zmm23, %zmm22

// CHECK: vgetmantpbf16 $123, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x4f,0x26,0xf7,0x7b]
          vgetmantpbf16 $123, %zmm23, %zmm22 {%k7}

// CHECK: vgetmantpbf16 $123, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0xcf,0x26,0xf7,0x7b]
          vgetmantpbf16 $123, %zmm23, %zmm22 {%k7} {z}

// CHECK: vgetmantpbf16 $123, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x26,0xf7,0x7b]
          vgetmantpbf16 $123, %ymm23, %ymm22

// CHECK: vgetmantpbf16 $123, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x2f,0x26,0xf7,0x7b]
          vgetmantpbf16 $123, %ymm23, %ymm22 {%k7}

// CHECK: vgetmantpbf16 $123, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0xaf,0x26,0xf7,0x7b]
          vgetmantpbf16 $123, %ymm23, %ymm22 {%k7} {z}

// CHECK: vgetmantpbf16 $123, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x26,0xf7,0x7b]
          vgetmantpbf16 $123, %xmm23, %xmm22

// CHECK: vgetmantpbf16 $123, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x0f,0x26,0xf7,0x7b]
          vgetmantpbf16 $123, %xmm23, %xmm22 {%k7}

// CHECK: vgetmantpbf16 $123, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0x8f,0x26,0xf7,0x7b]
          vgetmantpbf16 $123, %xmm23, %xmm22 {%k7} {z}

// CHECK: vgetmantpbf16  $123, 268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantpbf16  $123, 268435456(%rbp,%r14,8), %xmm22

// CHECK: vgetmantpbf16  $123, 291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x0f,0x26,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vgetmantpbf16  $123, 291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vgetmantpbf16  $123, (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x18,0x26,0x35,0x00,0x00,0x00,0x00,0x7b]
          vgetmantpbf16  $123, (%rip){1to8}, %xmm22

// CHECK: vgetmantpbf16  $123, -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x08,0x26,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vgetmantpbf16  $123, -512(,%rbp,2), %xmm22

// CHECK: vgetmantpbf16  $123, 2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0x8f,0x26,0x71,0x7f,0x7b]
          vgetmantpbf16  $123, 2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vgetmantpbf16  $123, -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0x9f,0x26,0x72,0x80,0x7b]
          vgetmantpbf16  $123, -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vgetmantpbf16  $123, 268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantpbf16  $123, 268435456(%rbp,%r14,8), %ymm22

// CHECK: vgetmantpbf16  $123, 291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x2f,0x26,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vgetmantpbf16  $123, 291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vgetmantpbf16  $123, (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x38,0x26,0x35,0x00,0x00,0x00,0x00,0x7b]
          vgetmantpbf16  $123, (%rip){1to16}, %ymm22

// CHECK: vgetmantpbf16  $123, -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x28,0x26,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vgetmantpbf16  $123, -1024(,%rbp,2), %ymm22

// CHECK: vgetmantpbf16  $123, 4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xaf,0x26,0x71,0x7f,0x7b]
          vgetmantpbf16  $123, 4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vgetmantpbf16  $123, -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xbf,0x26,0x72,0x80,0x7b]
          vgetmantpbf16  $123, -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vgetmantpbf16  $123, 268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantpbf16  $123, 268435456(%rbp,%r14,8), %zmm22

// CHECK: vgetmantpbf16  $123, 291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x4f,0x26,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vgetmantpbf16  $123, 291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vgetmantpbf16  $123, (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x58,0x26,0x35,0x00,0x00,0x00,0x00,0x7b]
          vgetmantpbf16  $123, (%rip){1to32}, %zmm22

// CHECK: vgetmantpbf16  $123, -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x26,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vgetmantpbf16  $123, -2048(,%rbp,2), %zmm22

// CHECK: vgetmantpbf16  $123, 8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xcf,0x26,0x71,0x7f,0x7b]
          vgetmantpbf16  $123, 8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vgetmantpbf16  $123, -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xdf,0x26,0x72,0x80,0x7b]
          vgetmantpbf16  $123, -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vmaxpbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5f,0xf0]
          vmaxpbf16 %ymm24, %ymm23, %ymm22

// CHECK: vmaxpbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5f,0xf0]
          vmaxpbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vmaxpbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5f,0xf0]
          vmaxpbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmaxpbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5f,0xf0]
          vmaxpbf16 %zmm24, %zmm23, %zmm22

// CHECK: vmaxpbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5f,0xf0]
          vmaxpbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vmaxpbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5f,0xf0]
          vmaxpbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vmaxpbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5f,0xf0]
          vmaxpbf16 %xmm24, %xmm23, %xmm22

// CHECK: vmaxpbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5f,0xf0]
          vmaxpbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vmaxpbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5f,0xf0]
          vmaxpbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vmaxpbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxpbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vmaxpbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmaxpbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vmaxpbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5f,0x35,0x00,0x00,0x00,0x00]
          vmaxpbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vmaxpbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5f,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vmaxpbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vmaxpbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5f,0x71,0x7f]
          vmaxpbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vmaxpbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5f,0x72,0x80]
          vmaxpbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vmaxpbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxpbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vmaxpbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmaxpbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vmaxpbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5f,0x35,0x00,0x00,0x00,0x00]
          vmaxpbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vmaxpbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5f,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vmaxpbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vmaxpbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5f,0x71,0x7f]
          vmaxpbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vmaxpbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5f,0x72,0x80]
          vmaxpbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmaxpbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxpbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vmaxpbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmaxpbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vmaxpbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5f,0x35,0x00,0x00,0x00,0x00]
          vmaxpbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vmaxpbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5f,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vmaxpbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vmaxpbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5f,0x71,0x7f]
          vmaxpbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vmaxpbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5f,0x72,0x80]
          vmaxpbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminpbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5d,0xf0]
          vminpbf16 %ymm24, %ymm23, %ymm22

// CHECK: vminpbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5d,0xf0]
          vminpbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vminpbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5d,0xf0]
          vminpbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminpbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5d,0xf0]
          vminpbf16 %zmm24, %zmm23, %zmm22

// CHECK: vminpbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5d,0xf0]
          vminpbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vminpbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5d,0xf0]
          vminpbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vminpbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5d,0xf0]
          vminpbf16 %xmm24, %xmm23, %xmm22

// CHECK: vminpbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5d,0xf0]
          vminpbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vminpbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5d,0xf0]
          vminpbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminpbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminpbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vminpbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vminpbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vminpbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5d,0x35,0x00,0x00,0x00,0x00]
          vminpbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vminpbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5d,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vminpbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vminpbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5d,0x71,0x7f]
          vminpbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vminpbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5d,0x72,0x80]
          vminpbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vminpbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminpbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vminpbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vminpbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vminpbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5d,0x35,0x00,0x00,0x00,0x00]
          vminpbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vminpbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5d,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vminpbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vminpbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5d,0x71,0x7f]
          vminpbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vminpbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5d,0x72,0x80]
          vminpbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminpbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminpbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vminpbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vminpbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vminpbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5d,0x35,0x00,0x00,0x00,0x00]
          vminpbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vminpbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5d,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vminpbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vminpbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5d,0x71,0x7f]
          vminpbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vminpbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5d,0x72,0x80]
          vminpbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vmulnepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x59,0xf0]
          vmulnepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vmulnepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x59,0xf0]
          vmulnepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vmulnepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x59,0xf0]
          vmulnepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmulnepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x59,0xf0]
          vmulnepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vmulnepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x59,0xf0]
          vmulnepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vmulnepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x59,0xf0]
          vmulnepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vmulnepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x59,0xf0]
          vmulnepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vmulnepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x59,0xf0]
          vmulnepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vmulnepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x59,0xf0]
          vmulnepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vmulnepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulnepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vmulnepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x59,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmulnepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vmulnepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x59,0x35,0x00,0x00,0x00,0x00]
          vmulnepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vmulnepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x59,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vmulnepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vmulnepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x59,0x71,0x7f]
          vmulnepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vmulnepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x59,0x72,0x80]
          vmulnepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vmulnepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulnepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vmulnepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x59,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmulnepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vmulnepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x59,0x35,0x00,0x00,0x00,0x00]
          vmulnepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vmulnepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x59,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vmulnepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vmulnepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x59,0x71,0x7f]
          vmulnepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vmulnepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x59,0x72,0x80]
          vmulnepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmulnepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulnepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vmulnepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x59,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmulnepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vmulnepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x59,0x35,0x00,0x00,0x00,0x00]
          vmulnepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vmulnepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x59,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vmulnepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vmulnepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x59,0x71,0x7f]
          vmulnepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vmulnepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x59,0x72,0x80]
          vmulnepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vrcppbf16 %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4c,0xf7]
          vrcppbf16 %xmm23, %xmm22

// CHECK: vrcppbf16 %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x7c,0x0f,0x4c,0xf7]
          vrcppbf16 %xmm23, %xmm22 {%k7}

// CHECK: vrcppbf16 %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x7c,0x8f,0x4c,0xf7]
          vrcppbf16 %xmm23, %xmm22 {%k7} {z}

// CHECK: vrcppbf16 %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4c,0xf7]
          vrcppbf16 %zmm23, %zmm22

// CHECK: vrcppbf16 %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x7c,0x4f,0x4c,0xf7]
          vrcppbf16 %zmm23, %zmm22 {%k7}

// CHECK: vrcppbf16 %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x7c,0xcf,0x4c,0xf7]
          vrcppbf16 %zmm23, %zmm22 {%k7} {z}

// CHECK: vrcppbf16 %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4c,0xf7]
          vrcppbf16 %ymm23, %ymm22

// CHECK: vrcppbf16 %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x7c,0x2f,0x4c,0xf7]
          vrcppbf16 %ymm23, %ymm22 {%k7}

// CHECK: vrcppbf16 %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x7c,0xaf,0x4c,0xf7]
          vrcppbf16 %ymm23, %ymm22 {%k7} {z}

// CHECK: vrcppbf16  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcppbf16  268435456(%rbp,%r14,8), %xmm22

// CHECK: vrcppbf16  291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x7c,0x0f,0x4c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrcppbf16  291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vrcppbf16  (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x18,0x4c,0x35,0x00,0x00,0x00,0x00]
          vrcppbf16  (%rip){1to8}, %xmm22

// CHECK: vrcppbf16  -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x08,0x4c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vrcppbf16  -512(,%rbp,2), %xmm22

// CHECK: vrcppbf16  2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0x8f,0x4c,0x71,0x7f]
          vrcppbf16  2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vrcppbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0x9f,0x4c,0x72,0x80]
          vrcppbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vrcppbf16  268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcppbf16  268435456(%rbp,%r14,8), %ymm22

// CHECK: vrcppbf16  291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x7c,0x2f,0x4c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrcppbf16  291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vrcppbf16  (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x38,0x4c,0x35,0x00,0x00,0x00,0x00]
          vrcppbf16  (%rip){1to16}, %ymm22

// CHECK: vrcppbf16  -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x28,0x4c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vrcppbf16  -1024(,%rbp,2), %ymm22

// CHECK: vrcppbf16  4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xaf,0x4c,0x71,0x7f]
          vrcppbf16  4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vrcppbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xbf,0x4c,0x72,0x80]
          vrcppbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vrcppbf16  268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcppbf16  268435456(%rbp,%r14,8), %zmm22

// CHECK: vrcppbf16  291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x7c,0x4f,0x4c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrcppbf16  291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vrcppbf16  (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x58,0x4c,0x35,0x00,0x00,0x00,0x00]
          vrcppbf16  (%rip){1to32}, %zmm22

// CHECK: vrcppbf16  -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x48,0x4c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vrcppbf16  -2048(,%rbp,2), %zmm22

// CHECK: vrcppbf16  8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xcf,0x4c,0x71,0x7f]
          vrcppbf16  8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vrcppbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xdf,0x4c,0x72,0x80]
          vrcppbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vreducenepbf16 $123, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x56,0xf7,0x7b]
          vreducenepbf16 $123, %zmm23, %zmm22

// CHECK: vreducenepbf16 $123, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x4f,0x56,0xf7,0x7b]
          vreducenepbf16 $123, %zmm23, %zmm22 {%k7}

// CHECK: vreducenepbf16 $123, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0xcf,0x56,0xf7,0x7b]
          vreducenepbf16 $123, %zmm23, %zmm22 {%k7} {z}

// CHECK: vreducenepbf16 $123, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x56,0xf7,0x7b]
          vreducenepbf16 $123, %ymm23, %ymm22

// CHECK: vreducenepbf16 $123, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x2f,0x56,0xf7,0x7b]
          vreducenepbf16 $123, %ymm23, %ymm22 {%k7}

// CHECK: vreducenepbf16 $123, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0xaf,0x56,0xf7,0x7b]
          vreducenepbf16 $123, %ymm23, %ymm22 {%k7} {z}

// CHECK: vreducenepbf16 $123, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x56,0xf7,0x7b]
          vreducenepbf16 $123, %xmm23, %xmm22

// CHECK: vreducenepbf16 $123, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x0f,0x56,0xf7,0x7b]
          vreducenepbf16 $123, %xmm23, %xmm22 {%k7}

// CHECK: vreducenepbf16 $123, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0x8f,0x56,0xf7,0x7b]
          vreducenepbf16 $123, %xmm23, %xmm22 {%k7} {z}

// CHECK: vreducenepbf16  $123, 268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreducenepbf16  $123, 268435456(%rbp,%r14,8), %xmm22

// CHECK: vreducenepbf16  $123, 291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x0f,0x56,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vreducenepbf16  $123, 291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vreducenepbf16  $123, (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x18,0x56,0x35,0x00,0x00,0x00,0x00,0x7b]
          vreducenepbf16  $123, (%rip){1to8}, %xmm22

// CHECK: vreducenepbf16  $123, -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x08,0x56,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vreducenepbf16  $123, -512(,%rbp,2), %xmm22

// CHECK: vreducenepbf16  $123, 2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0x8f,0x56,0x71,0x7f,0x7b]
          vreducenepbf16  $123, 2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vreducenepbf16  $123, -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0x9f,0x56,0x72,0x80,0x7b]
          vreducenepbf16  $123, -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vreducenepbf16  $123, 268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreducenepbf16  $123, 268435456(%rbp,%r14,8), %ymm22

// CHECK: vreducenepbf16  $123, 291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x2f,0x56,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vreducenepbf16  $123, 291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vreducenepbf16  $123, (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x38,0x56,0x35,0x00,0x00,0x00,0x00,0x7b]
          vreducenepbf16  $123, (%rip){1to16}, %ymm22

// CHECK: vreducenepbf16  $123, -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x28,0x56,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vreducenepbf16  $123, -1024(,%rbp,2), %ymm22

// CHECK: vreducenepbf16  $123, 4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xaf,0x56,0x71,0x7f,0x7b]
          vreducenepbf16  $123, 4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vreducenepbf16  $123, -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xbf,0x56,0x72,0x80,0x7b]
          vreducenepbf16  $123, -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vreducenepbf16  $123, 268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreducenepbf16  $123, 268435456(%rbp,%r14,8), %zmm22

// CHECK: vreducenepbf16  $123, 291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x4f,0x56,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vreducenepbf16  $123, 291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vreducenepbf16  $123, (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x58,0x56,0x35,0x00,0x00,0x00,0x00,0x7b]
          vreducenepbf16  $123, (%rip){1to32}, %zmm22

// CHECK: vreducenepbf16  $123, -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x56,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vreducenepbf16  $123, -2048(,%rbp,2), %zmm22

// CHECK: vreducenepbf16  $123, 8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xcf,0x56,0x71,0x7f,0x7b]
          vreducenepbf16  $123, 8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vreducenepbf16  $123, -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xdf,0x56,0x72,0x80,0x7b]
          vreducenepbf16  $123, -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vrndscalenepbf16 $123, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x08,0xf7,0x7b]
          vrndscalenepbf16 $123, %zmm23, %zmm22

// CHECK: vrndscalenepbf16 $123, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x4f,0x08,0xf7,0x7b]
          vrndscalenepbf16 $123, %zmm23, %zmm22 {%k7}

// CHECK: vrndscalenepbf16 $123, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0xcf,0x08,0xf7,0x7b]
          vrndscalenepbf16 $123, %zmm23, %zmm22 {%k7} {z}

// CHECK: vrndscalenepbf16 $123, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x08,0xf7,0x7b]
          vrndscalenepbf16 $123, %ymm23, %ymm22

// CHECK: vrndscalenepbf16 $123, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x2f,0x08,0xf7,0x7b]
          vrndscalenepbf16 $123, %ymm23, %ymm22 {%k7}

// CHECK: vrndscalenepbf16 $123, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0xaf,0x08,0xf7,0x7b]
          vrndscalenepbf16 $123, %ymm23, %ymm22 {%k7} {z}

// CHECK: vrndscalenepbf16 $123, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x08,0xf7,0x7b]
          vrndscalenepbf16 $123, %xmm23, %xmm22

// CHECK: vrndscalenepbf16 $123, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x0f,0x08,0xf7,0x7b]
          vrndscalenepbf16 $123, %xmm23, %xmm22 {%k7}

// CHECK: vrndscalenepbf16 $123, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0x8f,0x08,0xf7,0x7b]
          vrndscalenepbf16 $123, %xmm23, %xmm22 {%k7} {z}

// CHECK: vrndscalenepbf16  $123, 268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscalenepbf16  $123, 268435456(%rbp,%r14,8), %xmm22

// CHECK: vrndscalenepbf16  $123, 291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x0f,0x08,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vrndscalenepbf16  $123, 291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vrndscalenepbf16  $123, (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x18,0x08,0x35,0x00,0x00,0x00,0x00,0x7b]
          vrndscalenepbf16  $123, (%rip){1to8}, %xmm22

// CHECK: vrndscalenepbf16  $123, -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x08,0x08,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vrndscalenepbf16  $123, -512(,%rbp,2), %xmm22

// CHECK: vrndscalenepbf16  $123, 2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0x8f,0x08,0x71,0x7f,0x7b]
          vrndscalenepbf16  $123, 2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vrndscalenepbf16  $123, -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0x9f,0x08,0x72,0x80,0x7b]
          vrndscalenepbf16  $123, -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vrndscalenepbf16  $123, 268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscalenepbf16  $123, 268435456(%rbp,%r14,8), %ymm22

// CHECK: vrndscalenepbf16  $123, 291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x2f,0x08,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vrndscalenepbf16  $123, 291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vrndscalenepbf16  $123, (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x38,0x08,0x35,0x00,0x00,0x00,0x00,0x7b]
          vrndscalenepbf16  $123, (%rip){1to16}, %ymm22

// CHECK: vrndscalenepbf16  $123, -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x28,0x08,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vrndscalenepbf16  $123, -1024(,%rbp,2), %ymm22

// CHECK: vrndscalenepbf16  $123, 4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xaf,0x08,0x71,0x7f,0x7b]
          vrndscalenepbf16  $123, 4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vrndscalenepbf16  $123, -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xbf,0x08,0x72,0x80,0x7b]
          vrndscalenepbf16  $123, -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vrndscalenepbf16  $123, 268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscalenepbf16  $123, 268435456(%rbp,%r14,8), %zmm22

// CHECK: vrndscalenepbf16  $123, 291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x4f,0x08,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vrndscalenepbf16  $123, 291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vrndscalenepbf16  $123, (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x58,0x08,0x35,0x00,0x00,0x00,0x00,0x7b]
          vrndscalenepbf16  $123, (%rip){1to32}, %zmm22

// CHECK: vrndscalenepbf16  $123, -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x08,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vrndscalenepbf16  $123, -2048(,%rbp,2), %zmm22

// CHECK: vrndscalenepbf16  $123, 8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xcf,0x08,0x71,0x7f,0x7b]
          vrndscalenepbf16  $123, 8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vrndscalenepbf16  $123, -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xdf,0x08,0x72,0x80,0x7b]
          vrndscalenepbf16  $123, -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vrsqrtpbf16 %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4e,0xf7]
          vrsqrtpbf16 %xmm23, %xmm22

// CHECK: vrsqrtpbf16 %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x7c,0x0f,0x4e,0xf7]
          vrsqrtpbf16 %xmm23, %xmm22 {%k7}

// CHECK: vrsqrtpbf16 %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x7c,0x8f,0x4e,0xf7]
          vrsqrtpbf16 %xmm23, %xmm22 {%k7} {z}

// CHECK: vrsqrtpbf16 %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4e,0xf7]
          vrsqrtpbf16 %zmm23, %zmm22

// CHECK: vrsqrtpbf16 %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x7c,0x4f,0x4e,0xf7]
          vrsqrtpbf16 %zmm23, %zmm22 {%k7}

// CHECK: vrsqrtpbf16 %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x7c,0xcf,0x4e,0xf7]
          vrsqrtpbf16 %zmm23, %zmm22 {%k7} {z}

// CHECK: vrsqrtpbf16 %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4e,0xf7]
          vrsqrtpbf16 %ymm23, %ymm22

// CHECK: vrsqrtpbf16 %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x7c,0x2f,0x4e,0xf7]
          vrsqrtpbf16 %ymm23, %ymm22 {%k7}

// CHECK: vrsqrtpbf16 %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x7c,0xaf,0x4e,0xf7]
          vrsqrtpbf16 %ymm23, %ymm22 {%k7} {z}

// CHECK: vrsqrtpbf16  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtpbf16  268435456(%rbp,%r14,8), %xmm22

// CHECK: vrsqrtpbf16  291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x7c,0x0f,0x4e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrsqrtpbf16  291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vrsqrtpbf16  (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x18,0x4e,0x35,0x00,0x00,0x00,0x00]
          vrsqrtpbf16  (%rip){1to8}, %xmm22

// CHECK: vrsqrtpbf16  -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x08,0x4e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vrsqrtpbf16  -512(,%rbp,2), %xmm22

// CHECK: vrsqrtpbf16  2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0x8f,0x4e,0x71,0x7f]
          vrsqrtpbf16  2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vrsqrtpbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0x9f,0x4e,0x72,0x80]
          vrsqrtpbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vrsqrtpbf16  268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtpbf16  268435456(%rbp,%r14,8), %ymm22

// CHECK: vrsqrtpbf16  291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x7c,0x2f,0x4e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrsqrtpbf16  291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vrsqrtpbf16  (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x38,0x4e,0x35,0x00,0x00,0x00,0x00]
          vrsqrtpbf16  (%rip){1to16}, %ymm22

// CHECK: vrsqrtpbf16  -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x28,0x4e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vrsqrtpbf16  -1024(,%rbp,2), %ymm22

// CHECK: vrsqrtpbf16  4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xaf,0x4e,0x71,0x7f]
          vrsqrtpbf16  4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vrsqrtpbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xbf,0x4e,0x72,0x80]
          vrsqrtpbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vrsqrtpbf16  268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtpbf16  268435456(%rbp,%r14,8), %zmm22

// CHECK: vrsqrtpbf16  291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x7c,0x4f,0x4e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrsqrtpbf16  291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vrsqrtpbf16  (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x58,0x4e,0x35,0x00,0x00,0x00,0x00]
          vrsqrtpbf16  (%rip){1to32}, %zmm22

// CHECK: vrsqrtpbf16  -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x48,0x4e,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vrsqrtpbf16  -2048(,%rbp,2), %zmm22

// CHECK: vrsqrtpbf16  8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xcf,0x4e,0x71,0x7f]
          vrsqrtpbf16  8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vrsqrtpbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xdf,0x4e,0x72,0x80]
          vrsqrtpbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vscalefpbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x2c,0xf0]
          vscalefpbf16 %ymm24, %ymm23, %ymm22

// CHECK: vscalefpbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x2c,0xf0]
          vscalefpbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vscalefpbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x2c,0xf0]
          vscalefpbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vscalefpbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x2c,0xf0]
          vscalefpbf16 %zmm24, %zmm23, %zmm22

// CHECK: vscalefpbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x2c,0xf0]
          vscalefpbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vscalefpbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x2c,0xf0]
          vscalefpbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vscalefpbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x2c,0xf0]
          vscalefpbf16 %xmm24, %xmm23, %xmm22

// CHECK: vscalefpbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x2c,0xf0]
          vscalefpbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vscalefpbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x2c,0xf0]
          vscalefpbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vscalefpbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefpbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vscalefpbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x2c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vscalefpbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vscalefpbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x2c,0x35,0x00,0x00,0x00,0x00]
          vscalefpbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vscalefpbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x2c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vscalefpbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vscalefpbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x2c,0x71,0x7f]
          vscalefpbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vscalefpbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x2c,0x72,0x80]
          vscalefpbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vscalefpbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefpbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vscalefpbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x2c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vscalefpbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vscalefpbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x2c,0x35,0x00,0x00,0x00,0x00]
          vscalefpbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vscalefpbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x2c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vscalefpbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vscalefpbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x2c,0x71,0x7f]
          vscalefpbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vscalefpbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x2c,0x72,0x80]
          vscalefpbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vscalefpbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefpbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vscalefpbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x2c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vscalefpbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vscalefpbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x2c,0x35,0x00,0x00,0x00,0x00]
          vscalefpbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vscalefpbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x2c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vscalefpbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vscalefpbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x2c,0x71,0x7f]
          vscalefpbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vscalefpbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x2c,0x72,0x80]
          vscalefpbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vsqrtnepbf16 %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x51,0xf7]
          vsqrtnepbf16 %xmm23, %xmm22

// CHECK: vsqrtnepbf16 %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x51,0xf7]
          vsqrtnepbf16 %xmm23, %xmm22 {%k7}

// CHECK: vsqrtnepbf16 %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x51,0xf7]
          vsqrtnepbf16 %xmm23, %xmm22 {%k7} {z}

// CHECK: vsqrtnepbf16 %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x51,0xf7]
          vsqrtnepbf16 %zmm23, %zmm22

// CHECK: vsqrtnepbf16 %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x51,0xf7]
          vsqrtnepbf16 %zmm23, %zmm22 {%k7}

// CHECK: vsqrtnepbf16 %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7d,0xcf,0x51,0xf7]
          vsqrtnepbf16 %zmm23, %zmm22 {%k7} {z}

// CHECK: vsqrtnepbf16 %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x51,0xf7]
          vsqrtnepbf16 %ymm23, %ymm22

// CHECK: vsqrtnepbf16 %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x51,0xf7]
          vsqrtnepbf16 %ymm23, %ymm22 {%k7}

// CHECK: vsqrtnepbf16 %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7d,0xaf,0x51,0xf7]
          vsqrtnepbf16 %ymm23, %ymm22 {%k7} {z}

// CHECK: vsqrtnepbf16  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtnepbf16  268435456(%rbp,%r14,8), %xmm22

// CHECK: vsqrtnepbf16  291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsqrtnepbf16  291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vsqrtnepbf16  (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x51,0x35,0x00,0x00,0x00,0x00]
          vsqrtnepbf16  (%rip){1to8}, %xmm22

// CHECK: vsqrtnepbf16  -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x51,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vsqrtnepbf16  -512(,%rbp,2), %xmm22

// CHECK: vsqrtnepbf16  2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x51,0x71,0x7f]
          vsqrtnepbf16  2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vsqrtnepbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x51,0x72,0x80]
          vsqrtnepbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vsqrtnepbf16  268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtnepbf16  268435456(%rbp,%r14,8), %ymm22

// CHECK: vsqrtnepbf16  291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsqrtnepbf16  291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vsqrtnepbf16  (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x51,0x35,0x00,0x00,0x00,0x00]
          vsqrtnepbf16  (%rip){1to16}, %ymm22

// CHECK: vsqrtnepbf16  -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x51,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vsqrtnepbf16  -1024(,%rbp,2), %ymm22

// CHECK: vsqrtnepbf16  4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x51,0x71,0x7f]
          vsqrtnepbf16  4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vsqrtnepbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x51,0x72,0x80]
          vsqrtnepbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vsqrtnepbf16  268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtnepbf16  268435456(%rbp,%r14,8), %zmm22

// CHECK: vsqrtnepbf16  291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsqrtnepbf16  291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vsqrtnepbf16  (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x51,0x35,0x00,0x00,0x00,0x00]
          vsqrtnepbf16  (%rip){1to32}, %zmm22

// CHECK: vsqrtnepbf16  -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x51,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vsqrtnepbf16  -2048(,%rbp,2), %zmm22

// CHECK: vsqrtnepbf16  8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x51,0x71,0x7f]
          vsqrtnepbf16  8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vsqrtnepbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x51,0x72,0x80]
          vsqrtnepbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vsubnepbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5c,0xf0]
          vsubnepbf16 %ymm24, %ymm23, %ymm22

// CHECK: vsubnepbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5c,0xf0]
          vsubnepbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vsubnepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5c,0xf0]
          vsubnepbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vsubnepbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5c,0xf0]
          vsubnepbf16 %zmm24, %zmm23, %zmm22

// CHECK: vsubnepbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5c,0xf0]
          vsubnepbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vsubnepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5c,0xf0]
          vsubnepbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vsubnepbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5c,0xf0]
          vsubnepbf16 %xmm24, %xmm23, %xmm22

// CHECK: vsubnepbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5c,0xf0]
          vsubnepbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vsubnepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5c,0xf0]
          vsubnepbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vsubnepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubnepbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vsubnepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsubnepbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vsubnepbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5c,0x35,0x00,0x00,0x00,0x00]
          vsubnepbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vsubnepbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vsubnepbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vsubnepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5c,0x71,0x7f]
          vsubnepbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vsubnepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5c,0x72,0x80]
          vsubnepbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vsubnepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubnepbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vsubnepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsubnepbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vsubnepbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5c,0x35,0x00,0x00,0x00,0x00]
          vsubnepbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vsubnepbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vsubnepbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vsubnepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5c,0x71,0x7f]
          vsubnepbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vsubnepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5c,0x72,0x80]
          vsubnepbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vsubnepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubnepbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vsubnepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsubnepbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vsubnepbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5c,0x35,0x00,0x00,0x00,0x00]
          vsubnepbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vsubnepbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vsubnepbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vsubnepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5c,0x71,0x7f]
          vsubnepbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vsubnepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5c,0x72,0x80]
          vsubnepbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

