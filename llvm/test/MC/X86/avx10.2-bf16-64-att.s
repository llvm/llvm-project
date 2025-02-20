// RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s

// CHECK: vaddbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x58,0xf0]
          vaddbf16 %ymm24, %ymm23, %ymm22

// CHECK: vaddbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x58,0xf0]
          vaddbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vaddbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x58,0xf0]
          vaddbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vaddbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x58,0xf0]
          vaddbf16 %zmm24, %zmm23, %zmm22

// CHECK: vaddbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x58,0xf0]
          vaddbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vaddbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x58,0xf0]
          vaddbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vaddbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x58,0xf0]
          vaddbf16 %xmm24, %xmm23, %xmm22

// CHECK: vaddbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x58,0xf0]
          vaddbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vaddbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x58,0xf0]
          vaddbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vaddbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vaddbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x58,0xb4,0x80,0x23,0x01,0x00,0x00]
          vaddbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vaddbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x58,0x35,0x00,0x00,0x00,0x00]
          vaddbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vaddbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x58,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vaddbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vaddbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x58,0x71,0x7f]
          vaddbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vaddbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x58,0x72,0x80]
          vaddbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vaddbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vaddbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x58,0xb4,0x80,0x23,0x01,0x00,0x00]
          vaddbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vaddbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x58,0x35,0x00,0x00,0x00,0x00]
          vaddbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vaddbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x58,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vaddbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vaddbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x58,0x71,0x7f]
          vaddbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vaddbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x58,0x72,0x80]
          vaddbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vaddbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vaddbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x58,0xb4,0x80,0x23,0x01,0x00,0x00]
          vaddbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vaddbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x58,0x35,0x00,0x00,0x00,0x00]
          vaddbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vaddbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x58,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vaddbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vaddbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x58,0x71,0x7f]
          vaddbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vaddbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x58,0x72,0x80]
          vaddbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcmpbf16 $123, %ymm24, %ymm23, %k5
// CHECK: encoding: [0x62,0x93,0x47,0x20,0xc2,0xe8,0x7b]
          vcmpbf16 $123, %ymm24, %ymm23, %k5

// CHECK: vcmpbf16 $123, %ymm24, %ymm23, %k5 {%k7}
// CHECK: encoding: [0x62,0x93,0x47,0x27,0xc2,0xe8,0x7b]
          vcmpbf16 $123, %ymm24, %ymm23, %k5 {%k7}

// CHECK: vcmpbf16 $123, %xmm24, %xmm23, %k5
// CHECK: encoding: [0x62,0x93,0x47,0x00,0xc2,0xe8,0x7b]
          vcmpbf16 $123, %xmm24, %xmm23, %k5

// CHECK: vcmpbf16 $123, %xmm24, %xmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0x93,0x47,0x07,0xc2,0xe8,0x7b]
          vcmpbf16 $123, %xmm24, %xmm23, %k5 {%k7}

// CHECK: vcmpbf16 $123, %zmm24, %zmm23, %k5
// CHECK: encoding: [0x62,0x93,0x47,0x40,0xc2,0xe8,0x7b]
          vcmpbf16 $123, %zmm24, %zmm23, %k5

// CHECK: vcmpbf16 $123, %zmm24, %zmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0x93,0x47,0x47,0xc2,0xe8,0x7b]
          vcmpbf16 $123, %zmm24, %zmm23, %k5 {%k7}

// CHECK: vcmpbf16  $123, 268435456(%rbp,%r14,8), %zmm23, %k5
// CHECK: encoding: [0x62,0xb3,0x47,0x40,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vcmpbf16  $123, 268435456(%rbp,%r14,8), %zmm23, %k5

// CHECK: vcmpbf16  $123, 291(%r8,%rax,4), %zmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xd3,0x47,0x47,0xc2,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vcmpbf16  $123, 291(%r8,%rax,4), %zmm23, %k5 {%k7}

// CHECK: vcmpbf16  $123, (%rip){1to32}, %zmm23, %k5
// CHECK: encoding: [0x62,0xf3,0x47,0x50,0xc2,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vcmpbf16  $123, (%rip){1to32}, %zmm23, %k5

// CHECK: vcmpbf16  $123, -2048(,%rbp,2), %zmm23, %k5
// CHECK: encoding: [0x62,0xf3,0x47,0x40,0xc2,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vcmpbf16  $123, -2048(,%rbp,2), %zmm23, %k5

// CHECK: vcmpbf16  $123, 8128(%rcx), %zmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x47,0x47,0xc2,0x69,0x7f,0x7b]
          vcmpbf16  $123, 8128(%rcx), %zmm23, %k5 {%k7}

// CHECK: vcmpbf16  $123, -256(%rdx){1to32}, %zmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x47,0x57,0xc2,0x6a,0x80,0x7b]
          vcmpbf16  $123, -256(%rdx){1to32}, %zmm23, %k5 {%k7}

// CHECK: vcmpbf16  $123, 268435456(%rbp,%r14,8), %xmm23, %k5
// CHECK: encoding: [0x62,0xb3,0x47,0x00,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vcmpbf16  $123, 268435456(%rbp,%r14,8), %xmm23, %k5

// CHECK: vcmpbf16  $123, 291(%r8,%rax,4), %xmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xd3,0x47,0x07,0xc2,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vcmpbf16  $123, 291(%r8,%rax,4), %xmm23, %k5 {%k7}

// CHECK: vcmpbf16  $123, (%rip){1to8}, %xmm23, %k5
// CHECK: encoding: [0x62,0xf3,0x47,0x10,0xc2,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vcmpbf16  $123, (%rip){1to8}, %xmm23, %k5

// CHECK: vcmpbf16  $123, -512(,%rbp,2), %xmm23, %k5
// CHECK: encoding: [0x62,0xf3,0x47,0x00,0xc2,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vcmpbf16  $123, -512(,%rbp,2), %xmm23, %k5

// CHECK: vcmpbf16  $123, 2032(%rcx), %xmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x47,0x07,0xc2,0x69,0x7f,0x7b]
          vcmpbf16  $123, 2032(%rcx), %xmm23, %k5 {%k7}

// CHECK: vcmpbf16  $123, -256(%rdx){1to8}, %xmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x47,0x17,0xc2,0x6a,0x80,0x7b]
          vcmpbf16  $123, -256(%rdx){1to8}, %xmm23, %k5 {%k7}

// CHECK: vcmpbf16  $123, 268435456(%rbp,%r14,8), %ymm23, %k5
// CHECK: encoding: [0x62,0xb3,0x47,0x20,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vcmpbf16  $123, 268435456(%rbp,%r14,8), %ymm23, %k5

// CHECK: vcmpbf16  $123, 291(%r8,%rax,4), %ymm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xd3,0x47,0x27,0xc2,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vcmpbf16  $123, 291(%r8,%rax,4), %ymm23, %k5 {%k7}

// CHECK: vcmpbf16  $123, (%rip){1to16}, %ymm23, %k5
// CHECK: encoding: [0x62,0xf3,0x47,0x30,0xc2,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vcmpbf16  $123, (%rip){1to16}, %ymm23, %k5

// CHECK: vcmpbf16  $123, -1024(,%rbp,2), %ymm23, %k5
// CHECK: encoding: [0x62,0xf3,0x47,0x20,0xc2,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vcmpbf16  $123, -1024(,%rbp,2), %ymm23, %k5

// CHECK: vcmpbf16  $123, 4064(%rcx), %ymm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x47,0x27,0xc2,0x69,0x7f,0x7b]
          vcmpbf16  $123, 4064(%rcx), %ymm23, %k5 {%k7}

// CHECK: vcmpbf16  $123, -256(%rdx){1to16}, %ymm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x47,0x37,0xc2,0x6a,0x80,0x7b]
          vcmpbf16  $123, -256(%rdx){1to16}, %ymm23, %k5 {%k7}

// CHECK: vcomisbf16 %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x2f,0xf7]
          vcomisbf16 %xmm23, %xmm22

// CHECK: vcomisbf16  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x2f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcomisbf16  268435456(%rbp,%r14,8), %xmm22

// CHECK: vcomisbf16  291(%r8,%rax,4), %xmm22
// CHECK: encoding: [0x62,0xc5,0x7d,0x08,0x2f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcomisbf16  291(%r8,%rax,4), %xmm22

// CHECK: vcomisbf16  (%rip), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x35,0x00,0x00,0x00,0x00]
          vcomisbf16  (%rip), %xmm22

// CHECK: vcomisbf16  -64(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x34,0x6d,0xc0,0xff,0xff,0xff]
          vcomisbf16  -64(,%rbp,2), %xmm22

// CHECK: vcomisbf16  254(%rcx), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x71,0x7f]
          vcomisbf16  254(%rcx), %xmm22

// CHECK: vcomisbf16  -256(%rdx), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x72,0x80]
          vcomisbf16  -256(%rdx), %xmm22

// CHECK: vdivbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5e,0xf0]
          vdivbf16 %ymm24, %ymm23, %ymm22

// CHECK: vdivbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5e,0xf0]
          vdivbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vdivbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5e,0xf0]
          vdivbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vdivbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5e,0xf0]
          vdivbf16 %zmm24, %zmm23, %zmm22

// CHECK: vdivbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5e,0xf0]
          vdivbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vdivbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5e,0xf0]
          vdivbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vdivbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5e,0xf0]
          vdivbf16 %xmm24, %xmm23, %xmm22

// CHECK: vdivbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5e,0xf0]
          vdivbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vdivbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5e,0xf0]
          vdivbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vdivbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vdivbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdivbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vdivbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5e,0x35,0x00,0x00,0x00,0x00]
          vdivbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vdivbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5e,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vdivbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vdivbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5e,0x71,0x7f]
          vdivbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vdivbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5e,0x72,0x80]
          vdivbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vdivbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vdivbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdivbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vdivbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5e,0x35,0x00,0x00,0x00,0x00]
          vdivbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vdivbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vdivbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vdivbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5e,0x71,0x7f]
          vdivbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vdivbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5e,0x72,0x80]
          vdivbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vdivbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vdivbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdivbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vdivbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5e,0x35,0x00,0x00,0x00,0x00]
          vdivbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vdivbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vdivbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vdivbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5e,0x71,0x7f]
          vdivbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vdivbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5e,0x72,0x80]
          vdivbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd132bf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x98,0xf0]
          vfmadd132bf16 %ymm24, %ymm23, %ymm22

// CHECK: vfmadd132bf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x98,0xf0]
          vfmadd132bf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd132bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x98,0xf0]
          vfmadd132bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd132bf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x98,0xf0]
          vfmadd132bf16 %zmm24, %zmm23, %zmm22

// CHECK: vfmadd132bf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x98,0xf0]
          vfmadd132bf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfmadd132bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x98,0xf0]
          vfmadd132bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd132bf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x98,0xf0]
          vfmadd132bf16 %xmm24, %xmm23, %xmm22

// CHECK: vfmadd132bf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x98,0xf0]
          vfmadd132bf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfmadd132bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x98,0xf0]
          vfmadd132bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd132bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfmadd132bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x98,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd132bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfmadd132bf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x98,0x35,0x00,0x00,0x00,0x00]
          vfmadd132bf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfmadd132bf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x98,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd132bf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfmadd132bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x98,0x71,0x7f]
          vfmadd132bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd132bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x98,0x72,0x80]
          vfmadd132bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd132bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfmadd132bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x98,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd132bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfmadd132bf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x98,0x35,0x00,0x00,0x00,0x00]
          vfmadd132bf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfmadd132bf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x98,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd132bf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfmadd132bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x98,0x71,0x7f]
          vfmadd132bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd132bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x98,0x72,0x80]
          vfmadd132bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd132bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfmadd132bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x98,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd132bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfmadd132bf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x98,0x35,0x00,0x00,0x00,0x00]
          vfmadd132bf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfmadd132bf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x98,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd132bf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfmadd132bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x98,0x71,0x7f]
          vfmadd132bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd132bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x98,0x72,0x80]
          vfmadd132bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd213bf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xa8,0xf0]
          vfmadd213bf16 %ymm24, %ymm23, %ymm22

// CHECK: vfmadd213bf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xa8,0xf0]
          vfmadd213bf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd213bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xa8,0xf0]
          vfmadd213bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd213bf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xa8,0xf0]
          vfmadd213bf16 %zmm24, %zmm23, %zmm22

// CHECK: vfmadd213bf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xa8,0xf0]
          vfmadd213bf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfmadd213bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xa8,0xf0]
          vfmadd213bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd213bf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xa8,0xf0]
          vfmadd213bf16 %xmm24, %xmm23, %xmm22

// CHECK: vfmadd213bf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xa8,0xf0]
          vfmadd213bf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfmadd213bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xa8,0xf0]
          vfmadd213bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd213bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfmadd213bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xa8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd213bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfmadd213bf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xa8,0x35,0x00,0x00,0x00,0x00]
          vfmadd213bf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfmadd213bf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xa8,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd213bf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfmadd213bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xa8,0x71,0x7f]
          vfmadd213bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd213bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xa8,0x72,0x80]
          vfmadd213bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd213bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfmadd213bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xa8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd213bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfmadd213bf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xa8,0x35,0x00,0x00,0x00,0x00]
          vfmadd213bf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfmadd213bf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xa8,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd213bf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfmadd213bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xa8,0x71,0x7f]
          vfmadd213bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd213bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xa8,0x72,0x80]
          vfmadd213bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd213bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfmadd213bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xa8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd213bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfmadd213bf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xa8,0x35,0x00,0x00,0x00,0x00]
          vfmadd213bf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfmadd213bf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xa8,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd213bf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfmadd213bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xa8,0x71,0x7f]
          vfmadd213bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd213bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xa8,0x72,0x80]
          vfmadd213bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd231bf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xb8,0xf0]
          vfmadd231bf16 %ymm24, %ymm23, %ymm22

// CHECK: vfmadd231bf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xb8,0xf0]
          vfmadd231bf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd231bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xb8,0xf0]
          vfmadd231bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd231bf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xb8,0xf0]
          vfmadd231bf16 %zmm24, %zmm23, %zmm22

// CHECK: vfmadd231bf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xb8,0xf0]
          vfmadd231bf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfmadd231bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xb8,0xf0]
          vfmadd231bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd231bf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xb8,0xf0]
          vfmadd231bf16 %xmm24, %xmm23, %xmm22

// CHECK: vfmadd231bf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xb8,0xf0]
          vfmadd231bf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfmadd231bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xb8,0xf0]
          vfmadd231bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd231bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfmadd231bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xb8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd231bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfmadd231bf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xb8,0x35,0x00,0x00,0x00,0x00]
          vfmadd231bf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfmadd231bf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xb8,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd231bf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfmadd231bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xb8,0x71,0x7f]
          vfmadd231bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd231bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xb8,0x72,0x80]
          vfmadd231bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmadd231bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfmadd231bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xb8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd231bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfmadd231bf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xb8,0x35,0x00,0x00,0x00,0x00]
          vfmadd231bf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfmadd231bf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xb8,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd231bf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfmadd231bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xb8,0x71,0x7f]
          vfmadd231bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd231bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xb8,0x72,0x80]
          vfmadd231bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd231bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfmadd231bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xb8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd231bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfmadd231bf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xb8,0x35,0x00,0x00,0x00,0x00]
          vfmadd231bf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfmadd231bf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xb8,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd231bf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfmadd231bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xb8,0x71,0x7f]
          vfmadd231bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmadd231bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xb8,0x72,0x80]
          vfmadd231bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub132bf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x9a,0xf0]
          vfmsub132bf16 %ymm24, %ymm23, %ymm22

// CHECK: vfmsub132bf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x9a,0xf0]
          vfmsub132bf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub132bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x9a,0xf0]
          vfmsub132bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub132bf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x9a,0xf0]
          vfmsub132bf16 %zmm24, %zmm23, %zmm22

// CHECK: vfmsub132bf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x9a,0xf0]
          vfmsub132bf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfmsub132bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x9a,0xf0]
          vfmsub132bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub132bf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x9a,0xf0]
          vfmsub132bf16 %xmm24, %xmm23, %xmm22

// CHECK: vfmsub132bf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x9a,0xf0]
          vfmsub132bf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfmsub132bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x9a,0xf0]
          vfmsub132bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub132bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfmsub132bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x9a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub132bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfmsub132bf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x9a,0x35,0x00,0x00,0x00,0x00]
          vfmsub132bf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfmsub132bf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x9a,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub132bf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfmsub132bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x9a,0x71,0x7f]
          vfmsub132bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub132bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x9a,0x72,0x80]
          vfmsub132bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub132bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfmsub132bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x9a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub132bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfmsub132bf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x9a,0x35,0x00,0x00,0x00,0x00]
          vfmsub132bf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfmsub132bf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x9a,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub132bf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfmsub132bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x9a,0x71,0x7f]
          vfmsub132bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub132bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x9a,0x72,0x80]
          vfmsub132bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub132bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfmsub132bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x9a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub132bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfmsub132bf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x9a,0x35,0x00,0x00,0x00,0x00]
          vfmsub132bf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfmsub132bf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x9a,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub132bf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfmsub132bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x9a,0x71,0x7f]
          vfmsub132bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub132bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x9a,0x72,0x80]
          vfmsub132bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub213bf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xaa,0xf0]
          vfmsub213bf16 %ymm24, %ymm23, %ymm22

// CHECK: vfmsub213bf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xaa,0xf0]
          vfmsub213bf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub213bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xaa,0xf0]
          vfmsub213bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub213bf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xaa,0xf0]
          vfmsub213bf16 %zmm24, %zmm23, %zmm22

// CHECK: vfmsub213bf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xaa,0xf0]
          vfmsub213bf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfmsub213bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xaa,0xf0]
          vfmsub213bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub213bf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xaa,0xf0]
          vfmsub213bf16 %xmm24, %xmm23, %xmm22

// CHECK: vfmsub213bf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xaa,0xf0]
          vfmsub213bf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfmsub213bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xaa,0xf0]
          vfmsub213bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub213bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfmsub213bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xaa,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub213bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfmsub213bf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xaa,0x35,0x00,0x00,0x00,0x00]
          vfmsub213bf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfmsub213bf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xaa,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub213bf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfmsub213bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xaa,0x71,0x7f]
          vfmsub213bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub213bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xaa,0x72,0x80]
          vfmsub213bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub213bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfmsub213bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xaa,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub213bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfmsub213bf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xaa,0x35,0x00,0x00,0x00,0x00]
          vfmsub213bf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfmsub213bf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xaa,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub213bf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfmsub213bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xaa,0x71,0x7f]
          vfmsub213bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub213bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xaa,0x72,0x80]
          vfmsub213bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub213bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfmsub213bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xaa,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub213bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfmsub213bf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xaa,0x35,0x00,0x00,0x00,0x00]
          vfmsub213bf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfmsub213bf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xaa,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub213bf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfmsub213bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xaa,0x71,0x7f]
          vfmsub213bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub213bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xaa,0x72,0x80]
          vfmsub213bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub231bf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xba,0xf0]
          vfmsub231bf16 %ymm24, %ymm23, %ymm22

// CHECK: vfmsub231bf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xba,0xf0]
          vfmsub231bf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub231bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xba,0xf0]
          vfmsub231bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub231bf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xba,0xf0]
          vfmsub231bf16 %zmm24, %zmm23, %zmm22

// CHECK: vfmsub231bf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xba,0xf0]
          vfmsub231bf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfmsub231bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xba,0xf0]
          vfmsub231bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub231bf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xba,0xf0]
          vfmsub231bf16 %xmm24, %xmm23, %xmm22

// CHECK: vfmsub231bf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xba,0xf0]
          vfmsub231bf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfmsub231bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xba,0xf0]
          vfmsub231bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub231bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfmsub231bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xba,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub231bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfmsub231bf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xba,0x35,0x00,0x00,0x00,0x00]
          vfmsub231bf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfmsub231bf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xba,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub231bf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfmsub231bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xba,0x71,0x7f]
          vfmsub231bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub231bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xba,0x72,0x80]
          vfmsub231bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfmsub231bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfmsub231bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xba,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub231bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfmsub231bf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xba,0x35,0x00,0x00,0x00,0x00]
          vfmsub231bf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfmsub231bf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xba,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub231bf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfmsub231bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xba,0x71,0x7f]
          vfmsub231bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub231bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xba,0x72,0x80]
          vfmsub231bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub231bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfmsub231bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xba,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub231bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfmsub231bf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xba,0x35,0x00,0x00,0x00,0x00]
          vfmsub231bf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfmsub231bf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xba,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub231bf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfmsub231bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xba,0x71,0x7f]
          vfmsub231bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfmsub231bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xba,0x72,0x80]
          vfmsub231bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd132bf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x9c,0xf0]
          vfnmadd132bf16 %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd132bf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x9c,0xf0]
          vfnmadd132bf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd132bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x9c,0xf0]
          vfnmadd132bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd132bf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x9c,0xf0]
          vfnmadd132bf16 %zmm24, %zmm23, %zmm22

// CHECK: vfnmadd132bf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x9c,0xf0]
          vfnmadd132bf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfnmadd132bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x9c,0xf0]
          vfnmadd132bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd132bf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x9c,0xf0]
          vfnmadd132bf16 %xmm24, %xmm23, %xmm22

// CHECK: vfnmadd132bf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x9c,0xf0]
          vfnmadd132bf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfnmadd132bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x9c,0xf0]
          vfnmadd132bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd132bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfnmadd132bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x9c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd132bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfnmadd132bf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x9c,0x35,0x00,0x00,0x00,0x00]
          vfnmadd132bf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfnmadd132bf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x9c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd132bf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfnmadd132bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x9c,0x71,0x7f]
          vfnmadd132bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd132bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x9c,0x72,0x80]
          vfnmadd132bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd132bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfnmadd132bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x9c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd132bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd132bf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x9c,0x35,0x00,0x00,0x00,0x00]
          vfnmadd132bf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfnmadd132bf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x9c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd132bf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfnmadd132bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x9c,0x71,0x7f]
          vfnmadd132bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd132bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x9c,0x72,0x80]
          vfnmadd132bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd132bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfnmadd132bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x9c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd132bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfnmadd132bf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x9c,0x35,0x00,0x00,0x00,0x00]
          vfnmadd132bf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfnmadd132bf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x9c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd132bf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfnmadd132bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x9c,0x71,0x7f]
          vfnmadd132bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd132bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x9c,0x72,0x80]
          vfnmadd132bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd213bf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xac,0xf0]
          vfnmadd213bf16 %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd213bf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xac,0xf0]
          vfnmadd213bf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd213bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xac,0xf0]
          vfnmadd213bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd213bf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xac,0xf0]
          vfnmadd213bf16 %zmm24, %zmm23, %zmm22

// CHECK: vfnmadd213bf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xac,0xf0]
          vfnmadd213bf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfnmadd213bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xac,0xf0]
          vfnmadd213bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd213bf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xac,0xf0]
          vfnmadd213bf16 %xmm24, %xmm23, %xmm22

// CHECK: vfnmadd213bf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xac,0xf0]
          vfnmadd213bf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfnmadd213bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xac,0xf0]
          vfnmadd213bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd213bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfnmadd213bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xac,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd213bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfnmadd213bf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xac,0x35,0x00,0x00,0x00,0x00]
          vfnmadd213bf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfnmadd213bf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xac,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd213bf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfnmadd213bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xac,0x71,0x7f]
          vfnmadd213bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd213bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xac,0x72,0x80]
          vfnmadd213bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd213bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfnmadd213bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xac,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd213bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd213bf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xac,0x35,0x00,0x00,0x00,0x00]
          vfnmadd213bf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfnmadd213bf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xac,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd213bf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfnmadd213bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xac,0x71,0x7f]
          vfnmadd213bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd213bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xac,0x72,0x80]
          vfnmadd213bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd213bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfnmadd213bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xac,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd213bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfnmadd213bf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xac,0x35,0x00,0x00,0x00,0x00]
          vfnmadd213bf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfnmadd213bf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xac,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd213bf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfnmadd213bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xac,0x71,0x7f]
          vfnmadd213bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd213bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xac,0x72,0x80]
          vfnmadd213bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd231bf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xbc,0xf0]
          vfnmadd231bf16 %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd231bf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xbc,0xf0]
          vfnmadd231bf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd231bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xbc,0xf0]
          vfnmadd231bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd231bf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xbc,0xf0]
          vfnmadd231bf16 %zmm24, %zmm23, %zmm22

// CHECK: vfnmadd231bf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xbc,0xf0]
          vfnmadd231bf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfnmadd231bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xbc,0xf0]
          vfnmadd231bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd231bf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xbc,0xf0]
          vfnmadd231bf16 %xmm24, %xmm23, %xmm22

// CHECK: vfnmadd231bf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xbc,0xf0]
          vfnmadd231bf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfnmadd231bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xbc,0xf0]
          vfnmadd231bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd231bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfnmadd231bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xbc,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd231bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfnmadd231bf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xbc,0x35,0x00,0x00,0x00,0x00]
          vfnmadd231bf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfnmadd231bf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xbc,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd231bf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfnmadd231bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xbc,0x71,0x7f]
          vfnmadd231bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd231bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xbc,0x72,0x80]
          vfnmadd231bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmadd231bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfnmadd231bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xbc,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd231bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd231bf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xbc,0x35,0x00,0x00,0x00,0x00]
          vfnmadd231bf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfnmadd231bf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xbc,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd231bf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfnmadd231bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xbc,0x71,0x7f]
          vfnmadd231bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd231bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xbc,0x72,0x80]
          vfnmadd231bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd231bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfnmadd231bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xbc,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd231bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfnmadd231bf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xbc,0x35,0x00,0x00,0x00,0x00]
          vfnmadd231bf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfnmadd231bf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xbc,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd231bf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfnmadd231bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xbc,0x71,0x7f]
          vfnmadd231bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmadd231bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xbc,0x72,0x80]
          vfnmadd231bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub132bf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x9e,0xf0]
          vfnmsub132bf16 %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub132bf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x9e,0xf0]
          vfnmsub132bf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub132bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x9e,0xf0]
          vfnmsub132bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub132bf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x9e,0xf0]
          vfnmsub132bf16 %zmm24, %zmm23, %zmm22

// CHECK: vfnmsub132bf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x9e,0xf0]
          vfnmsub132bf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfnmsub132bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x9e,0xf0]
          vfnmsub132bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub132bf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x9e,0xf0]
          vfnmsub132bf16 %xmm24, %xmm23, %xmm22

// CHECK: vfnmsub132bf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x9e,0xf0]
          vfnmsub132bf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfnmsub132bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x9e,0xf0]
          vfnmsub132bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub132bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfnmsub132bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x9e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub132bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfnmsub132bf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x9e,0x35,0x00,0x00,0x00,0x00]
          vfnmsub132bf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfnmsub132bf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x9e,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub132bf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfnmsub132bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x9e,0x71,0x7f]
          vfnmsub132bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub132bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x9e,0x72,0x80]
          vfnmsub132bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub132bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfnmsub132bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x9e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub132bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub132bf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x9e,0x35,0x00,0x00,0x00,0x00]
          vfnmsub132bf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfnmsub132bf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x9e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub132bf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfnmsub132bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x9e,0x71,0x7f]
          vfnmsub132bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub132bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x9e,0x72,0x80]
          vfnmsub132bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub132bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfnmsub132bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x9e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub132bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfnmsub132bf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x9e,0x35,0x00,0x00,0x00,0x00]
          vfnmsub132bf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfnmsub132bf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x9e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub132bf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfnmsub132bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x9e,0x71,0x7f]
          vfnmsub132bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub132bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x9e,0x72,0x80]
          vfnmsub132bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub213bf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xae,0xf0]
          vfnmsub213bf16 %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub213bf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xae,0xf0]
          vfnmsub213bf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub213bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xae,0xf0]
          vfnmsub213bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub213bf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xae,0xf0]
          vfnmsub213bf16 %zmm24, %zmm23, %zmm22

// CHECK: vfnmsub213bf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xae,0xf0]
          vfnmsub213bf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfnmsub213bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xae,0xf0]
          vfnmsub213bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub213bf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xae,0xf0]
          vfnmsub213bf16 %xmm24, %xmm23, %xmm22

// CHECK: vfnmsub213bf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xae,0xf0]
          vfnmsub213bf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfnmsub213bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xae,0xf0]
          vfnmsub213bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub213bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfnmsub213bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xae,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub213bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfnmsub213bf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xae,0x35,0x00,0x00,0x00,0x00]
          vfnmsub213bf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfnmsub213bf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xae,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub213bf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfnmsub213bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xae,0x71,0x7f]
          vfnmsub213bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub213bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xae,0x72,0x80]
          vfnmsub213bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub213bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfnmsub213bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xae,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub213bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub213bf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xae,0x35,0x00,0x00,0x00,0x00]
          vfnmsub213bf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfnmsub213bf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xae,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub213bf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfnmsub213bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xae,0x71,0x7f]
          vfnmsub213bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub213bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xae,0x72,0x80]
          vfnmsub213bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub213bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfnmsub213bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xae,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub213bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfnmsub213bf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xae,0x35,0x00,0x00,0x00,0x00]
          vfnmsub213bf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfnmsub213bf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xae,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub213bf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfnmsub213bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xae,0x71,0x7f]
          vfnmsub213bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub213bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xae,0x72,0x80]
          vfnmsub213bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub231bf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xbe,0xf0]
          vfnmsub231bf16 %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub231bf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xbe,0xf0]
          vfnmsub231bf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub231bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xbe,0xf0]
          vfnmsub231bf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub231bf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xbe,0xf0]
          vfnmsub231bf16 %zmm24, %zmm23, %zmm22

// CHECK: vfnmsub231bf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xbe,0xf0]
          vfnmsub231bf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vfnmsub231bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xbe,0xf0]
          vfnmsub231bf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub231bf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xbe,0xf0]
          vfnmsub231bf16 %xmm24, %xmm23, %xmm22

// CHECK: vfnmsub231bf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xbe,0xf0]
          vfnmsub231bf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vfnmsub231bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xbe,0xf0]
          vfnmsub231bf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub231bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231bf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vfnmsub231bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xbe,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub231bf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vfnmsub231bf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xbe,0x35,0x00,0x00,0x00,0x00]
          vfnmsub231bf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vfnmsub231bf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xbe,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub231bf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vfnmsub231bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xbe,0x71,0x7f]
          vfnmsub231bf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub231bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xbe,0x72,0x80]
          vfnmsub231bf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vfnmsub231bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231bf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vfnmsub231bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xbe,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub231bf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub231bf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xbe,0x35,0x00,0x00,0x00,0x00]
          vfnmsub231bf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vfnmsub231bf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xbe,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub231bf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vfnmsub231bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xbe,0x71,0x7f]
          vfnmsub231bf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub231bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xbe,0x72,0x80]
          vfnmsub231bf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub231bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231bf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vfnmsub231bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xbe,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub231bf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vfnmsub231bf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xbe,0x35,0x00,0x00,0x00,0x00]
          vfnmsub231bf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vfnmsub231bf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xbe,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub231bf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vfnmsub231bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xbe,0x71,0x7f]
          vfnmsub231bf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vfnmsub231bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xbe,0x72,0x80]
          vfnmsub231bf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vfpclassbf16 $123, %zmm23, %k5
// CHECK: encoding: [0x62,0xb3,0x7f,0x48,0x66,0xef,0x7b]
          vfpclassbf16 $123, %zmm23, %k5

// CHECK: vfpclassbf16 $123, %zmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xb3,0x7f,0x4f,0x66,0xef,0x7b]
          vfpclassbf16 $123, %zmm23, %k5 {%k7}

// CHECK: vfpclassbf16 $123, %ymm23, %k5
// CHECK: encoding: [0x62,0xb3,0x7f,0x28,0x66,0xef,0x7b]
          vfpclassbf16 $123, %ymm23, %k5

// CHECK: vfpclassbf16 $123, %ymm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xb3,0x7f,0x2f,0x66,0xef,0x7b]
          vfpclassbf16 $123, %ymm23, %k5 {%k7}

// CHECK: vfpclassbf16 $123, %xmm23, %k5
// CHECK: encoding: [0x62,0xb3,0x7f,0x08,0x66,0xef,0x7b]
          vfpclassbf16 $123, %xmm23, %k5

// CHECK: vfpclassbf16 $123, %xmm23, %k5 {%k7}
// CHECK: encoding: [0x62,0xb3,0x7f,0x0f,0x66,0xef,0x7b]
          vfpclassbf16 $123, %xmm23, %k5 {%k7}

// CHECK: vfpclassbf16x  $123, 268435456(%rbp,%r14,8), %k5
// CHECK: encoding: [0x62,0xb3,0x7f,0x08,0x66,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vfpclassbf16x  $123, 268435456(%rbp,%r14,8), %k5

// CHECK: vfpclassbf16x  $123, 291(%r8,%rax,4), %k5 {%k7}
// CHECK: encoding: [0x62,0xd3,0x7f,0x0f,0x66,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vfpclassbf16x  $123, 291(%r8,%rax,4), %k5 {%k7}

// CHECK: vfpclassbf16  $123, (%rip){1to8}, %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x66,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vfpclassbf16  $123, (%rip){1to8}, %k5

// CHECK: vfpclassbf16x  $123, -512(,%rbp,2), %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x66,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vfpclassbf16x  $123, -512(,%rbp,2), %k5

// CHECK: vfpclassbf16x  $123, 2032(%rcx), %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x66,0x69,0x7f,0x7b]
          vfpclassbf16x  $123, 2032(%rcx), %k5 {%k7}

// CHECK: vfpclassbf16  $123, -256(%rdx){1to8}, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x1f,0x66,0x6a,0x80,0x7b]
          vfpclassbf16  $123, -256(%rdx){1to8}, %k5 {%k7}

// CHECK: vfpclassbf16  $123, (%rip){1to16}, %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x66,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vfpclassbf16  $123, (%rip){1to16}, %k5

// CHECK: vfpclassbf16y  $123, -1024(,%rbp,2), %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x66,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vfpclassbf16y  $123, -1024(,%rbp,2), %k5

// CHECK: vfpclassbf16y  $123, 4064(%rcx), %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x66,0x69,0x7f,0x7b]
          vfpclassbf16y  $123, 4064(%rcx), %k5 {%k7}

// CHECK: vfpclassbf16  $123, -256(%rdx){1to16}, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x3f,0x66,0x6a,0x80,0x7b]
          vfpclassbf16  $123, -256(%rdx){1to16}, %k5 {%k7}

// CHECK: vfpclassbf16  $123, (%rip){1to32}, %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x66,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vfpclassbf16  $123, (%rip){1to32}, %k5

// CHECK: vfpclassbf16z  $123, -2048(,%rbp,2), %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x66,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vfpclassbf16z  $123, -2048(,%rbp,2), %k5

// CHECK: vfpclassbf16z  $123, 8128(%rcx), %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x66,0x69,0x7f,0x7b]
          vfpclassbf16z  $123, 8128(%rcx), %k5 {%k7}

// CHECK: vfpclassbf16  $123, -256(%rdx){1to32}, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x5f,0x66,0x6a,0x80,0x7b]
          vfpclassbf16  $123, -256(%rdx){1to32}, %k5 {%k7}

// CHECK: vgetexpbf16 %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x42,0xf7]
          vgetexpbf16 %xmm23, %xmm22

// CHECK: vgetexpbf16 %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x42,0xf7]
          vgetexpbf16 %xmm23, %xmm22 {%k7}

// CHECK: vgetexpbf16 %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x42,0xf7]
          vgetexpbf16 %xmm23, %xmm22 {%k7} {z}

// CHECK: vgetexpbf16 %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x42,0xf7]
          vgetexpbf16 %zmm23, %zmm22

// CHECK: vgetexpbf16 %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x42,0xf7]
          vgetexpbf16 %zmm23, %zmm22 {%k7}

// CHECK: vgetexpbf16 %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7d,0xcf,0x42,0xf7]
          vgetexpbf16 %zmm23, %zmm22 {%k7} {z}

// CHECK: vgetexpbf16 %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x42,0xf7]
          vgetexpbf16 %ymm23, %ymm22

// CHECK: vgetexpbf16 %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x42,0xf7]
          vgetexpbf16 %ymm23, %ymm22 {%k7}

// CHECK: vgetexpbf16 %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7d,0xaf,0x42,0xf7]
          vgetexpbf16 %ymm23, %ymm22 {%k7} {z}

// CHECK: vgetexpbf16  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexpbf16  268435456(%rbp,%r14,8), %xmm22

// CHECK: vgetexpbf16  291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x42,0xb4,0x80,0x23,0x01,0x00,0x00]
          vgetexpbf16  291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vgetexpbf16  (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x42,0x35,0x00,0x00,0x00,0x00]
          vgetexpbf16  (%rip){1to8}, %xmm22

// CHECK: vgetexpbf16  -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x42,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vgetexpbf16  -512(,%rbp,2), %xmm22

// CHECK: vgetexpbf16  2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x42,0x71,0x7f]
          vgetexpbf16  2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vgetexpbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x42,0x72,0x80]
          vgetexpbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vgetexpbf16  268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexpbf16  268435456(%rbp,%r14,8), %ymm22

// CHECK: vgetexpbf16  291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x42,0xb4,0x80,0x23,0x01,0x00,0x00]
          vgetexpbf16  291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vgetexpbf16  (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x42,0x35,0x00,0x00,0x00,0x00]
          vgetexpbf16  (%rip){1to16}, %ymm22

// CHECK: vgetexpbf16  -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x42,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vgetexpbf16  -1024(,%rbp,2), %ymm22

// CHECK: vgetexpbf16  4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x42,0x71,0x7f]
          vgetexpbf16  4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vgetexpbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x42,0x72,0x80]
          vgetexpbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vgetexpbf16  268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexpbf16  268435456(%rbp,%r14,8), %zmm22

// CHECK: vgetexpbf16  291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x42,0xb4,0x80,0x23,0x01,0x00,0x00]
          vgetexpbf16  291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vgetexpbf16  (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x42,0x35,0x00,0x00,0x00,0x00]
          vgetexpbf16  (%rip){1to32}, %zmm22

// CHECK: vgetexpbf16  -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x42,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vgetexpbf16  -2048(,%rbp,2), %zmm22

// CHECK: vgetexpbf16  8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x42,0x71,0x7f]
          vgetexpbf16  8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vgetexpbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x42,0x72,0x80]
          vgetexpbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vgetmantbf16 $123, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x26,0xf7,0x7b]
          vgetmantbf16 $123, %zmm23, %zmm22

// CHECK: vgetmantbf16 $123, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x4f,0x26,0xf7,0x7b]
          vgetmantbf16 $123, %zmm23, %zmm22 {%k7}

// CHECK: vgetmantbf16 $123, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0xcf,0x26,0xf7,0x7b]
          vgetmantbf16 $123, %zmm23, %zmm22 {%k7} {z}

// CHECK: vgetmantbf16 $123, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x26,0xf7,0x7b]
          vgetmantbf16 $123, %ymm23, %ymm22

// CHECK: vgetmantbf16 $123, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x2f,0x26,0xf7,0x7b]
          vgetmantbf16 $123, %ymm23, %ymm22 {%k7}

// CHECK: vgetmantbf16 $123, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0xaf,0x26,0xf7,0x7b]
          vgetmantbf16 $123, %ymm23, %ymm22 {%k7} {z}

// CHECK: vgetmantbf16 $123, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x26,0xf7,0x7b]
          vgetmantbf16 $123, %xmm23, %xmm22

// CHECK: vgetmantbf16 $123, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x0f,0x26,0xf7,0x7b]
          vgetmantbf16 $123, %xmm23, %xmm22 {%k7}

// CHECK: vgetmantbf16 $123, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0x8f,0x26,0xf7,0x7b]
          vgetmantbf16 $123, %xmm23, %xmm22 {%k7} {z}

// CHECK: vgetmantbf16  $123, 268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantbf16  $123, 268435456(%rbp,%r14,8), %xmm22

// CHECK: vgetmantbf16  $123, 291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x0f,0x26,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vgetmantbf16  $123, 291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vgetmantbf16  $123, (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x18,0x26,0x35,0x00,0x00,0x00,0x00,0x7b]
          vgetmantbf16  $123, (%rip){1to8}, %xmm22

// CHECK: vgetmantbf16  $123, -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x08,0x26,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vgetmantbf16  $123, -512(,%rbp,2), %xmm22

// CHECK: vgetmantbf16  $123, 2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0x8f,0x26,0x71,0x7f,0x7b]
          vgetmantbf16  $123, 2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vgetmantbf16  $123, -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0x9f,0x26,0x72,0x80,0x7b]
          vgetmantbf16  $123, -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vgetmantbf16  $123, 268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantbf16  $123, 268435456(%rbp,%r14,8), %ymm22

// CHECK: vgetmantbf16  $123, 291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x2f,0x26,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vgetmantbf16  $123, 291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vgetmantbf16  $123, (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x38,0x26,0x35,0x00,0x00,0x00,0x00,0x7b]
          vgetmantbf16  $123, (%rip){1to16}, %ymm22

// CHECK: vgetmantbf16  $123, -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x28,0x26,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vgetmantbf16  $123, -1024(,%rbp,2), %ymm22

// CHECK: vgetmantbf16  $123, 4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xaf,0x26,0x71,0x7f,0x7b]
          vgetmantbf16  $123, 4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vgetmantbf16  $123, -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xbf,0x26,0x72,0x80,0x7b]
          vgetmantbf16  $123, -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vgetmantbf16  $123, 268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantbf16  $123, 268435456(%rbp,%r14,8), %zmm22

// CHECK: vgetmantbf16  $123, 291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x4f,0x26,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vgetmantbf16  $123, 291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vgetmantbf16  $123, (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x58,0x26,0x35,0x00,0x00,0x00,0x00,0x7b]
          vgetmantbf16  $123, (%rip){1to32}, %zmm22

// CHECK: vgetmantbf16  $123, -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x26,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vgetmantbf16  $123, -2048(,%rbp,2), %zmm22

// CHECK: vgetmantbf16  $123, 8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xcf,0x26,0x71,0x7f,0x7b]
          vgetmantbf16  $123, 8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vgetmantbf16  $123, -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xdf,0x26,0x72,0x80,0x7b]
          vgetmantbf16  $123, -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vmaxbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5f,0xf0]
          vmaxbf16 %ymm24, %ymm23, %ymm22

// CHECK: vmaxbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5f,0xf0]
          vmaxbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vmaxbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5f,0xf0]
          vmaxbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmaxbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5f,0xf0]
          vmaxbf16 %zmm24, %zmm23, %zmm22

// CHECK: vmaxbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5f,0xf0]
          vmaxbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vmaxbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5f,0xf0]
          vmaxbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vmaxbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5f,0xf0]
          vmaxbf16 %xmm24, %xmm23, %xmm22

// CHECK: vmaxbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5f,0xf0]
          vmaxbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vmaxbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5f,0xf0]
          vmaxbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vmaxbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vmaxbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmaxbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vmaxbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5f,0x35,0x00,0x00,0x00,0x00]
          vmaxbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vmaxbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5f,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vmaxbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vmaxbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5f,0x71,0x7f]
          vmaxbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vmaxbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5f,0x72,0x80]
          vmaxbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vmaxbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vmaxbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmaxbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vmaxbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5f,0x35,0x00,0x00,0x00,0x00]
          vmaxbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vmaxbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5f,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vmaxbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vmaxbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5f,0x71,0x7f]
          vmaxbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vmaxbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5f,0x72,0x80]
          vmaxbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmaxbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vmaxbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmaxbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vmaxbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5f,0x35,0x00,0x00,0x00,0x00]
          vmaxbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vmaxbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5f,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vmaxbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vmaxbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5f,0x71,0x7f]
          vmaxbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vmaxbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5f,0x72,0x80]
          vmaxbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5d,0xf0]
          vminbf16 %ymm24, %ymm23, %ymm22

// CHECK: vminbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5d,0xf0]
          vminbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vminbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5d,0xf0]
          vminbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5d,0xf0]
          vminbf16 %zmm24, %zmm23, %zmm22

// CHECK: vminbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5d,0xf0]
          vminbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vminbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5d,0xf0]
          vminbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vminbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5d,0xf0]
          vminbf16 %xmm24, %xmm23, %xmm22

// CHECK: vminbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5d,0xf0]
          vminbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vminbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5d,0xf0]
          vminbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vminbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vminbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vminbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5d,0x35,0x00,0x00,0x00,0x00]
          vminbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vminbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5d,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vminbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vminbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5d,0x71,0x7f]
          vminbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vminbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5d,0x72,0x80]
          vminbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vminbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vminbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vminbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vminbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5d,0x35,0x00,0x00,0x00,0x00]
          vminbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vminbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5d,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vminbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vminbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5d,0x71,0x7f]
          vminbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vminbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5d,0x72,0x80]
          vminbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vminbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vminbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vminbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5d,0x35,0x00,0x00,0x00,0x00]
          vminbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vminbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5d,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vminbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vminbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5d,0x71,0x7f]
          vminbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vminbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5d,0x72,0x80]
          vminbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vmulbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x59,0xf0]
          vmulbf16 %ymm24, %ymm23, %ymm22

// CHECK: vmulbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x59,0xf0]
          vmulbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vmulbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x59,0xf0]
          vmulbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmulbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x59,0xf0]
          vmulbf16 %zmm24, %zmm23, %zmm22

// CHECK: vmulbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x59,0xf0]
          vmulbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vmulbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x59,0xf0]
          vmulbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vmulbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x59,0xf0]
          vmulbf16 %xmm24, %xmm23, %xmm22

// CHECK: vmulbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x59,0xf0]
          vmulbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vmulbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x59,0xf0]
          vmulbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vmulbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vmulbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x59,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmulbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vmulbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x59,0x35,0x00,0x00,0x00,0x00]
          vmulbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vmulbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x59,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vmulbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vmulbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x59,0x71,0x7f]
          vmulbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vmulbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x59,0x72,0x80]
          vmulbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vmulbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vmulbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x59,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmulbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vmulbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x59,0x35,0x00,0x00,0x00,0x00]
          vmulbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vmulbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x59,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vmulbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vmulbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x59,0x71,0x7f]
          vmulbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vmulbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x59,0x72,0x80]
          vmulbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmulbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vmulbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x59,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmulbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vmulbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x59,0x35,0x00,0x00,0x00,0x00]
          vmulbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vmulbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x59,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vmulbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vmulbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x59,0x71,0x7f]
          vmulbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vmulbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x59,0x72,0x80]
          vmulbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vrcpbf16 %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4c,0xf7]
          vrcpbf16 %xmm23, %xmm22

// CHECK: vrcpbf16 %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x7c,0x0f,0x4c,0xf7]
          vrcpbf16 %xmm23, %xmm22 {%k7}

// CHECK: vrcpbf16 %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x7c,0x8f,0x4c,0xf7]
          vrcpbf16 %xmm23, %xmm22 {%k7} {z}

// CHECK: vrcpbf16 %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4c,0xf7]
          vrcpbf16 %zmm23, %zmm22

// CHECK: vrcpbf16 %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x7c,0x4f,0x4c,0xf7]
          vrcpbf16 %zmm23, %zmm22 {%k7}

// CHECK: vrcpbf16 %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x7c,0xcf,0x4c,0xf7]
          vrcpbf16 %zmm23, %zmm22 {%k7} {z}

// CHECK: vrcpbf16 %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4c,0xf7]
          vrcpbf16 %ymm23, %ymm22

// CHECK: vrcpbf16 %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x7c,0x2f,0x4c,0xf7]
          vrcpbf16 %ymm23, %ymm22 {%k7}

// CHECK: vrcpbf16 %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x7c,0xaf,0x4c,0xf7]
          vrcpbf16 %ymm23, %ymm22 {%k7} {z}

// CHECK: vrcpbf16  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcpbf16  268435456(%rbp,%r14,8), %xmm22

// CHECK: vrcpbf16  291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x7c,0x0f,0x4c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrcpbf16  291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vrcpbf16  (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x18,0x4c,0x35,0x00,0x00,0x00,0x00]
          vrcpbf16  (%rip){1to8}, %xmm22

// CHECK: vrcpbf16  -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x08,0x4c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vrcpbf16  -512(,%rbp,2), %xmm22

// CHECK: vrcpbf16  2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0x8f,0x4c,0x71,0x7f]
          vrcpbf16  2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vrcpbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0x9f,0x4c,0x72,0x80]
          vrcpbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vrcpbf16  268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcpbf16  268435456(%rbp,%r14,8), %ymm22

// CHECK: vrcpbf16  291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x7c,0x2f,0x4c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrcpbf16  291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vrcpbf16  (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x38,0x4c,0x35,0x00,0x00,0x00,0x00]
          vrcpbf16  (%rip){1to16}, %ymm22

// CHECK: vrcpbf16  -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x28,0x4c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vrcpbf16  -1024(,%rbp,2), %ymm22

// CHECK: vrcpbf16  4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xaf,0x4c,0x71,0x7f]
          vrcpbf16  4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vrcpbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xbf,0x4c,0x72,0x80]
          vrcpbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vrcpbf16  268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcpbf16  268435456(%rbp,%r14,8), %zmm22

// CHECK: vrcpbf16  291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x7c,0x4f,0x4c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrcpbf16  291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vrcpbf16  (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x58,0x4c,0x35,0x00,0x00,0x00,0x00]
          vrcpbf16  (%rip){1to32}, %zmm22

// CHECK: vrcpbf16  -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x48,0x4c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vrcpbf16  -2048(,%rbp,2), %zmm22

// CHECK: vrcpbf16  8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xcf,0x4c,0x71,0x7f]
          vrcpbf16  8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vrcpbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xdf,0x4c,0x72,0x80]
          vrcpbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vreducebf16 $123, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x56,0xf7,0x7b]
          vreducebf16 $123, %zmm23, %zmm22

// CHECK: vreducebf16 $123, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x4f,0x56,0xf7,0x7b]
          vreducebf16 $123, %zmm23, %zmm22 {%k7}

// CHECK: vreducebf16 $123, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0xcf,0x56,0xf7,0x7b]
          vreducebf16 $123, %zmm23, %zmm22 {%k7} {z}

// CHECK: vreducebf16 $123, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x56,0xf7,0x7b]
          vreducebf16 $123, %ymm23, %ymm22

// CHECK: vreducebf16 $123, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x2f,0x56,0xf7,0x7b]
          vreducebf16 $123, %ymm23, %ymm22 {%k7}

// CHECK: vreducebf16 $123, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0xaf,0x56,0xf7,0x7b]
          vreducebf16 $123, %ymm23, %ymm22 {%k7} {z}

// CHECK: vreducebf16 $123, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x56,0xf7,0x7b]
          vreducebf16 $123, %xmm23, %xmm22

// CHECK: vreducebf16 $123, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x0f,0x56,0xf7,0x7b]
          vreducebf16 $123, %xmm23, %xmm22 {%k7}

// CHECK: vreducebf16 $123, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0x8f,0x56,0xf7,0x7b]
          vreducebf16 $123, %xmm23, %xmm22 {%k7} {z}

// CHECK: vreducebf16  $123, 268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreducebf16  $123, 268435456(%rbp,%r14,8), %xmm22

// CHECK: vreducebf16  $123, 291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x0f,0x56,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vreducebf16  $123, 291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vreducebf16  $123, (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x18,0x56,0x35,0x00,0x00,0x00,0x00,0x7b]
          vreducebf16  $123, (%rip){1to8}, %xmm22

// CHECK: vreducebf16  $123, -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x08,0x56,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vreducebf16  $123, -512(,%rbp,2), %xmm22

// CHECK: vreducebf16  $123, 2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0x8f,0x56,0x71,0x7f,0x7b]
          vreducebf16  $123, 2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vreducebf16  $123, -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0x9f,0x56,0x72,0x80,0x7b]
          vreducebf16  $123, -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vreducebf16  $123, 268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreducebf16  $123, 268435456(%rbp,%r14,8), %ymm22

// CHECK: vreducebf16  $123, 291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x2f,0x56,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vreducebf16  $123, 291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vreducebf16  $123, (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x38,0x56,0x35,0x00,0x00,0x00,0x00,0x7b]
          vreducebf16  $123, (%rip){1to16}, %ymm22

// CHECK: vreducebf16  $123, -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x28,0x56,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vreducebf16  $123, -1024(,%rbp,2), %ymm22

// CHECK: vreducebf16  $123, 4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xaf,0x56,0x71,0x7f,0x7b]
          vreducebf16  $123, 4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vreducebf16  $123, -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xbf,0x56,0x72,0x80,0x7b]
          vreducebf16  $123, -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vreducebf16  $123, 268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreducebf16  $123, 268435456(%rbp,%r14,8), %zmm22

// CHECK: vreducebf16  $123, 291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x4f,0x56,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vreducebf16  $123, 291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vreducebf16  $123, (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x58,0x56,0x35,0x00,0x00,0x00,0x00,0x7b]
          vreducebf16  $123, (%rip){1to32}, %zmm22

// CHECK: vreducebf16  $123, -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x56,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vreducebf16  $123, -2048(,%rbp,2), %zmm22

// CHECK: vreducebf16  $123, 8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xcf,0x56,0x71,0x7f,0x7b]
          vreducebf16  $123, 8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vreducebf16  $123, -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xdf,0x56,0x72,0x80,0x7b]
          vreducebf16  $123, -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vrndscalebf16 $123, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x08,0xf7,0x7b]
          vrndscalebf16 $123, %zmm23, %zmm22

// CHECK: vrndscalebf16 $123, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x4f,0x08,0xf7,0x7b]
          vrndscalebf16 $123, %zmm23, %zmm22 {%k7}

// CHECK: vrndscalebf16 $123, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0xcf,0x08,0xf7,0x7b]
          vrndscalebf16 $123, %zmm23, %zmm22 {%k7} {z}

// CHECK: vrndscalebf16 $123, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x08,0xf7,0x7b]
          vrndscalebf16 $123, %ymm23, %ymm22

// CHECK: vrndscalebf16 $123, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x2f,0x08,0xf7,0x7b]
          vrndscalebf16 $123, %ymm23, %ymm22 {%k7}

// CHECK: vrndscalebf16 $123, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0xaf,0x08,0xf7,0x7b]
          vrndscalebf16 $123, %ymm23, %ymm22 {%k7} {z}

// CHECK: vrndscalebf16 $123, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x08,0xf7,0x7b]
          vrndscalebf16 $123, %xmm23, %xmm22

// CHECK: vrndscalebf16 $123, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x7f,0x0f,0x08,0xf7,0x7b]
          vrndscalebf16 $123, %xmm23, %xmm22 {%k7}

// CHECK: vrndscalebf16 $123, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x7f,0x8f,0x08,0xf7,0x7b]
          vrndscalebf16 $123, %xmm23, %xmm22 {%k7} {z}

// CHECK: vrndscalebf16  $123, 268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscalebf16  $123, 268435456(%rbp,%r14,8), %xmm22

// CHECK: vrndscalebf16  $123, 291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x0f,0x08,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vrndscalebf16  $123, 291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vrndscalebf16  $123, (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x18,0x08,0x35,0x00,0x00,0x00,0x00,0x7b]
          vrndscalebf16  $123, (%rip){1to8}, %xmm22

// CHECK: vrndscalebf16  $123, -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x08,0x08,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vrndscalebf16  $123, -512(,%rbp,2), %xmm22

// CHECK: vrndscalebf16  $123, 2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0x8f,0x08,0x71,0x7f,0x7b]
          vrndscalebf16  $123, 2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vrndscalebf16  $123, -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0x9f,0x08,0x72,0x80,0x7b]
          vrndscalebf16  $123, -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vrndscalebf16  $123, 268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscalebf16  $123, 268435456(%rbp,%r14,8), %ymm22

// CHECK: vrndscalebf16  $123, 291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x2f,0x08,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vrndscalebf16  $123, 291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vrndscalebf16  $123, (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x38,0x08,0x35,0x00,0x00,0x00,0x00,0x7b]
          vrndscalebf16  $123, (%rip){1to16}, %ymm22

// CHECK: vrndscalebf16  $123, -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x28,0x08,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vrndscalebf16  $123, -1024(,%rbp,2), %ymm22

// CHECK: vrndscalebf16  $123, 4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xaf,0x08,0x71,0x7f,0x7b]
          vrndscalebf16  $123, 4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vrndscalebf16  $123, -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xbf,0x08,0x72,0x80,0x7b]
          vrndscalebf16  $123, -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vrndscalebf16  $123, 268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscalebf16  $123, 268435456(%rbp,%r14,8), %zmm22

// CHECK: vrndscalebf16  $123, 291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x7f,0x4f,0x08,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vrndscalebf16  $123, 291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vrndscalebf16  $123, (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x58,0x08,0x35,0x00,0x00,0x00,0x00,0x7b]
          vrndscalebf16  $123, (%rip){1to32}, %zmm22

// CHECK: vrndscalebf16  $123, -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x08,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vrndscalebf16  $123, -2048(,%rbp,2), %zmm22

// CHECK: vrndscalebf16  $123, 8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xcf,0x08,0x71,0x7f,0x7b]
          vrndscalebf16  $123, 8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vrndscalebf16  $123, -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x7f,0xdf,0x08,0x72,0x80,0x7b]
          vrndscalebf16  $123, -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vrsqrtbf16 %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4e,0xf7]
          vrsqrtbf16 %xmm23, %xmm22

// CHECK: vrsqrtbf16 %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x7c,0x0f,0x4e,0xf7]
          vrsqrtbf16 %xmm23, %xmm22 {%k7}

// CHECK: vrsqrtbf16 %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x7c,0x8f,0x4e,0xf7]
          vrsqrtbf16 %xmm23, %xmm22 {%k7} {z}

// CHECK: vrsqrtbf16 %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4e,0xf7]
          vrsqrtbf16 %zmm23, %zmm22

// CHECK: vrsqrtbf16 %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x7c,0x4f,0x4e,0xf7]
          vrsqrtbf16 %zmm23, %zmm22 {%k7}

// CHECK: vrsqrtbf16 %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x7c,0xcf,0x4e,0xf7]
          vrsqrtbf16 %zmm23, %zmm22 {%k7} {z}

// CHECK: vrsqrtbf16 %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4e,0xf7]
          vrsqrtbf16 %ymm23, %ymm22

// CHECK: vrsqrtbf16 %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x7c,0x2f,0x4e,0xf7]
          vrsqrtbf16 %ymm23, %ymm22 {%k7}

// CHECK: vrsqrtbf16 %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x7c,0xaf,0x4e,0xf7]
          vrsqrtbf16 %ymm23, %ymm22 {%k7} {z}

// CHECK: vrsqrtbf16  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtbf16  268435456(%rbp,%r14,8), %xmm22

// CHECK: vrsqrtbf16  291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x7c,0x0f,0x4e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrsqrtbf16  291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vrsqrtbf16  (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x18,0x4e,0x35,0x00,0x00,0x00,0x00]
          vrsqrtbf16  (%rip){1to8}, %xmm22

// CHECK: vrsqrtbf16  -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x08,0x4e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vrsqrtbf16  -512(,%rbp,2), %xmm22

// CHECK: vrsqrtbf16  2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0x8f,0x4e,0x71,0x7f]
          vrsqrtbf16  2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vrsqrtbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0x9f,0x4e,0x72,0x80]
          vrsqrtbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vrsqrtbf16  268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtbf16  268435456(%rbp,%r14,8), %ymm22

// CHECK: vrsqrtbf16  291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x7c,0x2f,0x4e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrsqrtbf16  291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vrsqrtbf16  (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x38,0x4e,0x35,0x00,0x00,0x00,0x00]
          vrsqrtbf16  (%rip){1to16}, %ymm22

// CHECK: vrsqrtbf16  -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x28,0x4e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vrsqrtbf16  -1024(,%rbp,2), %ymm22

// CHECK: vrsqrtbf16  4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xaf,0x4e,0x71,0x7f]
          vrsqrtbf16  4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vrsqrtbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xbf,0x4e,0x72,0x80]
          vrsqrtbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vrsqrtbf16  268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtbf16  268435456(%rbp,%r14,8), %zmm22

// CHECK: vrsqrtbf16  291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x7c,0x4f,0x4e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrsqrtbf16  291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vrsqrtbf16  (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x58,0x4e,0x35,0x00,0x00,0x00,0x00]
          vrsqrtbf16  (%rip){1to32}, %zmm22

// CHECK: vrsqrtbf16  -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe6,0x7c,0x48,0x4e,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vrsqrtbf16  -2048(,%rbp,2), %zmm22

// CHECK: vrsqrtbf16  8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xcf,0x4e,0x71,0x7f]
          vrsqrtbf16  8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vrsqrtbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x7c,0xdf,0x4e,0x72,0x80]
          vrsqrtbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vscalefbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x2c,0xf0]
          vscalefbf16 %ymm24, %ymm23, %ymm22

// CHECK: vscalefbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x2c,0xf0]
          vscalefbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vscalefbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x2c,0xf0]
          vscalefbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vscalefbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x2c,0xf0]
          vscalefbf16 %zmm24, %zmm23, %zmm22

// CHECK: vscalefbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x2c,0xf0]
          vscalefbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vscalefbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x2c,0xf0]
          vscalefbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vscalefbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x2c,0xf0]
          vscalefbf16 %xmm24, %xmm23, %xmm22

// CHECK: vscalefbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x2c,0xf0]
          vscalefbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vscalefbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x2c,0xf0]
          vscalefbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vscalefbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vscalefbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x2c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vscalefbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vscalefbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x2c,0x35,0x00,0x00,0x00,0x00]
          vscalefbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vscalefbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x2c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vscalefbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vscalefbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x2c,0x71,0x7f]
          vscalefbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vscalefbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x2c,0x72,0x80]
          vscalefbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vscalefbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vscalefbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x2c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vscalefbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vscalefbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x2c,0x35,0x00,0x00,0x00,0x00]
          vscalefbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vscalefbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x2c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vscalefbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vscalefbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x2c,0x71,0x7f]
          vscalefbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vscalefbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x2c,0x72,0x80]
          vscalefbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vscalefbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vscalefbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x2c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vscalefbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vscalefbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x2c,0x35,0x00,0x00,0x00,0x00]
          vscalefbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vscalefbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x2c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vscalefbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vscalefbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x2c,0x71,0x7f]
          vscalefbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vscalefbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x2c,0x72,0x80]
          vscalefbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vsqrtbf16 %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x51,0xf7]
          vsqrtbf16 %xmm23, %xmm22

// CHECK: vsqrtbf16 %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x51,0xf7]
          vsqrtbf16 %xmm23, %xmm22 {%k7}

// CHECK: vsqrtbf16 %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x51,0xf7]
          vsqrtbf16 %xmm23, %xmm22 {%k7} {z}

// CHECK: vsqrtbf16 %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x51,0xf7]
          vsqrtbf16 %zmm23, %zmm22

// CHECK: vsqrtbf16 %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x51,0xf7]
          vsqrtbf16 %zmm23, %zmm22 {%k7}

// CHECK: vsqrtbf16 %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7d,0xcf,0x51,0xf7]
          vsqrtbf16 %zmm23, %zmm22 {%k7} {z}

// CHECK: vsqrtbf16 %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x51,0xf7]
          vsqrtbf16 %ymm23, %ymm22

// CHECK: vsqrtbf16 %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x51,0xf7]
          vsqrtbf16 %ymm23, %ymm22 {%k7}

// CHECK: vsqrtbf16 %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7d,0xaf,0x51,0xf7]
          vsqrtbf16 %ymm23, %ymm22 {%k7} {z}

// CHECK: vsqrtbf16  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtbf16  268435456(%rbp,%r14,8), %xmm22

// CHECK: vsqrtbf16  291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsqrtbf16  291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vsqrtbf16  (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x51,0x35,0x00,0x00,0x00,0x00]
          vsqrtbf16  (%rip){1to8}, %xmm22

// CHECK: vsqrtbf16  -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x51,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vsqrtbf16  -512(,%rbp,2), %xmm22

// CHECK: vsqrtbf16  2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x51,0x71,0x7f]
          vsqrtbf16  2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vsqrtbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x51,0x72,0x80]
          vsqrtbf16  -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vsqrtbf16  268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtbf16  268435456(%rbp,%r14,8), %ymm22

// CHECK: vsqrtbf16  291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsqrtbf16  291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vsqrtbf16  (%rip){1to16}, %ymm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x51,0x35,0x00,0x00,0x00,0x00]
          vsqrtbf16  (%rip){1to16}, %ymm22

// CHECK: vsqrtbf16  -1024(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x51,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vsqrtbf16  -1024(,%rbp,2), %ymm22

// CHECK: vsqrtbf16  4064(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x51,0x71,0x7f]
          vsqrtbf16  4064(%rcx), %ymm22 {%k7} {z}

// CHECK: vsqrtbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x51,0x72,0x80]
          vsqrtbf16  -256(%rdx){1to16}, %ymm22 {%k7} {z}

// CHECK: vsqrtbf16  268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtbf16  268435456(%rbp,%r14,8), %zmm22

// CHECK: vsqrtbf16  291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsqrtbf16  291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vsqrtbf16  (%rip){1to32}, %zmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x51,0x35,0x00,0x00,0x00,0x00]
          vsqrtbf16  (%rip){1to32}, %zmm22

// CHECK: vsqrtbf16  -2048(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x51,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vsqrtbf16  -2048(,%rbp,2), %zmm22

// CHECK: vsqrtbf16  8128(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x51,0x71,0x7f]
          vsqrtbf16  8128(%rcx), %zmm22 {%k7} {z}

// CHECK: vsqrtbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x51,0x72,0x80]
          vsqrtbf16  -256(%rdx){1to32}, %zmm22 {%k7} {z}

// CHECK: vsubbf16 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5c,0xf0]
          vsubbf16 %ymm24, %ymm23, %ymm22

// CHECK: vsubbf16 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5c,0xf0]
          vsubbf16 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vsubbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5c,0xf0]
          vsubbf16 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vsubbf16 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5c,0xf0]
          vsubbf16 %zmm24, %zmm23, %zmm22

// CHECK: vsubbf16 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5c,0xf0]
          vsubbf16 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vsubbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5c,0xf0]
          vsubbf16 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vsubbf16 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5c,0xf0]
          vsubbf16 %xmm24, %xmm23, %xmm22

// CHECK: vsubbf16 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5c,0xf0]
          vsubbf16 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vsubbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5c,0xf0]
          vsubbf16 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vsubbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubbf16  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vsubbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsubbf16  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vsubbf16  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5c,0x35,0x00,0x00,0x00,0x00]
          vsubbf16  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vsubbf16  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vsubbf16  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vsubbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5c,0x71,0x7f]
          vsubbf16  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vsubbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5c,0x72,0x80]
          vsubbf16  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vsubbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubbf16  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vsubbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsubbf16  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vsubbf16  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5c,0x35,0x00,0x00,0x00,0x00]
          vsubbf16  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vsubbf16  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vsubbf16  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vsubbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5c,0x71,0x7f]
          vsubbf16  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vsubbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5c,0x72,0x80]
          vsubbf16  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vsubbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubbf16  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vsubbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsubbf16  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vsubbf16  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5c,0x35,0x00,0x00,0x00,0x00]
          vsubbf16  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vsubbf16  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vsubbf16  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vsubbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5c,0x71,0x7f]
          vsubbf16  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vsubbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5c,0x72,0x80]
          vsubbf16  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

