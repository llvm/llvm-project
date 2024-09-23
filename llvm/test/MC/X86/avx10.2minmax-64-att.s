// RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s

// CHECK: vminmaxnepbf16 $123, %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x83,0x47,0x00,0x52,0xf0,0x7b]
          vminmaxnepbf16 $123, %xmm24, %xmm23, %xmm22

// CHECK: vminmaxnepbf16 $123, %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x47,0x07,0x52,0xf0,0x7b]
          vminmaxnepbf16 $123, %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vminmaxnepbf16 $123, %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x47,0x87,0x52,0xf0,0x7b]
          vminmaxnepbf16 $123, %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxnepbf16 $123, %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x83,0x47,0x40,0x52,0xf0,0x7b]
          vminmaxnepbf16 $123, %zmm24, %zmm23, %zmm22

// CHECK: vminmaxnepbf16 $123, %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x47,0x47,0x52,0xf0,0x7b]
          vminmaxnepbf16 $123, %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vminmaxnepbf16 $123, %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x47,0xc7,0x52,0xf0,0x7b]
          vminmaxnepbf16 $123, %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vminmaxnepbf16 $123, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x83,0x47,0x20,0x52,0xf0,0x7b]
          vminmaxnepbf16 $123, %ymm24, %ymm23, %ymm22

// CHECK: vminmaxnepbf16 $123, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x47,0x27,0x52,0xf0,0x7b]
          vminmaxnepbf16 $123, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vminmaxnepbf16 $123, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x47,0xa7,0x52,0xf0,0x7b]
          vminmaxnepbf16 $123, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminmaxnepbf16  $123, 268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x47,0x20,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxnepbf16  $123, 268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vminmaxnepbf16  $123, 291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x47,0x27,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxnepbf16  $123, 291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vminmaxnepbf16  $123, (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe3,0x47,0x30,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxnepbf16  $123, (%rip){1to16}, %ymm23, %ymm22

// CHECK: vminmaxnepbf16  $123, -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe3,0x47,0x20,0x52,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxnepbf16  $123, -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vminmaxnepbf16  $123, 4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x47,0xa7,0x52,0x71,0x7f,0x7b]
          vminmaxnepbf16  $123, 4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vminmaxnepbf16  $123, -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x47,0xb7,0x52,0x72,0x80,0x7b]
          vminmaxnepbf16  $123, -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminmaxnepbf16  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0x47,0x00,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxnepbf16  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vminmaxnepbf16  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x47,0x07,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxnepbf16  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vminmaxnepbf16  $123, (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0x47,0x10,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxnepbf16  $123, (%rip){1to8}, %xmm23, %xmm22

// CHECK: vminmaxnepbf16  $123, -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0x47,0x00,0x52,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxnepbf16  $123, -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vminmaxnepbf16  $123, 2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x47,0x87,0x52,0x71,0x7f,0x7b]
          vminmaxnepbf16  $123, 2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxnepbf16  $123, -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x47,0x97,0x52,0x72,0x80,0x7b]
          vminmaxnepbf16  $123, -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxnepbf16  $123, 268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa3,0x47,0x40,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxnepbf16  $123, 268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vminmaxnepbf16  $123, 291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x47,0x47,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxnepbf16  $123, 291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vminmaxnepbf16  $123, (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe3,0x47,0x50,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxnepbf16  $123, (%rip){1to32}, %zmm23, %zmm22

// CHECK: vminmaxnepbf16  $123, -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe3,0x47,0x40,0x52,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxnepbf16  $123, -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vminmaxnepbf16  $123, 8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x47,0xc7,0x52,0x71,0x7f,0x7b]
          vminmaxnepbf16  $123, 8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vminmaxnepbf16  $123, -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x47,0xd7,0x52,0x72,0x80,0x7b]
          vminmaxnepbf16  $123, -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vminmaxpd $123, %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x83,0xc5,0x00,0x52,0xf0,0x7b]
          vminmaxpd $123, %xmm24, %xmm23, %xmm22

// CHECK: vminmaxpd $123, %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x83,0xc5,0x07,0x52,0xf0,0x7b]
          vminmaxpd $123, %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vminmaxpd $123, %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0xc5,0x87,0x52,0xf0,0x7b]
          vminmaxpd $123, %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxpd $123, %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x83,0xc5,0x40,0x52,0xf0,0x7b]
          vminmaxpd $123, %zmm24, %zmm23, %zmm22

// CHECK: vminmaxpd $123, {sae}, %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x83,0xc5,0x10,0x52,0xf0,0x7b]
          vminmaxpd $123, {sae}, %zmm24, %zmm23, %zmm22

// CHECK: vminmaxpd $123, %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x83,0xc5,0x47,0x52,0xf0,0x7b]
          vminmaxpd $123, %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vminmaxpd $123, {sae}, %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0xc5,0x97,0x52,0xf0,0x7b]
          vminmaxpd $123, {sae}, %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vminmaxpd $123, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x83,0xc5,0x20,0x52,0xf0,0x7b]
          vminmaxpd $123, %ymm24, %ymm23, %ymm22

// CHECK: vminmaxpd $123, {sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x83,0xc1,0x10,0x52,0xf0,0x7b]
          vminmaxpd $123, {sae}, %ymm24, %ymm23, %ymm22

// CHECK: vminmaxpd $123, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x83,0xc5,0x27,0x52,0xf0,0x7b]
          vminmaxpd $123, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vminmaxpd $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0xc1,0x97,0x52,0xf0,0x7b]
          vminmaxpd $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminmaxpd  $123, 268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0xc5,0x20,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxpd  $123, 268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vminmaxpd  $123, 291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0xc5,0x27,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxpd  $123, 291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vminmaxpd  $123, (%rip){1to4}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe3,0xc5,0x30,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxpd  $123, (%rip){1to4}, %ymm23, %ymm22

// CHECK: vminmaxpd  $123, -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe3,0xc5,0x20,0x52,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxpd  $123, -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vminmaxpd  $123, 4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0xc5,0xa7,0x52,0x71,0x7f,0x7b]
          vminmaxpd  $123, 4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vminmaxpd  $123, -1024(%rdx){1to4}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0xc5,0xb7,0x52,0x72,0x80,0x7b]
          vminmaxpd  $123, -1024(%rdx){1to4}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminmaxpd  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0xc5,0x00,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxpd  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vminmaxpd  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0xc5,0x07,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxpd  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vminmaxpd  $123, (%rip){1to2}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0xc5,0x10,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxpd  $123, (%rip){1to2}, %xmm23, %xmm22

// CHECK: vminmaxpd  $123, -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0xc5,0x00,0x52,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxpd  $123, -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vminmaxpd  $123, 2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0xc5,0x87,0x52,0x71,0x7f,0x7b]
          vminmaxpd  $123, 2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxpd  $123, -1024(%rdx){1to2}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0xc5,0x97,0x52,0x72,0x80,0x7b]
          vminmaxpd  $123, -1024(%rdx){1to2}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxpd  $123, 268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa3,0xc5,0x40,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxpd  $123, 268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vminmaxpd  $123, 291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0xc5,0x47,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxpd  $123, 291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vminmaxpd  $123, (%rip){1to8}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe3,0xc5,0x50,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxpd  $123, (%rip){1to8}, %zmm23, %zmm22

// CHECK: vminmaxpd  $123, -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe3,0xc5,0x40,0x52,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxpd  $123, -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vminmaxpd  $123, 8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0xc5,0xc7,0x52,0x71,0x7f,0x7b]
          vminmaxpd  $123, 8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vminmaxpd  $123, -1024(%rdx){1to8}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0xc5,0xd7,0x52,0x72,0x80,0x7b]
          vminmaxpd  $123, -1024(%rdx){1to8}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vminmaxph $123, %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x83,0x44,0x00,0x52,0xf0,0x7b]
          vminmaxph $123, %xmm24, %xmm23, %xmm22

// CHECK: vminmaxph $123, %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x44,0x07,0x52,0xf0,0x7b]
          vminmaxph $123, %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vminmaxph $123, %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x44,0x87,0x52,0xf0,0x7b]
          vminmaxph $123, %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxph $123, %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x83,0x44,0x40,0x52,0xf0,0x7b]
          vminmaxph $123, %zmm24, %zmm23, %zmm22

// CHECK: vminmaxph $123, {sae}, %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x83,0x44,0x10,0x52,0xf0,0x7b]
          vminmaxph $123, {sae}, %zmm24, %zmm23, %zmm22

// CHECK: vminmaxph $123, %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x44,0x47,0x52,0xf0,0x7b]
          vminmaxph $123, %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vminmaxph $123, {sae}, %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x44,0x97,0x52,0xf0,0x7b]
          vminmaxph $123, {sae}, %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vminmaxph $123, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x83,0x44,0x20,0x52,0xf0,0x7b]
          vminmaxph $123, %ymm24, %ymm23, %ymm22

// CHECK: vminmaxph $123, {sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x83,0x40,0x10,0x52,0xf0,0x7b]
          vminmaxph $123, {sae}, %ymm24, %ymm23, %ymm22

// CHECK: vminmaxph $123, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x44,0x27,0x52,0xf0,0x7b]
          vminmaxph $123, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vminmaxph $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x40,0x97,0x52,0xf0,0x7b]
          vminmaxph $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminmaxph  $123, 268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x44,0x20,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxph  $123, 268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vminmaxph  $123, 291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x44,0x27,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxph  $123, 291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vminmaxph  $123, (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe3,0x44,0x30,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxph  $123, (%rip){1to16}, %ymm23, %ymm22

// CHECK: vminmaxph  $123, -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe3,0x44,0x20,0x52,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxph  $123, -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vminmaxph  $123, 4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x44,0xa7,0x52,0x71,0x7f,0x7b]
          vminmaxph  $123, 4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vminmaxph  $123, -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x44,0xb7,0x52,0x72,0x80,0x7b]
          vminmaxph  $123, -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminmaxph  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0x44,0x00,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxph  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vminmaxph  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x44,0x07,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxph  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vminmaxph  $123, (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0x44,0x10,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxph  $123, (%rip){1to8}, %xmm23, %xmm22

// CHECK: vminmaxph  $123, -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0x44,0x00,0x52,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxph  $123, -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vminmaxph  $123, 2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x44,0x87,0x52,0x71,0x7f,0x7b]
          vminmaxph  $123, 2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxph  $123, -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x44,0x97,0x52,0x72,0x80,0x7b]
          vminmaxph  $123, -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxph  $123, 268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa3,0x44,0x40,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxph  $123, 268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vminmaxph  $123, 291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x44,0x47,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxph  $123, 291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vminmaxph  $123, (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe3,0x44,0x50,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxph  $123, (%rip){1to32}, %zmm23, %zmm22

// CHECK: vminmaxph  $123, -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe3,0x44,0x40,0x52,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxph  $123, -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vminmaxph  $123, 8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x44,0xc7,0x52,0x71,0x7f,0x7b]
          vminmaxph  $123, 8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vminmaxph  $123, -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x44,0xd7,0x52,0x72,0x80,0x7b]
          vminmaxph  $123, -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vminmaxps $123, %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x83,0x45,0x00,0x52,0xf0,0x7b]
          vminmaxps $123, %xmm24, %xmm23, %xmm22

// CHECK: vminmaxps $123, %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x45,0x07,0x52,0xf0,0x7b]
          vminmaxps $123, %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vminmaxps $123, %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x45,0x87,0x52,0xf0,0x7b]
          vminmaxps $123, %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxps $123, %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x83,0x45,0x40,0x52,0xf0,0x7b]
          vminmaxps $123, %zmm24, %zmm23, %zmm22

// CHECK: vminmaxps $123, {sae}, %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x83,0x45,0x10,0x52,0xf0,0x7b]
          vminmaxps $123, {sae}, %zmm24, %zmm23, %zmm22

// CHECK: vminmaxps $123, %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x45,0x47,0x52,0xf0,0x7b]
          vminmaxps $123, %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vminmaxps $123, {sae}, %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x45,0x97,0x52,0xf0,0x7b]
          vminmaxps $123, {sae}, %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vminmaxps $123, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x83,0x45,0x20,0x52,0xf0,0x7b]
          vminmaxps $123, %ymm24, %ymm23, %ymm22

// CHECK: vminmaxps $123, {sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x83,0x41,0x10,0x52,0xf0,0x7b]
          vminmaxps $123, {sae}, %ymm24, %ymm23, %ymm22

// CHECK: vminmaxps $123, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x45,0x27,0x52,0xf0,0x7b]
          vminmaxps $123, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vminmaxps $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x41,0x97,0x52,0xf0,0x7b]
          vminmaxps $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminmaxps  $123, 268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x45,0x20,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxps  $123, 268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vminmaxps  $123, 291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x45,0x27,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxps  $123, 291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vminmaxps  $123, (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe3,0x45,0x30,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxps  $123, (%rip){1to8}, %ymm23, %ymm22

// CHECK: vminmaxps  $123, -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe3,0x45,0x20,0x52,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxps  $123, -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vminmaxps  $123, 4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x45,0xa7,0x52,0x71,0x7f,0x7b]
          vminmaxps  $123, 4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vminmaxps  $123, -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x45,0xb7,0x52,0x72,0x80,0x7b]
          vminmaxps  $123, -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminmaxps  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0x45,0x00,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxps  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vminmaxps  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x45,0x07,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxps  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vminmaxps  $123, (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0x45,0x10,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxps  $123, (%rip){1to4}, %xmm23, %xmm22

// CHECK: vminmaxps  $123, -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0x45,0x00,0x52,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxps  $123, -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vminmaxps  $123, 2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x45,0x87,0x52,0x71,0x7f,0x7b]
          vminmaxps  $123, 2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxps  $123, -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x45,0x97,0x52,0x72,0x80,0x7b]
          vminmaxps  $123, -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxps  $123, 268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa3,0x45,0x40,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxps  $123, 268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vminmaxps  $123, 291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x45,0x47,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxps  $123, 291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vminmaxps  $123, (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe3,0x45,0x50,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxps  $123, (%rip){1to16}, %zmm23, %zmm22

// CHECK: vminmaxps  $123, -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe3,0x45,0x40,0x52,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxps  $123, -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vminmaxps  $123, 8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x45,0xc7,0x52,0x71,0x7f,0x7b]
          vminmaxps  $123, 8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vminmaxps  $123, -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x45,0xd7,0x52,0x72,0x80,0x7b]
          vminmaxps  $123, -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vminmaxsd $123, %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x83,0xc5,0x00,0x53,0xf0,0x7b]
          vminmaxsd $123, %xmm24, %xmm23, %xmm22

// CHECK: vminmaxsd $123, {sae}, %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x83,0xc5,0x10,0x53,0xf0,0x7b]
          vminmaxsd $123, {sae}, %xmm24, %xmm23, %xmm22

// CHECK: vminmaxsd $123, %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x83,0xc5,0x07,0x53,0xf0,0x7b]
          vminmaxsd $123, %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vminmaxsd $123, {sae}, %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0xc5,0x97,0x53,0xf0,0x7b]
          vminmaxsd $123, {sae}, %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxsd  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0xc5,0x00,0x53,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxsd  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vminmaxsd  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0xc5,0x07,0x53,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxsd  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vminmaxsd  $123, (%rip), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0xc5,0x00,0x53,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxsd  $123, (%rip), %xmm23, %xmm22

// CHECK: vminmaxsd  $123, -256(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0xc5,0x00,0x53,0x34,0x6d,0x00,0xff,0xff,0xff,0x7b]
          vminmaxsd  $123, -256(,%rbp,2), %xmm23, %xmm22

// CHECK: vminmaxsd  $123, 1016(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0xc5,0x87,0x53,0x71,0x7f,0x7b]
          vminmaxsd  $123, 1016(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxsd  $123, -1024(%rdx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0xc5,0x87,0x53,0x72,0x80,0x7b]
          vminmaxsd  $123, -1024(%rdx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxsh $123, %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x83,0x44,0x00,0x53,0xf0,0x7b]
          vminmaxsh $123, %xmm24, %xmm23, %xmm22

// CHECK: vminmaxsh $123, {sae}, %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x83,0x44,0x10,0x53,0xf0,0x7b]
          vminmaxsh $123, {sae}, %xmm24, %xmm23, %xmm22

// CHECK: vminmaxsh $123, %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x44,0x07,0x53,0xf0,0x7b]
          vminmaxsh $123, %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vminmaxsh $123, {sae}, %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x44,0x97,0x53,0xf0,0x7b]
          vminmaxsh $123, {sae}, %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxsh  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0x44,0x00,0x53,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxsh  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vminmaxsh  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x44,0x07,0x53,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxsh  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vminmaxsh  $123, (%rip), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0x44,0x00,0x53,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxsh  $123, (%rip), %xmm23, %xmm22

// CHECK: vminmaxsh  $123, -64(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0x44,0x00,0x53,0x34,0x6d,0xc0,0xff,0xff,0xff,0x7b]
          vminmaxsh  $123, -64(,%rbp,2), %xmm23, %xmm22

// CHECK: vminmaxsh  $123, 254(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x44,0x87,0x53,0x71,0x7f,0x7b]
          vminmaxsh  $123, 254(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxsh  $123, -256(%rdx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x44,0x87,0x53,0x72,0x80,0x7b]
          vminmaxsh  $123, -256(%rdx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxss $123, %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x83,0x45,0x00,0x53,0xf0,0x7b]
          vminmaxss $123, %xmm24, %xmm23, %xmm22

// CHECK: vminmaxss $123, {sae}, %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x83,0x45,0x10,0x53,0xf0,0x7b]
          vminmaxss $123, {sae}, %xmm24, %xmm23, %xmm22

// CHECK: vminmaxss $123, %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x45,0x07,0x53,0xf0,0x7b]
          vminmaxss $123, %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vminmaxss $123, {sae}, %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x45,0x97,0x53,0xf0,0x7b]
          vminmaxss $123, {sae}, %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxss  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0x45,0x00,0x53,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxss  $123, 268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vminmaxss  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc3,0x45,0x07,0x53,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxss  $123, 291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vminmaxss  $123, (%rip), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0x45,0x00,0x53,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxss  $123, (%rip), %xmm23, %xmm22

// CHECK: vminmaxss  $123, -128(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe3,0x45,0x00,0x53,0x34,0x6d,0x80,0xff,0xff,0xff,0x7b]
          vminmaxss  $123, -128(,%rbp,2), %xmm23, %xmm22

// CHECK: vminmaxss  $123, 508(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x45,0x87,0x53,0x71,0x7f,0x7b]
          vminmaxss  $123, 508(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vminmaxss  $123, -512(%rdx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe3,0x45,0x87,0x53,0x72,0x80,0x7b]
          vminmaxss  $123, -512(%rdx), %xmm23, %xmm22 {%k7} {z}

