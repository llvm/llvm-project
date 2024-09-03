// RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s

// VNNI FP16

// CHECK: vdpphps %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x44,0x00,0x52,0xf0]
          vdpphps %xmm24, %xmm23, %xmm22

// CHECK: vdpphps %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x07,0x52,0xf0]
          vdpphps %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vdpphps %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0x87,0x52,0xf0]
          vdpphps %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vdpphps %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x44,0x20,0x52,0xf0]
          vdpphps %ymm24, %ymm23, %ymm22

// CHECK: vdpphps %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x27,0x52,0xf0]
          vdpphps %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vdpphps %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0xa7,0x52,0xf0]
          vdpphps %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vdpphps %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x44,0x40,0x52,0xf0]
          vdpphps %zmm24, %zmm23, %zmm22

// CHECK: vdpphps %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x47,0x52,0xf0]
          vdpphps %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vdpphps %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0xc7,0x52,0xf0]
          vdpphps %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vdpphps  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x44,0x00,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdpphps  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vdpphps  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x07,0x52,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdpphps  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vdpphps  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x10,0x52,0x35,0x00,0x00,0x00,0x00]
          vdpphps  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vdpphps  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x00,0x52,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vdpphps  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vdpphps  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0x87,0x52,0x71,0x7f]
          vdpphps  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vdpphps  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0x97,0x52,0x72,0x80]
          vdpphps  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vdpphps  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x44,0x20,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdpphps  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vdpphps  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x27,0x52,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdpphps  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vdpphps  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x44,0x30,0x52,0x35,0x00,0x00,0x00,0x00]
          vdpphps  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vdpphps  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x44,0x20,0x52,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vdpphps  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vdpphps  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xa7,0x52,0x71,0x7f]
          vdpphps  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vdpphps  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xb7,0x52,0x72,0x80]
          vdpphps  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vdpphps  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x44,0x40,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdpphps  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vdpphps  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x47,0x52,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdpphps  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vdpphps  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x50,0x52,0x35,0x00,0x00,0x00,0x00]
          vdpphps  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vdpphps  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x40,0x52,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vdpphps  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vdpphps  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xc7,0x52,0x71,0x7f]
          vdpphps  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vdpphps  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xd7,0x52,0x72,0x80]
          vdpphps  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// VNNI INT8

// CHECK: vpdpbssd %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x47,0x00,0x50,0xf0]
          vpdpbssd %xmm24, %xmm23, %xmm22

// CHECK: vpdpbssd %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x47,0x07,0x50,0xf0]
          vpdpbssd %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vpdpbssd %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x47,0x87,0x50,0xf0]
          vpdpbssd %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbssd %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x47,0x20,0x50,0xf0]
          vpdpbssd %ymm24, %ymm23, %ymm22

// CHECK: vpdpbssd %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x47,0x27,0x50,0xf0]
          vpdpbssd %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vpdpbssd %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x47,0xa7,0x50,0xf0]
          vpdpbssd %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbssd %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x47,0x40,0x50,0xf0]
          vpdpbssd %zmm24, %zmm23, %zmm22

// CHECK: vpdpbssd %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x47,0x47,0x50,0xf0]
          vpdpbssd %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vpdpbssd %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x47,0xc7,0x50,0xf0]
          vpdpbssd %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbssd  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x47,0x00,0x50,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbssd  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vpdpbssd  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x47,0x07,0x50,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbssd  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vpdpbssd  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x47,0x10,0x50,0x35,0x00,0x00,0x00,0x00]
          vpdpbssd  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vpdpbssd  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x47,0x00,0x50,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vpdpbssd  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vpdpbssd  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0x87,0x50,0x71,0x7f]
          vpdpbssd  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbssd  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0x97,0x50,0x72,0x80]
          vpdpbssd  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbssd  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x47,0x20,0x50,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbssd  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vpdpbssd  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x47,0x27,0x50,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbssd  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vpdpbssd  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x47,0x30,0x50,0x35,0x00,0x00,0x00,0x00]
          vpdpbssd  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vpdpbssd  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x47,0x20,0x50,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vpdpbssd  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vpdpbssd  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0xa7,0x50,0x71,0x7f]
          vpdpbssd  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbssd  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0xb7,0x50,0x72,0x80]
          vpdpbssd  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbssd  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x47,0x40,0x50,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbssd  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vpdpbssd  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x47,0x47,0x50,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbssd  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vpdpbssd  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x47,0x50,0x50,0x35,0x00,0x00,0x00,0x00]
          vpdpbssd  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vpdpbssd  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x47,0x40,0x50,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vpdpbssd  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vpdpbssd  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0xc7,0x50,0x71,0x7f]
          vpdpbssd  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbssd  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0xd7,0x50,0x72,0x80]
          vpdpbssd  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbssds %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x47,0x00,0x51,0xf0]
          vpdpbssds %xmm24, %xmm23, %xmm22

// CHECK: vpdpbssds %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x47,0x07,0x51,0xf0]
          vpdpbssds %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vpdpbssds %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x47,0x87,0x51,0xf0]
          vpdpbssds %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbssds %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x47,0x20,0x51,0xf0]
          vpdpbssds %ymm24, %ymm23, %ymm22

// CHECK: vpdpbssds %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x47,0x27,0x51,0xf0]
          vpdpbssds %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vpdpbssds %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x47,0xa7,0x51,0xf0]
          vpdpbssds %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbssds %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x47,0x40,0x51,0xf0]
          vpdpbssds %zmm24, %zmm23, %zmm22

// CHECK: vpdpbssds %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x47,0x47,0x51,0xf0]
          vpdpbssds %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vpdpbssds %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x47,0xc7,0x51,0xf0]
          vpdpbssds %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbssds  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x47,0x00,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbssds  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vpdpbssds  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x47,0x07,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbssds  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vpdpbssds  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x47,0x10,0x51,0x35,0x00,0x00,0x00,0x00]
          vpdpbssds  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vpdpbssds  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x47,0x00,0x51,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vpdpbssds  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vpdpbssds  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0x87,0x51,0x71,0x7f]
          vpdpbssds  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbssds  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0x97,0x51,0x72,0x80]
          vpdpbssds  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbssds  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x47,0x20,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbssds  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vpdpbssds  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x47,0x27,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbssds  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vpdpbssds  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x47,0x30,0x51,0x35,0x00,0x00,0x00,0x00]
          vpdpbssds  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vpdpbssds  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x47,0x20,0x51,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vpdpbssds  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vpdpbssds  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0xa7,0x51,0x71,0x7f]
          vpdpbssds  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbssds  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0xb7,0x51,0x72,0x80]
          vpdpbssds  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbssds  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x47,0x40,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbssds  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vpdpbssds  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x47,0x47,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbssds  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vpdpbssds  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x47,0x50,0x51,0x35,0x00,0x00,0x00,0x00]
          vpdpbssds  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vpdpbssds  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x47,0x40,0x51,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vpdpbssds  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vpdpbssds  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0xc7,0x51,0x71,0x7f]
          vpdpbssds  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbssds  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0xd7,0x51,0x72,0x80]
          vpdpbssds  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbsud %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x46,0x00,0x50,0xf0]
          vpdpbsud %xmm24, %xmm23, %xmm22

// CHECK: vpdpbsud %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x46,0x07,0x50,0xf0]
          vpdpbsud %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vpdpbsud %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x46,0x87,0x50,0xf0]
          vpdpbsud %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbsud %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x46,0x20,0x50,0xf0]
          vpdpbsud %ymm24, %ymm23, %ymm22

// CHECK: vpdpbsud %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x46,0x27,0x50,0xf0]
          vpdpbsud %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vpdpbsud %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x46,0xa7,0x50,0xf0]
          vpdpbsud %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbsud %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x46,0x40,0x50,0xf0]
          vpdpbsud %zmm24, %zmm23, %zmm22

// CHECK: vpdpbsud %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x46,0x47,0x50,0xf0]
          vpdpbsud %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vpdpbsud %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x46,0xc7,0x50,0xf0]
          vpdpbsud %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbsud  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x46,0x00,0x50,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbsud  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vpdpbsud  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x46,0x07,0x50,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbsud  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vpdpbsud  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x10,0x50,0x35,0x00,0x00,0x00,0x00]
          vpdpbsud  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vpdpbsud  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x00,0x50,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vpdpbsud  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vpdpbsud  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0x87,0x50,0x71,0x7f]
          vpdpbsud  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbsud  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0x97,0x50,0x72,0x80]
          vpdpbsud  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbsud  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x46,0x20,0x50,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbsud  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vpdpbsud  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x46,0x27,0x50,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbsud  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vpdpbsud  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x46,0x30,0x50,0x35,0x00,0x00,0x00,0x00]
          vpdpbsud  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vpdpbsud  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x46,0x20,0x50,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vpdpbsud  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vpdpbsud  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xa7,0x50,0x71,0x7f]
          vpdpbsud  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbsud  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xb7,0x50,0x72,0x80]
          vpdpbsud  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbsud  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x46,0x40,0x50,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbsud  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vpdpbsud  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x46,0x47,0x50,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbsud  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vpdpbsud  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x50,0x50,0x35,0x00,0x00,0x00,0x00]
          vpdpbsud  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vpdpbsud  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x40,0x50,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vpdpbsud  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vpdpbsud  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xc7,0x50,0x71,0x7f]
          vpdpbsud  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbsud  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xd7,0x50,0x72,0x80]
          vpdpbsud  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbsuds %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x46,0x00,0x51,0xf0]
          vpdpbsuds %xmm24, %xmm23, %xmm22

// CHECK: vpdpbsuds %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x46,0x07,0x51,0xf0]
          vpdpbsuds %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vpdpbsuds %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x46,0x87,0x51,0xf0]
          vpdpbsuds %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbsuds %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x46,0x20,0x51,0xf0]
          vpdpbsuds %ymm24, %ymm23, %ymm22

// CHECK: vpdpbsuds %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x46,0x27,0x51,0xf0]
          vpdpbsuds %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vpdpbsuds %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x46,0xa7,0x51,0xf0]
          vpdpbsuds %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbsuds %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x46,0x40,0x51,0xf0]
          vpdpbsuds %zmm24, %zmm23, %zmm22

// CHECK: vpdpbsuds %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x46,0x47,0x51,0xf0]
          vpdpbsuds %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vpdpbsuds %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x46,0xc7,0x51,0xf0]
          vpdpbsuds %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbsuds  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x46,0x00,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbsuds  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vpdpbsuds  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x46,0x07,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbsuds  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vpdpbsuds  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x10,0x51,0x35,0x00,0x00,0x00,0x00]
          vpdpbsuds  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vpdpbsuds  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x00,0x51,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vpdpbsuds  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vpdpbsuds  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0x87,0x51,0x71,0x7f]
          vpdpbsuds  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbsuds  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0x97,0x51,0x72,0x80]
          vpdpbsuds  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbsuds  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x46,0x20,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbsuds  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vpdpbsuds  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x46,0x27,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbsuds  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vpdpbsuds  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x46,0x30,0x51,0x35,0x00,0x00,0x00,0x00]
          vpdpbsuds  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vpdpbsuds  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x46,0x20,0x51,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vpdpbsuds  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vpdpbsuds  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xa7,0x51,0x71,0x7f]
          vpdpbsuds  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbsuds  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xb7,0x51,0x72,0x80]
          vpdpbsuds  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbsuds  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x46,0x40,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbsuds  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vpdpbsuds  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x46,0x47,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbsuds  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vpdpbsuds  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x50,0x51,0x35,0x00,0x00,0x00,0x00]
          vpdpbsuds  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vpdpbsuds  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x40,0x51,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vpdpbsuds  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vpdpbsuds  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xc7,0x51,0x71,0x7f]
          vpdpbsuds  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbsuds  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xd7,0x51,0x72,0x80]
          vpdpbsuds  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbuud %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x44,0x00,0x50,0xf0]
          vpdpbuud %xmm24, %xmm23, %xmm22

// CHECK: vpdpbuud %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x07,0x50,0xf0]
          vpdpbuud %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vpdpbuud %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0x87,0x50,0xf0]
          vpdpbuud %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbuud %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x44,0x20,0x50,0xf0]
          vpdpbuud %ymm24, %ymm23, %ymm22

// CHECK: vpdpbuud %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x27,0x50,0xf0]
          vpdpbuud %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vpdpbuud %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0xa7,0x50,0xf0]
          vpdpbuud %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbuud %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x44,0x40,0x50,0xf0]
          vpdpbuud %zmm24, %zmm23, %zmm22

// CHECK: vpdpbuud %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x47,0x50,0xf0]
          vpdpbuud %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vpdpbuud %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0xc7,0x50,0xf0]
          vpdpbuud %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbuud  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x44,0x00,0x50,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbuud  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vpdpbuud  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x07,0x50,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbuud  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vpdpbuud  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x10,0x50,0x35,0x00,0x00,0x00,0x00]
          vpdpbuud  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vpdpbuud  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x00,0x50,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vpdpbuud  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vpdpbuud  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0x87,0x50,0x71,0x7f]
          vpdpbuud  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbuud  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0x97,0x50,0x72,0x80]
          vpdpbuud  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbuud  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x44,0x20,0x50,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbuud  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vpdpbuud  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x27,0x50,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbuud  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vpdpbuud  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x44,0x30,0x50,0x35,0x00,0x00,0x00,0x00]
          vpdpbuud  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vpdpbuud  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x44,0x20,0x50,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vpdpbuud  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vpdpbuud  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xa7,0x50,0x71,0x7f]
          vpdpbuud  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbuud  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xb7,0x50,0x72,0x80]
          vpdpbuud  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbuud  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x44,0x40,0x50,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbuud  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vpdpbuud  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x47,0x50,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbuud  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vpdpbuud  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x50,0x50,0x35,0x00,0x00,0x00,0x00]
          vpdpbuud  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vpdpbuud  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x40,0x50,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vpdpbuud  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vpdpbuud  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xc7,0x50,0x71,0x7f]
          vpdpbuud  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbuud  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xd7,0x50,0x72,0x80]
          vpdpbuud  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbuuds %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x44,0x00,0x51,0xf0]
          vpdpbuuds %xmm24, %xmm23, %xmm22

// CHECK: vpdpbuuds %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x07,0x51,0xf0]
          vpdpbuuds %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vpdpbuuds %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0x87,0x51,0xf0]
          vpdpbuuds %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbuuds %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x44,0x20,0x51,0xf0]
          vpdpbuuds %ymm24, %ymm23, %ymm22

// CHECK: vpdpbuuds %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x27,0x51,0xf0]
          vpdpbuuds %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vpdpbuuds %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0xa7,0x51,0xf0]
          vpdpbuuds %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbuuds %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x44,0x40,0x51,0xf0]
          vpdpbuuds %zmm24, %zmm23, %zmm22

// CHECK: vpdpbuuds %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x47,0x51,0xf0]
          vpdpbuuds %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vpdpbuuds %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0xc7,0x51,0xf0]
          vpdpbuuds %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbuuds  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x44,0x00,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbuuds  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vpdpbuuds  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x07,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbuuds  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vpdpbuuds  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x10,0x51,0x35,0x00,0x00,0x00,0x00]
          vpdpbuuds  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vpdpbuuds  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x00,0x51,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vpdpbuuds  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vpdpbuuds  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0x87,0x51,0x71,0x7f]
          vpdpbuuds  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbuuds  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0x97,0x51,0x72,0x80]
          vpdpbuuds  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpbuuds  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x44,0x20,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbuuds  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vpdpbuuds  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x27,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbuuds  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vpdpbuuds  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x44,0x30,0x51,0x35,0x00,0x00,0x00,0x00]
          vpdpbuuds  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vpdpbuuds  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x44,0x20,0x51,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vpdpbuuds  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vpdpbuuds  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xa7,0x51,0x71,0x7f]
          vpdpbuuds  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbuuds  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xb7,0x51,0x72,0x80]
          vpdpbuuds  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpbuuds  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x44,0x40,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpbuuds  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vpdpbuuds  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x47,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpbuuds  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vpdpbuuds  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x50,0x51,0x35,0x00,0x00,0x00,0x00]
          vpdpbuuds  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vpdpbuuds  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x40,0x51,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vpdpbuuds  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vpdpbuuds  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xc7,0x51,0x71,0x7f]
          vpdpbuuds  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpbuuds  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xd7,0x51,0x72,0x80]
          vpdpbuuds  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// VNNI INT16

// CHECK: vpdpwsud %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x46,0x00,0xd2,0xf0]
          vpdpwsud %xmm24, %xmm23, %xmm22

// CHECK: vpdpwsud %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x46,0x07,0xd2,0xf0]
          vpdpwsud %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vpdpwsud %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x46,0x87,0xd2,0xf0]
          vpdpwsud %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwsud %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x46,0x20,0xd2,0xf0]
          vpdpwsud %ymm24, %ymm23, %ymm22

// CHECK: vpdpwsud %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x46,0x27,0xd2,0xf0]
          vpdpwsud %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vpdpwsud %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x46,0xa7,0xd2,0xf0]
          vpdpwsud %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwsud %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x46,0x40,0xd2,0xf0]
          vpdpwsud %zmm24, %zmm23, %zmm22

// CHECK: vpdpwsud %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x46,0x47,0xd2,0xf0]
          vpdpwsud %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vpdpwsud %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x46,0xc7,0xd2,0xf0]
          vpdpwsud %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwsud  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x46,0x00,0xd2,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwsud  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vpdpwsud  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x46,0x07,0xd2,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwsud  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vpdpwsud  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x10,0xd2,0x35,0x00,0x00,0x00,0x00]
          vpdpwsud  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vpdpwsud  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x00,0xd2,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwsud  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vpdpwsud  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0x87,0xd2,0x71,0x7f]
          vpdpwsud  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwsud  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0x97,0xd2,0x72,0x80]
          vpdpwsud  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwsud  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x46,0x20,0xd2,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwsud  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vpdpwsud  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x46,0x27,0xd2,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwsud  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vpdpwsud  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x46,0x30,0xd2,0x35,0x00,0x00,0x00,0x00]
          vpdpwsud  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vpdpwsud  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x46,0x20,0xd2,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwsud  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vpdpwsud  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xa7,0xd2,0x71,0x7f]
          vpdpwsud  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwsud  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xb7,0xd2,0x72,0x80]
          vpdpwsud  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwsud  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x46,0x40,0xd2,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwsud  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vpdpwsud  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x46,0x47,0xd2,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwsud  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vpdpwsud  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x50,0xd2,0x35,0x00,0x00,0x00,0x00]
          vpdpwsud  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vpdpwsud  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x40,0xd2,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vpdpwsud  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vpdpwsud  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xc7,0xd2,0x71,0x7f]
          vpdpwsud  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwsud  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xd7,0xd2,0x72,0x80]
          vpdpwsud  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwsuds %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x46,0x00,0xd3,0xf0]
          vpdpwsuds %xmm24, %xmm23, %xmm22

// CHECK: vpdpwsuds %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x46,0x07,0xd3,0xf0]
          vpdpwsuds %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vpdpwsuds %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x46,0x87,0xd3,0xf0]
          vpdpwsuds %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwsuds %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x46,0x20,0xd3,0xf0]
          vpdpwsuds %ymm24, %ymm23, %ymm22

// CHECK: vpdpwsuds %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x46,0x27,0xd3,0xf0]
          vpdpwsuds %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vpdpwsuds %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x46,0xa7,0xd3,0xf0]
          vpdpwsuds %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwsuds %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x46,0x40,0xd3,0xf0]
          vpdpwsuds %zmm24, %zmm23, %zmm22

// CHECK: vpdpwsuds %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x46,0x47,0xd3,0xf0]
          vpdpwsuds %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vpdpwsuds %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x46,0xc7,0xd3,0xf0]
          vpdpwsuds %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwsuds  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x46,0x00,0xd3,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwsuds  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vpdpwsuds  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x46,0x07,0xd3,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwsuds  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vpdpwsuds  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x10,0xd3,0x35,0x00,0x00,0x00,0x00]
          vpdpwsuds  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vpdpwsuds  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x00,0xd3,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwsuds  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vpdpwsuds  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0x87,0xd3,0x71,0x7f]
          vpdpwsuds  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwsuds  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0x97,0xd3,0x72,0x80]
          vpdpwsuds  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwsuds  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x46,0x20,0xd3,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwsuds  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vpdpwsuds  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x46,0x27,0xd3,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwsuds  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vpdpwsuds  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x46,0x30,0xd3,0x35,0x00,0x00,0x00,0x00]
          vpdpwsuds  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vpdpwsuds  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x46,0x20,0xd3,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwsuds  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vpdpwsuds  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xa7,0xd3,0x71,0x7f]
          vpdpwsuds  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwsuds  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xb7,0xd3,0x72,0x80]
          vpdpwsuds  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwsuds  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x46,0x40,0xd3,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwsuds  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vpdpwsuds  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x46,0x47,0xd3,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwsuds  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vpdpwsuds  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x50,0xd3,0x35,0x00,0x00,0x00,0x00]
          vpdpwsuds  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vpdpwsuds  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x46,0x40,0xd3,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vpdpwsuds  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vpdpwsuds  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xc7,0xd3,0x71,0x7f]
          vpdpwsuds  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwsuds  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x46,0xd7,0xd3,0x72,0x80]
          vpdpwsuds  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwusd %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x45,0x00,0xd2,0xf0]
          vpdpwusd %xmm24, %xmm23, %xmm22

// CHECK: vpdpwusd %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x45,0x07,0xd2,0xf0]
          vpdpwusd %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vpdpwusd %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x45,0x87,0xd2,0xf0]
          vpdpwusd %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwusd %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x45,0x20,0xd2,0xf0]
          vpdpwusd %ymm24, %ymm23, %ymm22

// CHECK: vpdpwusd %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x45,0x27,0xd2,0xf0]
          vpdpwusd %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vpdpwusd %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x45,0xa7,0xd2,0xf0]
          vpdpwusd %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwusd %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x45,0x40,0xd2,0xf0]
          vpdpwusd %zmm24, %zmm23, %zmm22

// CHECK: vpdpwusd %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x45,0x47,0xd2,0xf0]
          vpdpwusd %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vpdpwusd %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x45,0xc7,0xd2,0xf0]
          vpdpwusd %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwusd  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x45,0x00,0xd2,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwusd  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vpdpwusd  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x45,0x07,0xd2,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwusd  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vpdpwusd  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x45,0x10,0xd2,0x35,0x00,0x00,0x00,0x00]
          vpdpwusd  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vpdpwusd  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x45,0x00,0xd2,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwusd  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vpdpwusd  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0x87,0xd2,0x71,0x7f]
          vpdpwusd  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwusd  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0x97,0xd2,0x72,0x80]
          vpdpwusd  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwusd  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x45,0x20,0xd2,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwusd  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vpdpwusd  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x45,0x27,0xd2,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwusd  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vpdpwusd  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x45,0x30,0xd2,0x35,0x00,0x00,0x00,0x00]
          vpdpwusd  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vpdpwusd  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x45,0x20,0xd2,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwusd  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vpdpwusd  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0xa7,0xd2,0x71,0x7f]
          vpdpwusd  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwusd  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0xb7,0xd2,0x72,0x80]
          vpdpwusd  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwusd  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x45,0x40,0xd2,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwusd  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vpdpwusd  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x45,0x47,0xd2,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwusd  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vpdpwusd  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x45,0x50,0xd2,0x35,0x00,0x00,0x00,0x00]
          vpdpwusd  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vpdpwusd  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x45,0x40,0xd2,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vpdpwusd  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vpdpwusd  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0xc7,0xd2,0x71,0x7f]
          vpdpwusd  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwusd  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0xd7,0xd2,0x72,0x80]
          vpdpwusd  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwusds %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x45,0x00,0xd3,0xf0]
          vpdpwusds %xmm24, %xmm23, %xmm22

// CHECK: vpdpwusds %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x45,0x07,0xd3,0xf0]
          vpdpwusds %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vpdpwusds %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x45,0x87,0xd3,0xf0]
          vpdpwusds %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwusds %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x45,0x20,0xd3,0xf0]
          vpdpwusds %ymm24, %ymm23, %ymm22

// CHECK: vpdpwusds %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x45,0x27,0xd3,0xf0]
          vpdpwusds %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vpdpwusds %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x45,0xa7,0xd3,0xf0]
          vpdpwusds %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwusds %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x45,0x40,0xd3,0xf0]
          vpdpwusds %zmm24, %zmm23, %zmm22

// CHECK: vpdpwusds %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x45,0x47,0xd3,0xf0]
          vpdpwusds %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vpdpwusds %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x45,0xc7,0xd3,0xf0]
          vpdpwusds %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwusds  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x45,0x00,0xd3,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwusds  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vpdpwusds  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x45,0x07,0xd3,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwusds  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vpdpwusds  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x45,0x10,0xd3,0x35,0x00,0x00,0x00,0x00]
          vpdpwusds  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vpdpwusds  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x45,0x00,0xd3,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwusds  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vpdpwusds  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0x87,0xd3,0x71,0x7f]
          vpdpwusds  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwusds  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0x97,0xd3,0x72,0x80]
          vpdpwusds  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwusds  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x45,0x20,0xd3,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwusds  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vpdpwusds  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x45,0x27,0xd3,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwusds  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vpdpwusds  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x45,0x30,0xd3,0x35,0x00,0x00,0x00,0x00]
          vpdpwusds  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vpdpwusds  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x45,0x20,0xd3,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwusds  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vpdpwusds  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0xa7,0xd3,0x71,0x7f]
          vpdpwusds  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwusds  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0xb7,0xd3,0x72,0x80]
          vpdpwusds  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwusds  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x45,0x40,0xd3,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwusds  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vpdpwusds  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x45,0x47,0xd3,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwusds  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vpdpwusds  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x45,0x50,0xd3,0x35,0x00,0x00,0x00,0x00]
          vpdpwusds  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vpdpwusds  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x45,0x40,0xd3,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vpdpwusds  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vpdpwusds  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0xc7,0xd3,0x71,0x7f]
          vpdpwusds  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwusds  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0xd7,0xd3,0x72,0x80]
          vpdpwusds  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwuud %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x44,0x00,0xd2,0xf0]
          vpdpwuud %xmm24, %xmm23, %xmm22

// CHECK: vpdpwuud %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x07,0xd2,0xf0]
          vpdpwuud %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vpdpwuud %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0x87,0xd2,0xf0]
          vpdpwuud %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwuud %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x44,0x20,0xd2,0xf0]
          vpdpwuud %ymm24, %ymm23, %ymm22

// CHECK: vpdpwuud %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x27,0xd2,0xf0]
          vpdpwuud %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vpdpwuud %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0xa7,0xd2,0xf0]
          vpdpwuud %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwuud %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x44,0x40,0xd2,0xf0]
          vpdpwuud %zmm24, %zmm23, %zmm22

// CHECK: vpdpwuud %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x47,0xd2,0xf0]
          vpdpwuud %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vpdpwuud %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0xc7,0xd2,0xf0]
          vpdpwuud %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwuud  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x44,0x00,0xd2,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwuud  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vpdpwuud  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x07,0xd2,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwuud  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vpdpwuud  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x10,0xd2,0x35,0x00,0x00,0x00,0x00]
          vpdpwuud  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vpdpwuud  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x00,0xd2,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwuud  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vpdpwuud  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0x87,0xd2,0x71,0x7f]
          vpdpwuud  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwuud  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0x97,0xd2,0x72,0x80]
          vpdpwuud  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwuud  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x44,0x20,0xd2,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwuud  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vpdpwuud  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x27,0xd2,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwuud  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vpdpwuud  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x44,0x30,0xd2,0x35,0x00,0x00,0x00,0x00]
          vpdpwuud  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vpdpwuud  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x44,0x20,0xd2,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwuud  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vpdpwuud  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xa7,0xd2,0x71,0x7f]
          vpdpwuud  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwuud  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xb7,0xd2,0x72,0x80]
          vpdpwuud  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwuud  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x44,0x40,0xd2,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwuud  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vpdpwuud  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x47,0xd2,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwuud  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vpdpwuud  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x50,0xd2,0x35,0x00,0x00,0x00,0x00]
          vpdpwuud  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vpdpwuud  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x40,0xd2,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vpdpwuud  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vpdpwuud  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xc7,0xd2,0x71,0x7f]
          vpdpwuud  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwuud  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xd7,0xd2,0x72,0x80]
          vpdpwuud  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwuuds %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x44,0x00,0xd3,0xf0]
          vpdpwuuds %xmm24, %xmm23, %xmm22

// CHECK: vpdpwuuds %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x07,0xd3,0xf0]
          vpdpwuuds %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vpdpwuuds %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0x87,0xd3,0xf0]
          vpdpwuuds %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwuuds %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x44,0x20,0xd3,0xf0]
          vpdpwuuds %ymm24, %ymm23, %ymm22

// CHECK: vpdpwuuds %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x27,0xd3,0xf0]
          vpdpwuuds %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vpdpwuuds %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0xa7,0xd3,0xf0]
          vpdpwuuds %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwuuds %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x44,0x40,0xd3,0xf0]
          vpdpwuuds %zmm24, %zmm23, %zmm22

// CHECK: vpdpwuuds %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x47,0xd3,0xf0]
          vpdpwuuds %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vpdpwuuds %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0xc7,0xd3,0xf0]
          vpdpwuuds %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwuuds  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x44,0x00,0xd3,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwuuds  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vpdpwuuds  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x07,0xd3,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwuuds  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vpdpwuuds  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x10,0xd3,0x35,0x00,0x00,0x00,0x00]
          vpdpwuuds  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vpdpwuuds  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x00,0xd3,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwuuds  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vpdpwuuds  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0x87,0xd3,0x71,0x7f]
          vpdpwuuds  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwuuds  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0x97,0xd3,0x72,0x80]
          vpdpwuuds  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vpdpwuuds  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x44,0x20,0xd3,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwuuds  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vpdpwuuds  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x27,0xd3,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwuuds  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vpdpwuuds  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x44,0x30,0xd3,0x35,0x00,0x00,0x00,0x00]
          vpdpwuuds  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vpdpwuuds  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x44,0x20,0xd3,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwuuds  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vpdpwuuds  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xa7,0xd3,0x71,0x7f]
          vpdpwuuds  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwuuds  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xb7,0xd3,0x72,0x80]
          vpdpwuuds  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vpdpwuuds  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x44,0x40,0xd3,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwuuds  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vpdpwuuds  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x47,0xd3,0xb4,0x80,0x23,0x01,0x00,0x00]
          vpdpwuuds  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vpdpwuuds  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x50,0xd3,0x35,0x00,0x00,0x00,0x00]
          vpdpwuuds  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vpdpwuuds  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x40,0xd3,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vpdpwuuds  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vpdpwuuds  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xc7,0xd3,0x71,0x7f]
          vpdpwuuds  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vpdpwuuds  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xd7,0xd3,0x72,0x80]
          vpdpwuuds  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

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

// CHECK: vcmppd $123, {sae}, %ymm24, %ymm23, %k5
// CHECK: encoding: [0x62,0x91,0xc1,0x10,0xc2,0xe8,0x7b]
          vcmppd $123, {sae}, %ymm24, %ymm23, %k5

// CHECK: vcmppd $123, {sae}, %ymm24, %ymm23, %k5 {%k7}
// CHECK: encoding: [0x62,0x91,0xc1,0x17,0xc2,0xe8,0x7b]
          vcmppd $123, {sae}, %ymm24, %ymm23, %k5 {%k7}

// CHECK: vcmpph $123, {sae}, %ymm24, %ymm23, %k5
// CHECK: encoding: [0x62,0x93,0x40,0x10,0xc2,0xe8,0x7b]
          vcmpph $123, {sae}, %ymm24, %ymm23, %k5

// CHECK: vcmpph $123, {sae}, %ymm24, %ymm23, %k5 {%k7}
// CHECK: encoding: [0x62,0x93,0x40,0x17,0xc2,0xe8,0x7b]
          vcmpph $123, {sae}, %ymm24, %ymm23, %k5 {%k7}

// CHECK: vcmpps $123, {sae}, %ymm24, %ymm23, %k5
// CHECK: encoding: [0x62,0x91,0x40,0x10,0xc2,0xe8,0x7b]
          vcmpps $123, {sae}, %ymm24, %ymm23, %k5

// CHECK: vcmpps $123, {sae}, %ymm24, %ymm23, %k5 {%k7}
// CHECK: encoding: [0x62,0x91,0x40,0x17,0xc2,0xe8,0x7b]
          vcmpps $123, {sae}, %ymm24, %ymm23, %k5 {%k7}

// CHECK: vcvtdq2ph {rn-sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x78,0x18,0x5b,0xf7]
          vcvtdq2ph {rn-sae}, %ymm23, %xmm22

// CHECK: vcvtdq2ph {rd-sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x78,0x3f,0x5b,0xf7]
          vcvtdq2ph {rd-sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvtdq2ph {rz-sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x78,0xff,0x5b,0xf7]
          vcvtdq2ph {rz-sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtdq2ps {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0x78,0x18,0x5b,0xf7]
          vcvtdq2ps {rn-sae}, %ymm23, %ymm22

// CHECK: vcvtdq2ps {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0x78,0x3f,0x5b,0xf7]
          vcvtdq2ps {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvtdq2ps {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0x78,0xff,0x5b,0xf7]
          vcvtdq2ps {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvtpd2dq {rn-sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa1,0xfb,0x18,0xe6,0xf7]
          vcvtpd2dq {rn-sae}, %ymm23, %xmm22

// CHECK: vcvtpd2dq {rd-sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xfb,0x3f,0xe6,0xf7]
          vcvtpd2dq {rd-sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvtpd2dq {rz-sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xfb,0xff,0xe6,0xf7]
          vcvtpd2dq {rz-sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtpd2ph {rn-sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0xf9,0x18,0x5a,0xf7]
          vcvtpd2ph {rn-sae}, %ymm23, %xmm22

// CHECK: vcvtpd2ph {rd-sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0xf9,0x3f,0x5a,0xf7]
          vcvtpd2ph {rd-sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvtpd2ph {rz-sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0xf9,0xff,0x5a,0xf7]
          vcvtpd2ph {rz-sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtpd2ps {rn-sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa1,0xf9,0x18,0x5a,0xf7]
          vcvtpd2ps {rn-sae}, %ymm23, %xmm22

// CHECK: vcvtpd2ps {rd-sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xf9,0x3f,0x5a,0xf7]
          vcvtpd2ps {rd-sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvtpd2ps {rz-sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xf9,0xff,0x5a,0xf7]
          vcvtpd2ps {rz-sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtpd2qq {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0xf9,0x18,0x7b,0xf7]
          vcvtpd2qq {rn-sae}, %ymm23, %ymm22

// CHECK: vcvtpd2qq {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xf9,0x3f,0x7b,0xf7]
          vcvtpd2qq {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvtpd2qq {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xf9,0xff,0x7b,0xf7]
          vcvtpd2qq {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvtpd2udq {rn-sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa1,0xf8,0x18,0x79,0xf7]
          vcvtpd2udq {rn-sae}, %ymm23, %xmm22

// CHECK: vcvtpd2udq {rd-sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xf8,0x3f,0x79,0xf7]
          vcvtpd2udq {rd-sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvtpd2udq {rz-sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xf8,0xff,0x79,0xf7]
          vcvtpd2udq {rz-sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtpd2uqq {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0xf9,0x18,0x79,0xf7]
          vcvtpd2uqq {rn-sae}, %ymm23, %ymm22

// CHECK: vcvtpd2uqq {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xf9,0x3f,0x79,0xf7]
          vcvtpd2uqq {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvtpd2uqq {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xf9,0xff,0x79,0xf7]
          vcvtpd2uqq {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvtph2dq {rn-sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x5b,0xf7]
          vcvtph2dq {rn-sae}, %xmm23, %ymm22

// CHECK: vcvtph2dq {rd-sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x79,0x3f,0x5b,0xf7]
          vcvtph2dq {rd-sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvtph2dq {rz-sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x79,0xff,0x5b,0xf7]
          vcvtph2dq {rz-sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvtph2pd {sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x78,0x18,0x5a,0xf7]
          vcvtph2pd {sae}, %xmm23, %ymm22

// CHECK: vcvtph2pd {sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x78,0x1f,0x5a,0xf7]
          vcvtph2pd {sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvtph2pd {sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x78,0x9f,0x5a,0xf7]
          vcvtph2pd {sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvtph2ps {sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x79,0x18,0x13,0xf7]
          vcvtph2ps {sae}, %xmm23, %ymm22

// CHECK: vcvtph2ps {sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa2,0x79,0x1f,0x13,0xf7]
          vcvtph2ps {sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvtph2ps {sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa2,0x79,0x9f,0x13,0xf7]
          vcvtph2ps {sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvtph2psx {sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x79,0x18,0x13,0xf7]
          vcvtph2psx {sae}, %xmm23, %ymm22

// CHECK: vcvtph2psx {sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x79,0x1f,0x13,0xf7]
          vcvtph2psx {sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvtph2psx {sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x79,0x9f,0x13,0xf7]
          vcvtph2psx {sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvtph2qq {rn-sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x7b,0xf7]
          vcvtph2qq {rn-sae}, %xmm23, %ymm22

// CHECK: vcvtph2qq {rd-sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x79,0x3f,0x7b,0xf7]
          vcvtph2qq {rd-sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvtph2qq {rz-sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x79,0xff,0x7b,0xf7]
          vcvtph2qq {rz-sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvtph2udq {rn-sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x78,0x18,0x79,0xf7]
          vcvtph2udq {rn-sae}, %xmm23, %ymm22

// CHECK: vcvtph2udq {rd-sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x78,0x3f,0x79,0xf7]
          vcvtph2udq {rd-sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvtph2udq {rz-sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x78,0xff,0x79,0xf7]
          vcvtph2udq {rz-sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvtph2uqq {rn-sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x79,0xf7]
          vcvtph2uqq {rn-sae}, %xmm23, %ymm22

// CHECK: vcvtph2uqq {rd-sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x79,0x3f,0x79,0xf7]
          vcvtph2uqq {rd-sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvtph2uqq {rz-sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x79,0xff,0x79,0xf7]
          vcvtph2uqq {rz-sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvtph2uw {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x78,0x18,0x7d,0xf7]
          vcvtph2uw {rn-sae}, %ymm23, %ymm22

// CHECK: vcvtph2uw {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x78,0x3f,0x7d,0xf7]
          vcvtph2uw {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvtph2uw {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x78,0xff,0x7d,0xf7]
          vcvtph2uw {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvtph2w {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x7d,0xf7]
          vcvtph2w {rn-sae}, %ymm23, %ymm22

// CHECK: vcvtph2w {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x79,0x3f,0x7d,0xf7]
          vcvtph2w {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvtph2w {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x79,0xff,0x7d,0xf7]
          vcvtph2w {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvtps2dq {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0x79,0x18,0x5b,0xf7]
          vcvtps2dq {rn-sae}, %ymm23, %ymm22

// CHECK: vcvtps2dq {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0x79,0x3f,0x5b,0xf7]
          vcvtps2dq {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvtps2dq {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0x79,0xff,0x5b,0xf7]
          vcvtps2dq {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvtps2pd {sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0x78,0x18,0x5a,0xf7]
          vcvtps2pd {sae}, %xmm23, %ymm22

// CHECK: vcvtps2pd {sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0x78,0x1f,0x5a,0xf7]
          vcvtps2pd {sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvtps2pd {sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0x78,0x9f,0x5a,0xf7]
          vcvtps2pd {sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvtps2ph $123, {sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa3,0x79,0x18,0x1d,0xfe,0x7b]
          vcvtps2ph $123, {sae}, %ymm23, %xmm22

// CHECK: vcvtps2ph $123, {sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x79,0x1f,0x1d,0xfe,0x7b]
          vcvtps2ph $123, {sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvtps2ph $123, {sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x79,0x9f,0x1d,0xfe,0x7b]
          vcvtps2ph $123, {sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtps2phx {rn-sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x1d,0xf7]
          vcvtps2phx {rn-sae}, %ymm23, %xmm22

// CHECK: vcvtps2phx {rd-sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x79,0x3f,0x1d,0xf7]
          vcvtps2phx {rd-sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvtps2phx {rz-sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x79,0xff,0x1d,0xf7]
          vcvtps2phx {rz-sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtps2qq {rn-sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0x79,0x18,0x7b,0xf7]
          vcvtps2qq {rn-sae}, %xmm23, %ymm22

// CHECK: vcvtps2qq {rd-sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0x79,0x3f,0x7b,0xf7]
          vcvtps2qq {rd-sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvtps2qq {rz-sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0x79,0xff,0x7b,0xf7]
          vcvtps2qq {rz-sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvtps2udq {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0x78,0x18,0x79,0xf7]
          vcvtps2udq {rn-sae}, %ymm23, %ymm22

// CHECK: vcvtps2udq {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0x78,0x3f,0x79,0xf7]
          vcvtps2udq {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvtps2udq {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0x78,0xff,0x79,0xf7]
          vcvtps2udq {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvtps2uqq {rn-sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0x79,0x18,0x79,0xf7]
          vcvtps2uqq {rn-sae}, %xmm23, %ymm22

// CHECK: vcvtps2uqq {rd-sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0x79,0x3f,0x79,0xf7]
          vcvtps2uqq {rd-sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvtps2uqq {rz-sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0x79,0xff,0x79,0xf7]
          vcvtps2uqq {rz-sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvtqq2pd {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0xfa,0x18,0xe6,0xf7]
          vcvtqq2pd {rn-sae}, %ymm23, %ymm22

// CHECK: vcvtqq2pd {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xfa,0x3f,0xe6,0xf7]
          vcvtqq2pd {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvtqq2pd {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xfa,0xff,0xe6,0xf7]
          vcvtqq2pd {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvtqq2ph {rn-sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0xf8,0x18,0x5b,0xf7]
          vcvtqq2ph {rn-sae}, %ymm23, %xmm22

// CHECK: vcvtqq2ph {rd-sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0xf8,0x3f,0x5b,0xf7]
          vcvtqq2ph {rd-sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvtqq2ph {rz-sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0xf8,0xff,0x5b,0xf7]
          vcvtqq2ph {rz-sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtqq2ps {rn-sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa1,0xf8,0x18,0x5b,0xf7]
          vcvtqq2ps {rn-sae}, %ymm23, %xmm22

// CHECK: vcvtqq2ps {rd-sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xf8,0x3f,0x5b,0xf7]
          vcvtqq2ps {rd-sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvtqq2ps {rz-sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xf8,0xff,0x5b,0xf7]
          vcvtqq2ps {rz-sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvttpd2dq {sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa1,0xf9,0x18,0xe6,0xf7]
          vcvttpd2dq {sae}, %ymm23, %xmm22

// CHECK: vcvttpd2dq {sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xf9,0x1f,0xe6,0xf7]
          vcvttpd2dq {sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvttpd2dq {sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xf9,0x9f,0xe6,0xf7]
          vcvttpd2dq {sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvttpd2qq {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0xf9,0x18,0x7a,0xf7]
          vcvttpd2qq {sae}, %ymm23, %ymm22

// CHECK: vcvttpd2qq {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xf9,0x1f,0x7a,0xf7]
          vcvttpd2qq {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvttpd2qq {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xf9,0x9f,0x7a,0xf7]
          vcvttpd2qq {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvttpd2udq {sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa1,0xf8,0x18,0x78,0xf7]
          vcvttpd2udq {sae}, %ymm23, %xmm22

// CHECK: vcvttpd2udq {sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xf8,0x1f,0x78,0xf7]
          vcvttpd2udq {sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvttpd2udq {sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xf8,0x9f,0x78,0xf7]
          vcvttpd2udq {sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvttpd2uqq {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0xf9,0x18,0x78,0xf7]
          vcvttpd2uqq {sae}, %ymm23, %ymm22

// CHECK: vcvttpd2uqq {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xf9,0x1f,0x78,0xf7]
          vcvttpd2uqq {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvttpd2uqq {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xf9,0x9f,0x78,0xf7]
          vcvttpd2uqq {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvttph2dq {sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x7a,0x18,0x5b,0xf7]
          vcvttph2dq {sae}, %xmm23, %ymm22

// CHECK: vcvttph2dq {sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7a,0x1f,0x5b,0xf7]
          vcvttph2dq {sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvttph2dq {sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7a,0x9f,0x5b,0xf7]
          vcvttph2dq {sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvttph2qq {sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x7a,0xf7]
          vcvttph2qq {sae}, %xmm23, %ymm22

// CHECK: vcvttph2qq {sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x79,0x1f,0x7a,0xf7]
          vcvttph2qq {sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvttph2qq {sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x79,0x9f,0x7a,0xf7]
          vcvttph2qq {sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvttph2udq {sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x78,0x18,0x78,0xf7]
          vcvttph2udq {sae}, %xmm23, %ymm22

// CHECK: vcvttph2udq {sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x78,0x1f,0x78,0xf7]
          vcvttph2udq {sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvttph2udq {sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x78,0x9f,0x78,0xf7]
          vcvttph2udq {sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvttph2uqq {sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x78,0xf7]
          vcvttph2uqq {sae}, %xmm23, %ymm22

// CHECK: vcvttph2uqq {sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x79,0x1f,0x78,0xf7]
          vcvttph2uqq {sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvttph2uqq {sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x79,0x9f,0x78,0xf7]
          vcvttph2uqq {sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvttph2uw {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x78,0x18,0x7c,0xf7]
          vcvttph2uw {sae}, %ymm23, %ymm22

// CHECK: vcvttph2uw {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x78,0x1f,0x7c,0xf7]
          vcvttph2uw {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvttph2uw {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x78,0x9f,0x7c,0xf7]
          vcvttph2uw {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvttph2w {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x7c,0xf7]
          vcvttph2w {sae}, %ymm23, %ymm22

// CHECK: vcvttph2w {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x79,0x1f,0x7c,0xf7]
          vcvttph2w {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvttph2w {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x79,0x9f,0x7c,0xf7]
          vcvttph2w {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvttps2dq {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0x7a,0x18,0x5b,0xf7]
          vcvttps2dq {sae}, %ymm23, %ymm22

// CHECK: vcvttps2dq {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0x7a,0x1f,0x5b,0xf7]
          vcvttps2dq {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvttps2dq {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0x7a,0x9f,0x5b,0xf7]
          vcvttps2dq {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvttps2qq {sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0x79,0x18,0x7a,0xf7]
          vcvttps2qq {sae}, %xmm23, %ymm22

// CHECK: vcvttps2qq {sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0x79,0x1f,0x7a,0xf7]
          vcvttps2qq {sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvttps2qq {sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0x79,0x9f,0x7a,0xf7]
          vcvttps2qq {sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvttps2udq {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0x78,0x18,0x78,0xf7]
          vcvttps2udq {sae}, %ymm23, %ymm22

// CHECK: vcvttps2udq {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0x78,0x1f,0x78,0xf7]
          vcvttps2udq {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvttps2udq {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0x78,0x9f,0x78,0xf7]
          vcvttps2udq {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvttps2uqq {sae}, %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0x79,0x18,0x78,0xf7]
          vcvttps2uqq {sae}, %xmm23, %ymm22

// CHECK: vcvttps2uqq {sae}, %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0x79,0x1f,0x78,0xf7]
          vcvttps2uqq {sae}, %xmm23, %ymm22 {%k7}

// CHECK: vcvttps2uqq {sae}, %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0x79,0x9f,0x78,0xf7]
          vcvttps2uqq {sae}, %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvtudq2ph {rn-sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7b,0x18,0x7a,0xf7]
          vcvtudq2ph {rn-sae}, %ymm23, %xmm22

// CHECK: vcvtudq2ph {rd-sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7b,0x3f,0x7a,0xf7]
          vcvtudq2ph {rd-sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvtudq2ph {rz-sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7b,0xff,0x7a,0xf7]
          vcvtudq2ph {rz-sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtudq2ps {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0x7b,0x18,0x7a,0xf7]
          vcvtudq2ps {rn-sae}, %ymm23, %ymm22

// CHECK: vcvtudq2ps {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0x7b,0x3f,0x7a,0xf7]
          vcvtudq2ps {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvtudq2ps {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0x7b,0xff,0x7a,0xf7]
          vcvtudq2ps {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvtuqq2pd {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0xfa,0x18,0x7a,0xf7]
          vcvtuqq2pd {rn-sae}, %ymm23, %ymm22

// CHECK: vcvtuqq2pd {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xfa,0x3f,0x7a,0xf7]
          vcvtuqq2pd {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvtuqq2pd {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xfa,0xff,0x7a,0xf7]
          vcvtuqq2pd {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvtuqq2ph {rn-sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0xfb,0x18,0x7a,0xf7]
          vcvtuqq2ph {rn-sae}, %ymm23, %xmm22

// CHECK: vcvtuqq2ph {rd-sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0xfb,0x3f,0x7a,0xf7]
          vcvtuqq2ph {rd-sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvtuqq2ph {rz-sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0xfb,0xff,0x7a,0xf7]
          vcvtuqq2ph {rz-sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtuqq2ps {rn-sae}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa1,0xfb,0x18,0x7a,0xf7]
          vcvtuqq2ps {rn-sae}, %ymm23, %xmm22

// CHECK: vcvtuqq2ps {rd-sae}, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xfb,0x3f,0x7a,0xf7]
          vcvtuqq2ps {rd-sae}, %ymm23, %xmm22 {%k7}

// CHECK: vcvtuqq2ps {rz-sae}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xfb,0xff,0x7a,0xf7]
          vcvtuqq2ps {rz-sae}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtuw2ph {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x7b,0x18,0x7d,0xf7]
          vcvtuw2ph {rn-sae}, %ymm23, %ymm22

// CHECK: vcvtuw2ph {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7b,0x3f,0x7d,0xf7]
          vcvtuw2ph {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvtuw2ph {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7b,0xff,0x7d,0xf7]
          vcvtuw2ph {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvtw2ph {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x7a,0x18,0x7d,0xf7]
          vcvtw2ph {rn-sae}, %ymm23, %ymm22

// CHECK: vcvtw2ph {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7a,0x3f,0x7d,0xf7]
          vcvtw2ph {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vcvtw2ph {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7a,0xff,0x7d,0xf7]
          vcvtw2ph {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vdivpd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x81,0xc1,0x10,0x5e,0xf0]
          vdivpd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vdivpd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x81,0xc1,0x37,0x5e,0xf0]
          vdivpd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vdivpd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x81,0xc1,0xf7,0x5e,0xf0]
          vdivpd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vdivph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x40,0x10,0x5e,0xf0]
          vdivph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vdivph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x40,0x37,0x5e,0xf0]
          vdivph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vdivph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x40,0xf7,0x5e,0xf0]
          vdivph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vdivps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x81,0x40,0x10,0x5e,0xf0]
          vdivps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vdivps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x81,0x40,0x37,0x5e,0xf0]
          vdivps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vdivps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x81,0x40,0xf7,0x5e,0xf0]
          vdivps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfcmaddcph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x43,0x10,0x56,0xf0]
          vfcmaddcph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfcmaddcph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x43,0x37,0x56,0xf0]
          vfcmaddcph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfcmaddcph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x43,0xf7,0x56,0xf0]
          vfcmaddcph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfcmulcph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x43,0x10,0xd6,0xf0]
          vfcmulcph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfcmulcph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x43,0x37,0xd6,0xf0]
          vfcmulcph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfcmulcph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x43,0xf7,0xd6,0xf0]
          vfcmulcph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfixupimmpd $123, {sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x83,0xc1,0x10,0x54,0xf0,0x7b]
          vfixupimmpd $123, {sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfixupimmpd $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x83,0xc1,0x17,0x54,0xf0,0x7b]
          vfixupimmpd $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfixupimmpd $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0xc1,0x97,0x54,0xf0,0x7b]
          vfixupimmpd $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfixupimmps $123, {sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x83,0x41,0x10,0x54,0xf0,0x7b]
          vfixupimmps $123, {sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfixupimmps $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x41,0x17,0x54,0xf0,0x7b]
          vfixupimmps $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfixupimmps $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x41,0x97,0x54,0xf0,0x7b]
          vfixupimmps $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd132pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0x98,0xf0]
          vfmadd132pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmadd132pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0x98,0xf0]
          vfmadd132pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd132pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0x98,0xf0]
          vfmadd132pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd132ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0x98,0xf0]
          vfmadd132ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmadd132ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0x98,0xf0]
          vfmadd132ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd132ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0x98,0xf0]
          vfmadd132ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd132ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0x98,0xf0]
          vfmadd132ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmadd132ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0x98,0xf0]
          vfmadd132ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd132ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0x98,0xf0]
          vfmadd132ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd213pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0xa8,0xf0]
          vfmadd213pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmadd213pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0xa8,0xf0]
          vfmadd213pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd213pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0xa8,0xf0]
          vfmadd213pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd213ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0xa8,0xf0]
          vfmadd213ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmadd213ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0xa8,0xf0]
          vfmadd213ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd213ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0xa8,0xf0]
          vfmadd213ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd213ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0xa8,0xf0]
          vfmadd213ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmadd213ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0xa8,0xf0]
          vfmadd213ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd213ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0xa8,0xf0]
          vfmadd213ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd231pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0xb8,0xf0]
          vfmadd231pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmadd231pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0xb8,0xf0]
          vfmadd231pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd231pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0xb8,0xf0]
          vfmadd231pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd231ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0xb8,0xf0]
          vfmadd231ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmadd231ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0xb8,0xf0]
          vfmadd231ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd231ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0xb8,0xf0]
          vfmadd231ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmadd231ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0xb8,0xf0]
          vfmadd231ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmadd231ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0xb8,0xf0]
          vfmadd231ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmadd231ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0xb8,0xf0]
          vfmadd231ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmaddcph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x42,0x10,0x56,0xf0]
          vfmaddcph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmaddcph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x42,0x37,0x56,0xf0]
          vfmaddcph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmaddcph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x42,0xf7,0x56,0xf0]
          vfmaddcph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmaddsub132pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0x96,0xf0]
          vfmaddsub132pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmaddsub132pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0x96,0xf0]
          vfmaddsub132pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmaddsub132pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0x96,0xf0]
          vfmaddsub132pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmaddsub132ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0x96,0xf0]
          vfmaddsub132ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmaddsub132ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0x96,0xf0]
          vfmaddsub132ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmaddsub132ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0x96,0xf0]
          vfmaddsub132ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmaddsub132ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0x96,0xf0]
          vfmaddsub132ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmaddsub132ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0x96,0xf0]
          vfmaddsub132ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmaddsub132ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0x96,0xf0]
          vfmaddsub132ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmaddsub213pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0xa6,0xf0]
          vfmaddsub213pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmaddsub213pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0xa6,0xf0]
          vfmaddsub213pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmaddsub213pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0xa6,0xf0]
          vfmaddsub213pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmaddsub213ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0xa6,0xf0]
          vfmaddsub213ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmaddsub213ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0xa6,0xf0]
          vfmaddsub213ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmaddsub213ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0xa6,0xf0]
          vfmaddsub213ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmaddsub213ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0xa6,0xf0]
          vfmaddsub213ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmaddsub213ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0xa6,0xf0]
          vfmaddsub213ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmaddsub213ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0xa6,0xf0]
          vfmaddsub213ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmaddsub231pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0xb6,0xf0]
          vfmaddsub231pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmaddsub231pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0xb6,0xf0]
          vfmaddsub231pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmaddsub231pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0xb6,0xf0]
          vfmaddsub231pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmaddsub231ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0xb6,0xf0]
          vfmaddsub231ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmaddsub231ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0xb6,0xf0]
          vfmaddsub231ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmaddsub231ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0xb6,0xf0]
          vfmaddsub231ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmaddsub231ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0xb6,0xf0]
          vfmaddsub231ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmaddsub231ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0xb6,0xf0]
          vfmaddsub231ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmaddsub231ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0xb6,0xf0]
          vfmaddsub231ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub132pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0x9a,0xf0]
          vfmsub132pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsub132pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0x9a,0xf0]
          vfmsub132pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub132pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0x9a,0xf0]
          vfmsub132pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub132ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0x9a,0xf0]
          vfmsub132ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsub132ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0x9a,0xf0]
          vfmsub132ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub132ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0x9a,0xf0]
          vfmsub132ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub132ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0x9a,0xf0]
          vfmsub132ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsub132ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0x9a,0xf0]
          vfmsub132ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub132ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0x9a,0xf0]
          vfmsub132ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub213pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0xaa,0xf0]
          vfmsub213pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsub213pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0xaa,0xf0]
          vfmsub213pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub213pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0xaa,0xf0]
          vfmsub213pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub213ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0xaa,0xf0]
          vfmsub213ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsub213ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0xaa,0xf0]
          vfmsub213ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub213ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0xaa,0xf0]
          vfmsub213ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub213ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0xaa,0xf0]
          vfmsub213ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsub213ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0xaa,0xf0]
          vfmsub213ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub213ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0xaa,0xf0]
          vfmsub213ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub231pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0xba,0xf0]
          vfmsub231pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsub231pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0xba,0xf0]
          vfmsub231pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub231pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0xba,0xf0]
          vfmsub231pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub231ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0xba,0xf0]
          vfmsub231ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsub231ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0xba,0xf0]
          vfmsub231ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub231ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0xba,0xf0]
          vfmsub231ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsub231ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0xba,0xf0]
          vfmsub231ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsub231ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0xba,0xf0]
          vfmsub231ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsub231ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0xba,0xf0]
          vfmsub231ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsubadd132pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0x97,0xf0]
          vfmsubadd132pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsubadd132pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0x97,0xf0]
          vfmsubadd132pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsubadd132pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0x97,0xf0]
          vfmsubadd132pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsubadd132ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0x97,0xf0]
          vfmsubadd132ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsubadd132ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0x97,0xf0]
          vfmsubadd132ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsubadd132ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0x97,0xf0]
          vfmsubadd132ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsubadd132ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0x97,0xf0]
          vfmsubadd132ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsubadd132ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0x97,0xf0]
          vfmsubadd132ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsubadd132ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0x97,0xf0]
          vfmsubadd132ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsubadd213pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0xa7,0xf0]
          vfmsubadd213pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsubadd213pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0xa7,0xf0]
          vfmsubadd213pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsubadd213pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0xa7,0xf0]
          vfmsubadd213pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsubadd213ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0xa7,0xf0]
          vfmsubadd213ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsubadd213ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0xa7,0xf0]
          vfmsubadd213ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsubadd213ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0xa7,0xf0]
          vfmsubadd213ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsubadd213ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0xa7,0xf0]
          vfmsubadd213ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsubadd213ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0xa7,0xf0]
          vfmsubadd213ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsubadd213ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0xa7,0xf0]
          vfmsubadd213ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsubadd231pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0xb7,0xf0]
          vfmsubadd231pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsubadd231pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0xb7,0xf0]
          vfmsubadd231pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsubadd231pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0xb7,0xf0]
          vfmsubadd231pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsubadd231ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0xb7,0xf0]
          vfmsubadd231ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsubadd231ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0xb7,0xf0]
          vfmsubadd231ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsubadd231ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0xb7,0xf0]
          vfmsubadd231ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmsubadd231ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0xb7,0xf0]
          vfmsubadd231ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmsubadd231ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0xb7,0xf0]
          vfmsubadd231ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmsubadd231ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0xb7,0xf0]
          vfmsubadd231ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfmulcph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x42,0x10,0xd6,0xf0]
          vfmulcph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfmulcph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x42,0x37,0xd6,0xf0]
          vfmulcph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfmulcph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x42,0xf7,0xd6,0xf0]
          vfmulcph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd132pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0x9c,0xf0]
          vfnmadd132pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd132pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0x9c,0xf0]
          vfnmadd132pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd132pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0x9c,0xf0]
          vfnmadd132pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd132ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0x9c,0xf0]
          vfnmadd132ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd132ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0x9c,0xf0]
          vfnmadd132ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd132ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0x9c,0xf0]
          vfnmadd132ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd132ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0x9c,0xf0]
          vfnmadd132ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd132ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0x9c,0xf0]
          vfnmadd132ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd132ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0x9c,0xf0]
          vfnmadd132ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd213pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0xac,0xf0]
          vfnmadd213pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd213pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0xac,0xf0]
          vfnmadd213pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd213pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0xac,0xf0]
          vfnmadd213pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd213ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0xac,0xf0]
          vfnmadd213ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd213ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0xac,0xf0]
          vfnmadd213ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd213ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0xac,0xf0]
          vfnmadd213ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd213ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0xac,0xf0]
          vfnmadd213ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd213ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0xac,0xf0]
          vfnmadd213ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd213ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0xac,0xf0]
          vfnmadd213ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd231pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0xbc,0xf0]
          vfnmadd231pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd231pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0xbc,0xf0]
          vfnmadd231pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd231pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0xbc,0xf0]
          vfnmadd231pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd231ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0xbc,0xf0]
          vfnmadd231ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd231ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0xbc,0xf0]
          vfnmadd231ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd231ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0xbc,0xf0]
          vfnmadd231ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmadd231ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0xbc,0xf0]
          vfnmadd231ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmadd231ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0xbc,0xf0]
          vfnmadd231ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmadd231ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0xbc,0xf0]
          vfnmadd231ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub132pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0x9e,0xf0]
          vfnmsub132pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub132pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0x9e,0xf0]
          vfnmsub132pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub132pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0x9e,0xf0]
          vfnmsub132pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub132ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0x9e,0xf0]
          vfnmsub132ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub132ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0x9e,0xf0]
          vfnmsub132ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub132ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0x9e,0xf0]
          vfnmsub132ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub132ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0x9e,0xf0]
          vfnmsub132ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub132ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0x9e,0xf0]
          vfnmsub132ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub132ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0x9e,0xf0]
          vfnmsub132ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub213pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0xae,0xf0]
          vfnmsub213pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub213pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0xae,0xf0]
          vfnmsub213pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub213pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0xae,0xf0]
          vfnmsub213pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub213ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0xae,0xf0]
          vfnmsub213ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub213ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0xae,0xf0]
          vfnmsub213ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub213ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0xae,0xf0]
          vfnmsub213ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub213ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0xae,0xf0]
          vfnmsub213ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub213ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0xae,0xf0]
          vfnmsub213ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub213ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0xae,0xf0]
          vfnmsub213ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub231pd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0xbe,0xf0]
          vfnmsub231pd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub231pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0xbe,0xf0]
          vfnmsub231pd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub231pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0xbe,0xf0]
          vfnmsub231pd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub231ph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0xbe,0xf0]
          vfnmsub231ph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub231ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0xbe,0xf0]
          vfnmsub231ph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub231ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0xbe,0xf0]
          vfnmsub231ph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vfnmsub231ps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0xbe,0xf0]
          vfnmsub231ps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vfnmsub231ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0xbe,0xf0]
          vfnmsub231ps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vfnmsub231ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0xbe,0xf0]
          vfnmsub231ps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vgetexppd {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0xf9,0x18,0x42,0xf7]
          vgetexppd {sae}, %ymm23, %ymm22

// CHECK: vgetexppd {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa2,0xf9,0x1f,0x42,0xf7]
          vgetexppd {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vgetexppd {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa2,0xf9,0x9f,0x42,0xf7]
          vgetexppd {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vgetexpph {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa6,0x79,0x18,0x42,0xf7]
          vgetexpph {sae}, %ymm23, %ymm22

// CHECK: vgetexpph {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa6,0x79,0x1f,0x42,0xf7]
          vgetexpph {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vgetexpph {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa6,0x79,0x9f,0x42,0xf7]
          vgetexpph {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vgetexpps {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x79,0x18,0x42,0xf7]
          vgetexpps {sae}, %ymm23, %ymm22

// CHECK: vgetexpps {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa2,0x79,0x1f,0x42,0xf7]
          vgetexpps {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vgetexpps {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa2,0x79,0x9f,0x42,0xf7]
          vgetexpps {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vgetmantpd $123, {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0xf9,0x18,0x26,0xf7,0x7b]
          vgetmantpd $123, {sae}, %ymm23, %ymm22

// CHECK: vgetmantpd $123, {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0xf9,0x1f,0x26,0xf7,0x7b]
          vgetmantpd $123, {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vgetmantpd $123, {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0xf9,0x9f,0x26,0xf7,0x7b]
          vgetmantpd $123, {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vgetmantph $123, {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x78,0x18,0x26,0xf7,0x7b]
          vgetmantph $123, {sae}, %ymm23, %ymm22

// CHECK: vgetmantph $123, {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x78,0x1f,0x26,0xf7,0x7b]
          vgetmantph $123, {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vgetmantph $123, {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x78,0x9f,0x26,0xf7,0x7b]
          vgetmantph $123, {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vgetmantps $123, {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x79,0x18,0x26,0xf7,0x7b]
          vgetmantps $123, {sae}, %ymm23, %ymm22

// CHECK: vgetmantps $123, {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x79,0x1f,0x26,0xf7,0x7b]
          vgetmantps $123, {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vgetmantps $123, {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x79,0x9f,0x26,0xf7,0x7b]
          vgetmantps $123, {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmaxpd {sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x81,0xc1,0x10,0x5f,0xf0]
          vmaxpd {sae}, %ymm24, %ymm23, %ymm22

// CHECK: vmaxpd {sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x81,0xc1,0x17,0x5f,0xf0]
          vmaxpd {sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vmaxpd {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x81,0xc1,0x97,0x5f,0xf0]
          vmaxpd {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmaxph {sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x40,0x10,0x5f,0xf0]
          vmaxph {sae}, %ymm24, %ymm23, %ymm22

// CHECK: vmaxph {sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x40,0x17,0x5f,0xf0]
          vmaxph {sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vmaxph {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x40,0x97,0x5f,0xf0]
          vmaxph {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmaxps {sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x81,0x40,0x10,0x5f,0xf0]
          vmaxps {sae}, %ymm24, %ymm23, %ymm22

// CHECK: vmaxps {sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x81,0x40,0x17,0x5f,0xf0]
          vmaxps {sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vmaxps {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x81,0x40,0x97,0x5f,0xf0]
          vmaxps {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminpd {sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x81,0xc1,0x10,0x5d,0xf0]
          vminpd {sae}, %ymm24, %ymm23, %ymm22

// CHECK: vminpd {sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x81,0xc1,0x17,0x5d,0xf0]
          vminpd {sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vminpd {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x81,0xc1,0x97,0x5d,0xf0]
          vminpd {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminph {sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x40,0x10,0x5d,0xf0]
          vminph {sae}, %ymm24, %ymm23, %ymm22

// CHECK: vminph {sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x40,0x17,0x5d,0xf0]
          vminph {sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vminph {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x40,0x97,0x5d,0xf0]
          vminph {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vminps {sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x81,0x40,0x10,0x5d,0xf0]
          vminps {sae}, %ymm24, %ymm23, %ymm22

// CHECK: vminps {sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x81,0x40,0x17,0x5d,0xf0]
          vminps {sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vminps {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x81,0x40,0x97,0x5d,0xf0]
          vminps {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmulpd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x81,0xc1,0x10,0x59,0xf0]
          vmulpd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vmulpd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x81,0xc1,0x37,0x59,0xf0]
          vmulpd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vmulpd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x81,0xc1,0xf7,0x59,0xf0]
          vmulpd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmulph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x40,0x10,0x59,0xf0]
          vmulph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vmulph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x40,0x37,0x59,0xf0]
          vmulph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vmulph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x40,0xf7,0x59,0xf0]
          vmulph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vmulps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x81,0x40,0x10,0x59,0xf0]
          vmulps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vmulps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x81,0x40,0x37,0x59,0xf0]
          vmulps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vmulps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x81,0x40,0xf7,0x59,0xf0]
          vmulps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vrangepd $123, {sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x83,0xc1,0x10,0x50,0xf0,0x7b]
          vrangepd $123, {sae}, %ymm24, %ymm23, %ymm22

// CHECK: vrangepd $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x83,0xc1,0x17,0x50,0xf0,0x7b]
          vrangepd $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vrangepd $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0xc1,0x97,0x50,0xf0,0x7b]
          vrangepd $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vrangeps $123, {sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x83,0x41,0x10,0x50,0xf0,0x7b]
          vrangeps $123, {sae}, %ymm24, %ymm23, %ymm22

// CHECK: vrangeps $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x83,0x41,0x17,0x50,0xf0,0x7b]
          vrangeps $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vrangeps $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x83,0x41,0x97,0x50,0xf0,0x7b]
          vrangeps $123, {sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vreducepd $123, {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0xf9,0x18,0x56,0xf7,0x7b]
          vreducepd $123, {sae}, %ymm23, %ymm22

// CHECK: vreducepd $123, {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0xf9,0x1f,0x56,0xf7,0x7b]
          vreducepd $123, {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vreducepd $123, {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0xf9,0x9f,0x56,0xf7,0x7b]
          vreducepd $123, {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vreduceph $123, {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x78,0x18,0x56,0xf7,0x7b]
          vreduceph $123, {sae}, %ymm23, %ymm22

// CHECK: vreduceph $123, {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x78,0x1f,0x56,0xf7,0x7b]
          vreduceph $123, {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vreduceph $123, {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x78,0x9f,0x56,0xf7,0x7b]
          vreduceph $123, {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vreduceps $123, {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x79,0x18,0x56,0xf7,0x7b]
          vreduceps $123, {sae}, %ymm23, %ymm22

// CHECK: vreduceps $123, {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x79,0x1f,0x56,0xf7,0x7b]
          vreduceps $123, {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vreduceps $123, {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x79,0x9f,0x56,0xf7,0x7b]
          vreduceps $123, {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vrndscalepd $123, {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0xf9,0x18,0x09,0xf7,0x7b]
          vrndscalepd $123, {sae}, %ymm23, %ymm22

// CHECK: vrndscalepd $123, {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0xf9,0x1f,0x09,0xf7,0x7b]
          vrndscalepd $123, {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vrndscalepd $123, {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0xf9,0x9f,0x09,0xf7,0x7b]
          vrndscalepd $123, {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vrndscaleph $123, {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x78,0x18,0x08,0xf7,0x7b]
          vrndscaleph $123, {sae}, %ymm23, %ymm22

// CHECK: vrndscaleph $123, {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x78,0x1f,0x08,0xf7,0x7b]
          vrndscaleph $123, {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vrndscaleph $123, {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x78,0x9f,0x08,0xf7,0x7b]
          vrndscaleph $123, {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vrndscaleps $123, {sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa3,0x79,0x18,0x08,0xf7,0x7b]
          vrndscaleps $123, {sae}, %ymm23, %ymm22

// CHECK: vrndscaleps $123, {sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa3,0x79,0x1f,0x08,0xf7,0x7b]
          vrndscaleps $123, {sae}, %ymm23, %ymm22 {%k7}

// CHECK: vrndscaleps $123, {sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa3,0x79,0x9f,0x08,0xf7,0x7b]
          vrndscaleps $123, {sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vscalefpd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0xc1,0x10,0x2c,0xf0]
          vscalefpd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vscalefpd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0xc1,0x37,0x2c,0xf0]
          vscalefpd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vscalefpd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0xc1,0xf7,0x2c,0xf0]
          vscalefpd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vscalefph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x86,0x41,0x10,0x2c,0xf0]
          vscalefph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vscalefph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x86,0x41,0x37,0x2c,0xf0]
          vscalefph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vscalefph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x86,0x41,0xf7,0x2c,0xf0]
          vscalefph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vscalefps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0x2c,0xf0]
          vscalefps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vscalefps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x41,0x37,0x2c,0xf0]
          vscalefps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vscalefps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0x2c,0xf0]
          vscalefps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vsqrtpd {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0xf9,0x18,0x51,0xf7]
          vsqrtpd {rn-sae}, %ymm23, %ymm22

// CHECK: vsqrtpd {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0xf9,0x3f,0x51,0xf7]
          vsqrtpd {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vsqrtpd {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0xf9,0xff,0x51,0xf7]
          vsqrtpd {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vsqrtph {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x78,0x18,0x51,0xf7]
          vsqrtph {rn-sae}, %ymm23, %ymm22

// CHECK: vsqrtph {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x78,0x3f,0x51,0xf7]
          vsqrtph {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vsqrtph {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x78,0xff,0x51,0xf7]
          vsqrtph {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vsqrtps {rn-sae}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa1,0x78,0x18,0x51,0xf7]
          vsqrtps {rn-sae}, %ymm23, %ymm22

// CHECK: vsqrtps {rd-sae}, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa1,0x78,0x3f,0x51,0xf7]
          vsqrtps {rd-sae}, %ymm23, %ymm22 {%k7}

// CHECK: vsqrtps {rz-sae}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa1,0x78,0xff,0x51,0xf7]
          vsqrtps {rz-sae}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vsubpd {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x81,0xc1,0x10,0x5c,0xf0]
          vsubpd {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vsubpd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x81,0xc1,0x37,0x5c,0xf0]
          vsubpd {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vsubpd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x81,0xc1,0xf7,0x5c,0xf0]
          vsubpd {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vsubph {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x40,0x10,0x5c,0xf0]
          vsubph {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vsubph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x40,0x37,0x5c,0xf0]
          vsubph {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vsubph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x40,0xf7,0x5c,0xf0]
          vsubph {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vsubps {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x81,0x40,0x10,0x5c,0xf0]
          vsubps {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vsubps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x81,0x40,0x37,0x5c,0xf0]
          vsubps {rd-sae}, %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vsubps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x81,0x40,0xf7,0x5c,0xf0]
          vsubps {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
