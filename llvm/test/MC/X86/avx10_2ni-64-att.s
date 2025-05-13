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
