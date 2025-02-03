// RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s

// CHECK: vcvt2ps2phx %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x45,0x20,0x67,0xf0]
          vcvt2ps2phx %ymm24, %ymm23, %ymm22

// CHECK: vcvt2ps2phx {rn-sae}, %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x41,0x10,0x67,0xf0]
          vcvt2ps2phx {rn-sae}, %ymm24, %ymm23, %ymm22

// CHECK: vcvt2ps2phx %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x45,0x27,0x67,0xf0]
          vcvt2ps2phx %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vcvt2ps2phx {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0x67,0xf0]
          vcvt2ps2phx {rz-sae}, %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ps2phx %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x45,0x40,0x67,0xf0]
          vcvt2ps2phx %zmm24, %zmm23, %zmm22

// CHECK: vcvt2ps2phx {rn-sae}, %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x45,0x10,0x67,0xf0]
          vcvt2ps2phx {rn-sae}, %zmm24, %zmm23, %zmm22

// CHECK: vcvt2ps2phx %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x45,0x47,0x67,0xf0]
          vcvt2ps2phx %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vcvt2ps2phx {rz-sae}, %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x45,0xf7,0x67,0xf0]
          vcvt2ps2phx {rz-sae}, %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ps2phx %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x45,0x00,0x67,0xf0]
          vcvt2ps2phx %xmm24, %xmm23, %xmm22

// CHECK: vcvt2ps2phx %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x45,0x07,0x67,0xf0]
          vcvt2ps2phx %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vcvt2ps2phx %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x45,0x87,0x67,0xf0]
          vcvt2ps2phx %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvt2ps2phx  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x45,0x40,0x67,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ps2phx  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vcvt2ps2phx  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x45,0x47,0x67,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ps2phx  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vcvt2ps2phx  (%rip){1to16}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x45,0x50,0x67,0x35,0x00,0x00,0x00,0x00]
          vcvt2ps2phx  (%rip){1to16}, %zmm23, %zmm22

// CHECK: vcvt2ps2phx  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x45,0x40,0x67,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ps2phx  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vcvt2ps2phx  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0xc7,0x67,0x71,0x7f]
          vcvt2ps2phx  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ps2phx  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0xd7,0x67,0x72,0x80]
          vcvt2ps2phx  -512(%rdx){1to16}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ps2phx  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x45,0x20,0x67,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ps2phx  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vcvt2ps2phx  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x45,0x27,0x67,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ps2phx  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vcvt2ps2phx  (%rip){1to8}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x45,0x30,0x67,0x35,0x00,0x00,0x00,0x00]
          vcvt2ps2phx  (%rip){1to8}, %ymm23, %ymm22

// CHECK: vcvt2ps2phx  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x45,0x20,0x67,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ps2phx  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vcvt2ps2phx  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0xa7,0x67,0x71,0x7f]
          vcvt2ps2phx  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ps2phx  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0xb7,0x67,0x72,0x80]
          vcvt2ps2phx  -512(%rdx){1to8}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ps2phx  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x45,0x00,0x67,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ps2phx  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vcvt2ps2phx  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x45,0x07,0x67,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ps2phx  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vcvt2ps2phx  (%rip){1to4}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x45,0x10,0x67,0x35,0x00,0x00,0x00,0x00]
          vcvt2ps2phx  (%rip){1to4}, %xmm23, %xmm22

// CHECK: vcvt2ps2phx  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x45,0x00,0x67,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ps2phx  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vcvt2ps2phx  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0x87,0x67,0x71,0x7f]
          vcvt2ps2phx  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvt2ps2phx  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x45,0x97,0x67,0x72,0x80]
          vcvt2ps2phx  -512(%rdx){1to4}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8 %zmm24, %zmm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x44,0x40,0x74,0xf0]
          vcvtbiasph2bf8 %zmm24, %zmm23, %ymm22

// CHECK: vcvtbiasph2bf8 %zmm24, %zmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x47,0x74,0xf0]
          vcvtbiasph2bf8 %zmm24, %zmm23, %ymm22 {%k7}

// CHECK: vcvtbiasph2bf8 %zmm24, %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0xc7,0x74,0xf0]
          vcvtbiasph2bf8 %zmm24, %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x44,0x00,0x74,0xf0]
          vcvtbiasph2bf8 %xmm24, %xmm23, %xmm22

// CHECK: vcvtbiasph2bf8 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x07,0x74,0xf0]
          vcvtbiasph2bf8 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2bf8 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0x87,0x74,0xf0]
          vcvtbiasph2bf8 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8 %ymm24, %ymm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x44,0x20,0x74,0xf0]
          vcvtbiasph2bf8 %ymm24, %ymm23, %xmm22

// CHECK: vcvtbiasph2bf8 %ymm24, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x44,0x27,0x74,0xf0]
          vcvtbiasph2bf8 %ymm24, %ymm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2bf8 %ymm24, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x44,0xa7,0x74,0xf0]
          vcvtbiasph2bf8 %ymm24, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8  268435456(%rbp,%r14,8), %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x44,0x20,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8  268435456(%rbp,%r14,8), %ymm23, %xmm22

// CHECK: vcvtbiasph2bf8  291(%r8,%rax,4), %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x27,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8  291(%r8,%rax,4), %ymm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2bf8  (%rip){1to16}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x30,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2bf8  (%rip){1to16}, %ymm23, %xmm22

// CHECK: vcvtbiasph2bf8  -1024(,%rbp,2), %ymm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x20,0x74,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2bf8  -1024(,%rbp,2), %ymm23, %xmm22

// CHECK: vcvtbiasph2bf8  4064(%rcx), %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xa7,0x74,0x71,0x7f]
          vcvtbiasph2bf8  4064(%rcx), %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8  -256(%rdx){1to16}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xb7,0x74,0x72,0x80]
          vcvtbiasph2bf8  -256(%rdx){1to16}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8  268435456(%rbp,%r14,8), %zmm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x44,0x40,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8  268435456(%rbp,%r14,8), %zmm23, %ymm22

// CHECK: vcvtbiasph2bf8  291(%r8,%rax,4), %zmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x47,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8  291(%r8,%rax,4), %zmm23, %ymm22 {%k7}

// CHECK: vcvtbiasph2bf8  (%rip){1to32}, %zmm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x44,0x50,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2bf8  (%rip){1to32}, %zmm23, %ymm22

// CHECK: vcvtbiasph2bf8  -2048(,%rbp,2), %zmm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x44,0x40,0x74,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2bf8  -2048(,%rbp,2), %zmm23, %ymm22

// CHECK: vcvtbiasph2bf8  8128(%rcx), %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xc7,0x74,0x71,0x7f]
          vcvtbiasph2bf8  8128(%rcx), %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8  -256(%rdx){1to32}, %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0xd7,0x74,0x72,0x80]
          vcvtbiasph2bf8  -256(%rdx){1to32}, %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x44,0x00,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vcvtbiasph2bf8  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x44,0x07,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2bf8  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x10,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2bf8  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vcvtbiasph2bf8  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x44,0x00,0x74,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2bf8  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vcvtbiasph2bf8  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0x87,0x74,0x71,0x7f]
          vcvtbiasph2bf8  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x44,0x97,0x74,0x72,0x80]
          vcvtbiasph2bf8  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8s %zmm24, %zmm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x44,0x40,0x74,0xf0]
          vcvtbiasph2bf8s %zmm24, %zmm23, %ymm22

// CHECK: vcvtbiasph2bf8s %zmm24, %zmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x44,0x47,0x74,0xf0]
          vcvtbiasph2bf8s %zmm24, %zmm23, %ymm22 {%k7}

// CHECK: vcvtbiasph2bf8s %zmm24, %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x44,0xc7,0x74,0xf0]
          vcvtbiasph2bf8s %zmm24, %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8s %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x44,0x00,0x74,0xf0]
          vcvtbiasph2bf8s %xmm24, %xmm23, %xmm22

// CHECK: vcvtbiasph2bf8s %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x44,0x07,0x74,0xf0]
          vcvtbiasph2bf8s %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2bf8s %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x44,0x87,0x74,0xf0]
          vcvtbiasph2bf8s %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8s %ymm24, %ymm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x44,0x20,0x74,0xf0]
          vcvtbiasph2bf8s %ymm24, %ymm23, %xmm22

// CHECK: vcvtbiasph2bf8s %ymm24, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x44,0x27,0x74,0xf0]
          vcvtbiasph2bf8s %ymm24, %ymm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2bf8s %ymm24, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x44,0xa7,0x74,0xf0]
          vcvtbiasph2bf8s %ymm24, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8s  268435456(%rbp,%r14,8), %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x44,0x20,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8s  268435456(%rbp,%r14,8), %ymm23, %xmm22

// CHECK: vcvtbiasph2bf8s  291(%r8,%rax,4), %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x44,0x27,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8s  291(%r8,%rax,4), %ymm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2bf8s  (%rip){1to16}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x44,0x30,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2bf8s  (%rip){1to16}, %ymm23, %xmm22

// CHECK: vcvtbiasph2bf8s  -1024(,%rbp,2), %ymm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x44,0x20,0x74,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2bf8s  -1024(,%rbp,2), %ymm23, %xmm22

// CHECK: vcvtbiasph2bf8s  4064(%rcx), %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0xa7,0x74,0x71,0x7f]
          vcvtbiasph2bf8s  4064(%rcx), %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8s  -256(%rdx){1to16}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0xb7,0x74,0x72,0x80]
          vcvtbiasph2bf8s  -256(%rdx){1to16}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8s  268435456(%rbp,%r14,8), %zmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x44,0x40,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8s  268435456(%rbp,%r14,8), %zmm23, %ymm22

// CHECK: vcvtbiasph2bf8s  291(%r8,%rax,4), %zmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x44,0x47,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8s  291(%r8,%rax,4), %zmm23, %ymm22 {%k7}

// CHECK: vcvtbiasph2bf8s  (%rip){1to32}, %zmm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x44,0x50,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2bf8s  (%rip){1to32}, %zmm23, %ymm22

// CHECK: vcvtbiasph2bf8s  -2048(,%rbp,2), %zmm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x44,0x40,0x74,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2bf8s  -2048(,%rbp,2), %zmm23, %ymm22

// CHECK: vcvtbiasph2bf8s  8128(%rcx), %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0xc7,0x74,0x71,0x7f]
          vcvtbiasph2bf8s  8128(%rcx), %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8s  -256(%rdx){1to32}, %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0xd7,0x74,0x72,0x80]
          vcvtbiasph2bf8s  -256(%rdx){1to32}, %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8s  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x44,0x00,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8s  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vcvtbiasph2bf8s  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x44,0x07,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8s  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2bf8s  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x44,0x10,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2bf8s  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vcvtbiasph2bf8s  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x44,0x00,0x74,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2bf8s  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vcvtbiasph2bf8s  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0x87,0x74,0x71,0x7f]
          vcvtbiasph2bf8s  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2bf8s  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0x97,0x74,0x72,0x80]
          vcvtbiasph2bf8s  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8 %zmm24, %zmm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x44,0x40,0x18,0xf0]
          vcvtbiasph2hf8 %zmm24, %zmm23, %ymm22

// CHECK: vcvtbiasph2hf8 %zmm24, %zmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x44,0x47,0x18,0xf0]
          vcvtbiasph2hf8 %zmm24, %zmm23, %ymm22 {%k7}

// CHECK: vcvtbiasph2hf8 %zmm24, %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x44,0xc7,0x18,0xf0]
          vcvtbiasph2hf8 %zmm24, %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x44,0x00,0x18,0xf0]
          vcvtbiasph2hf8 %xmm24, %xmm23, %xmm22

// CHECK: vcvtbiasph2hf8 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x44,0x07,0x18,0xf0]
          vcvtbiasph2hf8 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2hf8 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x44,0x87,0x18,0xf0]
          vcvtbiasph2hf8 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8 %ymm24, %ymm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x44,0x20,0x18,0xf0]
          vcvtbiasph2hf8 %ymm24, %ymm23, %xmm22

// CHECK: vcvtbiasph2hf8 %ymm24, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x44,0x27,0x18,0xf0]
          vcvtbiasph2hf8 %ymm24, %ymm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2hf8 %ymm24, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x44,0xa7,0x18,0xf0]
          vcvtbiasph2hf8 %ymm24, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8  268435456(%rbp,%r14,8), %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x44,0x20,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8  268435456(%rbp,%r14,8), %ymm23, %xmm22

// CHECK: vcvtbiasph2hf8  291(%r8,%rax,4), %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x44,0x27,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8  291(%r8,%rax,4), %ymm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2hf8  (%rip){1to16}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x44,0x30,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2hf8  (%rip){1to16}, %ymm23, %xmm22

// CHECK: vcvtbiasph2hf8  -1024(,%rbp,2), %ymm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x44,0x20,0x18,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2hf8  -1024(,%rbp,2), %ymm23, %xmm22

// CHECK: vcvtbiasph2hf8  4064(%rcx), %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0xa7,0x18,0x71,0x7f]
          vcvtbiasph2hf8  4064(%rcx), %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8  -256(%rdx){1to16}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0xb7,0x18,0x72,0x80]
          vcvtbiasph2hf8  -256(%rdx){1to16}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8  268435456(%rbp,%r14,8), %zmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x44,0x40,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8  268435456(%rbp,%r14,8), %zmm23, %ymm22

// CHECK: vcvtbiasph2hf8  291(%r8,%rax,4), %zmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x44,0x47,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8  291(%r8,%rax,4), %zmm23, %ymm22 {%k7}

// CHECK: vcvtbiasph2hf8  (%rip){1to32}, %zmm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x44,0x50,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2hf8  (%rip){1to32}, %zmm23, %ymm22

// CHECK: vcvtbiasph2hf8  -2048(,%rbp,2), %zmm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x44,0x40,0x18,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2hf8  -2048(,%rbp,2), %zmm23, %ymm22

// CHECK: vcvtbiasph2hf8  8128(%rcx), %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0xc7,0x18,0x71,0x7f]
          vcvtbiasph2hf8  8128(%rcx), %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8  -256(%rdx){1to32}, %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0xd7,0x18,0x72,0x80]
          vcvtbiasph2hf8  -256(%rdx){1to32}, %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x44,0x00,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vcvtbiasph2hf8  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x44,0x07,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2hf8  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x44,0x10,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2hf8  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vcvtbiasph2hf8  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x44,0x00,0x18,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2hf8  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vcvtbiasph2hf8  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0x87,0x18,0x71,0x7f]
          vcvtbiasph2hf8  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0x97,0x18,0x72,0x80]
          vcvtbiasph2hf8  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8s %zmm24, %zmm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x44,0x40,0x1b,0xf0]
          vcvtbiasph2hf8s %zmm24, %zmm23, %ymm22

// CHECK: vcvtbiasph2hf8s %zmm24, %zmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x44,0x47,0x1b,0xf0]
          vcvtbiasph2hf8s %zmm24, %zmm23, %ymm22 {%k7}

// CHECK: vcvtbiasph2hf8s %zmm24, %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x44,0xc7,0x1b,0xf0]
          vcvtbiasph2hf8s %zmm24, %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8s %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x44,0x00,0x1b,0xf0]
          vcvtbiasph2hf8s %xmm24, %xmm23, %xmm22

// CHECK: vcvtbiasph2hf8s %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x44,0x07,0x1b,0xf0]
          vcvtbiasph2hf8s %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2hf8s %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x44,0x87,0x1b,0xf0]
          vcvtbiasph2hf8s %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8s %ymm24, %ymm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x44,0x20,0x1b,0xf0]
          vcvtbiasph2hf8s %ymm24, %ymm23, %xmm22

// CHECK: vcvtbiasph2hf8s %ymm24, %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x44,0x27,0x1b,0xf0]
          vcvtbiasph2hf8s %ymm24, %ymm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2hf8s %ymm24, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x44,0xa7,0x1b,0xf0]
          vcvtbiasph2hf8s %ymm24, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8s  268435456(%rbp,%r14,8), %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x44,0x20,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8s  268435456(%rbp,%r14,8), %ymm23, %xmm22

// CHECK: vcvtbiasph2hf8s  291(%r8,%rax,4), %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x44,0x27,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8s  291(%r8,%rax,4), %ymm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2hf8s  (%rip){1to16}, %ymm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x44,0x30,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2hf8s  (%rip){1to16}, %ymm23, %xmm22

// CHECK: vcvtbiasph2hf8s  -1024(,%rbp,2), %ymm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x44,0x20,0x1b,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2hf8s  -1024(,%rbp,2), %ymm23, %xmm22

// CHECK: vcvtbiasph2hf8s  4064(%rcx), %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0xa7,0x1b,0x71,0x7f]
          vcvtbiasph2hf8s  4064(%rcx), %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8s  -256(%rdx){1to16}, %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0xb7,0x1b,0x72,0x80]
          vcvtbiasph2hf8s  -256(%rdx){1to16}, %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8s  268435456(%rbp,%r14,8), %zmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x44,0x40,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8s  268435456(%rbp,%r14,8), %zmm23, %ymm22

// CHECK: vcvtbiasph2hf8s  291(%r8,%rax,4), %zmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x44,0x47,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8s  291(%r8,%rax,4), %zmm23, %ymm22 {%k7}

// CHECK: vcvtbiasph2hf8s  (%rip){1to32}, %zmm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x44,0x50,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2hf8s  (%rip){1to32}, %zmm23, %ymm22

// CHECK: vcvtbiasph2hf8s  -2048(,%rbp,2), %zmm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x44,0x40,0x1b,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2hf8s  -2048(,%rbp,2), %zmm23, %ymm22

// CHECK: vcvtbiasph2hf8s  8128(%rcx), %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0xc7,0x1b,0x71,0x7f]
          vcvtbiasph2hf8s  8128(%rcx), %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8s  -256(%rdx){1to32}, %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0xd7,0x1b,0x72,0x80]
          vcvtbiasph2hf8s  -256(%rdx){1to32}, %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8s  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x44,0x00,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8s  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vcvtbiasph2hf8s  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x44,0x07,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8s  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vcvtbiasph2hf8s  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x44,0x10,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2hf8s  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vcvtbiasph2hf8s  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x44,0x00,0x1b,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2hf8s  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vcvtbiasph2hf8s  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0x87,0x1b,0x71,0x7f]
          vcvtbiasph2hf8s  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtbiasph2hf8s  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x44,0x97,0x1b,0x72,0x80]
          vcvtbiasph2hf8s  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvthf82ph %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7f,0x08,0x1e,0xf7]
          vcvthf82ph %xmm23, %xmm22

// CHECK: vcvthf82ph %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7f,0x0f,0x1e,0xf7]
          vcvthf82ph %xmm23, %xmm22 {%k7}

// CHECK: vcvthf82ph %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7f,0x8f,0x1e,0xf7]
          vcvthf82ph %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvthf82ph %xmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x7f,0x28,0x1e,0xf7]
          vcvthf82ph %xmm23, %ymm22

// CHECK: vcvthf82ph %xmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7f,0x2f,0x1e,0xf7]
          vcvthf82ph %xmm23, %ymm22 {%k7}

// CHECK: vcvthf82ph %xmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7f,0xaf,0x1e,0xf7]
          vcvthf82ph %xmm23, %ymm22 {%k7} {z}

// CHECK: vcvthf82ph %ymm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x7f,0x48,0x1e,0xf7]
          vcvthf82ph %ymm23, %zmm22

// CHECK: vcvthf82ph %ymm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7f,0x4f,0x1e,0xf7]
          vcvthf82ph %ymm23, %zmm22 {%k7}

// CHECK: vcvthf82ph %ymm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7f,0xcf,0x1e,0xf7]
          vcvthf82ph %ymm23, %zmm22 {%k7} {z}

// CHECK: vcvthf82ph  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa5,0x7f,0x08,0x1e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvthf82ph  268435456(%rbp,%r14,8), %xmm22

// CHECK: vcvthf82ph  291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7f,0x0f,0x1e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvthf82ph  291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vcvthf82ph  (%rip), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7f,0x08,0x1e,0x35,0x00,0x00,0x00,0x00]
          vcvthf82ph  (%rip), %xmm22

// CHECK: vcvthf82ph  -256(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7f,0x08,0x1e,0x34,0x6d,0x00,0xff,0xff,0xff]
          vcvthf82ph  -256(,%rbp,2), %xmm22

// CHECK: vcvthf82ph  1016(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7f,0x8f,0x1e,0x71,0x7f]
          vcvthf82ph  1016(%rcx), %xmm22 {%k7} {z}

// CHECK: vcvthf82ph  -1024(%rdx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7f,0x8f,0x1e,0x72,0x80]
          vcvthf82ph  -1024(%rdx), %xmm22 {%k7} {z}

// CHECK: vcvthf82ph  268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa5,0x7f,0x28,0x1e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvthf82ph  268435456(%rbp,%r14,8), %ymm22

// CHECK: vcvthf82ph  291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7f,0x2f,0x1e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvthf82ph  291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vcvthf82ph  (%rip), %ymm22
// CHECK: encoding: [0x62,0xe5,0x7f,0x28,0x1e,0x35,0x00,0x00,0x00,0x00]
          vcvthf82ph  (%rip), %ymm22

// CHECK: vcvthf82ph  -512(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe5,0x7f,0x28,0x1e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvthf82ph  -512(,%rbp,2), %ymm22

// CHECK: vcvthf82ph  2032(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7f,0xaf,0x1e,0x71,0x7f]
          vcvthf82ph  2032(%rcx), %ymm22 {%k7} {z}

// CHECK: vcvthf82ph  -2048(%rdx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7f,0xaf,0x1e,0x72,0x80]
          vcvthf82ph  -2048(%rdx), %ymm22 {%k7} {z}

// CHECK: vcvthf82ph  268435456(%rbp,%r14,8), %zmm22
// CHECK: encoding: [0x62,0xa5,0x7f,0x48,0x1e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvthf82ph  268435456(%rbp,%r14,8), %zmm22

// CHECK: vcvthf82ph  291(%r8,%rax,4), %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7f,0x4f,0x1e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvthf82ph  291(%r8,%rax,4), %zmm22 {%k7}

// CHECK: vcvthf82ph  (%rip), %zmm22
// CHECK: encoding: [0x62,0xe5,0x7f,0x48,0x1e,0x35,0x00,0x00,0x00,0x00]
          vcvthf82ph  (%rip), %zmm22

// CHECK: vcvthf82ph  -1024(,%rbp,2), %zmm22
// CHECK: encoding: [0x62,0xe5,0x7f,0x48,0x1e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvthf82ph  -1024(,%rbp,2), %zmm22

// CHECK: vcvthf82ph  4064(%rcx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7f,0xcf,0x1e,0x71,0x7f]
          vcvthf82ph  4064(%rcx), %zmm22 {%k7} {z}

// CHECK: vcvthf82ph  -4096(%rdx), %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7f,0xcf,0x1e,0x72,0x80]
          vcvthf82ph  -4096(%rdx), %zmm22 {%k7} {z}

// CHECK: vcvt2ph2bf8 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x82,0x47,0x20,0x74,0xf0]
          vcvt2ph2bf8 %ymm24, %ymm23, %ymm22

// CHECK: vcvt2ph2bf8 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x47,0x27,0x74,0xf0]
          vcvt2ph2bf8 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vcvt2ph2bf8 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x47,0xa7,0x74,0xf0]
          vcvt2ph2bf8 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ph2bf8 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x82,0x47,0x40,0x74,0xf0]
          vcvt2ph2bf8 %zmm24, %zmm23, %zmm22

// CHECK: vcvt2ph2bf8 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x47,0x47,0x74,0xf0]
          vcvt2ph2bf8 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vcvt2ph2bf8 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x47,0xc7,0x74,0xf0]
          vcvt2ph2bf8 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ph2bf8 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x82,0x47,0x00,0x74,0xf0]
          vcvt2ph2bf8 %xmm24, %xmm23, %xmm22

// CHECK: vcvt2ph2bf8 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x82,0x47,0x07,0x74,0xf0]
          vcvt2ph2bf8 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vcvt2ph2bf8 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x82,0x47,0x87,0x74,0xf0]
          vcvt2ph2bf8 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvt2ph2bf8  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa2,0x47,0x40,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vcvt2ph2bf8  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x47,0x47,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vcvt2ph2bf8  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x47,0x50,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvt2ph2bf8  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vcvt2ph2bf8  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe2,0x47,0x40,0x74,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ph2bf8  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vcvt2ph2bf8  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0xc7,0x74,0x71,0x7f]
          vcvt2ph2bf8  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ph2bf8  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0xd7,0x74,0x72,0x80]
          vcvt2ph2bf8  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ph2bf8  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x47,0x20,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vcvt2ph2bf8  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x47,0x27,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vcvt2ph2bf8  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x47,0x30,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvt2ph2bf8  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vcvt2ph2bf8  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe2,0x47,0x20,0x74,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ph2bf8  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vcvt2ph2bf8  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0xa7,0x74,0x71,0x7f]
          vcvt2ph2bf8  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ph2bf8  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0xb7,0x74,0x72,0x80]
          vcvt2ph2bf8  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ph2bf8  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x47,0x00,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vcvt2ph2bf8  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x47,0x07,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vcvt2ph2bf8  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x47,0x10,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvt2ph2bf8  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vcvt2ph2bf8  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe2,0x47,0x00,0x74,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ph2bf8  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vcvt2ph2bf8  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0x87,0x74,0x71,0x7f]
          vcvt2ph2bf8  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvt2ph2bf8  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x47,0x97,0x74,0x72,0x80]
          vcvt2ph2bf8  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvt2ph2bf8s %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x47,0x20,0x74,0xf0]
          vcvt2ph2bf8s %ymm24, %ymm23, %ymm22

// CHECK: vcvt2ph2bf8s %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x47,0x27,0x74,0xf0]
          vcvt2ph2bf8s %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vcvt2ph2bf8s %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x47,0xa7,0x74,0xf0]
          vcvt2ph2bf8s %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ph2bf8s %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x47,0x40,0x74,0xf0]
          vcvt2ph2bf8s %zmm24, %zmm23, %zmm22

// CHECK: vcvt2ph2bf8s %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x47,0x47,0x74,0xf0]
          vcvt2ph2bf8s %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vcvt2ph2bf8s %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x47,0xc7,0x74,0xf0]
          vcvt2ph2bf8s %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ph2bf8s %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x47,0x00,0x74,0xf0]
          vcvt2ph2bf8s %xmm24, %xmm23, %xmm22

// CHECK: vcvt2ph2bf8s %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x47,0x07,0x74,0xf0]
          vcvt2ph2bf8s %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vcvt2ph2bf8s %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x47,0x87,0x74,0xf0]
          vcvt2ph2bf8s %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvt2ph2bf8s  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x47,0x40,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8s  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vcvt2ph2bf8s  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x47,0x47,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8s  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vcvt2ph2bf8s  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x47,0x50,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvt2ph2bf8s  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vcvt2ph2bf8s  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x47,0x40,0x74,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ph2bf8s  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vcvt2ph2bf8s  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0xc7,0x74,0x71,0x7f]
          vcvt2ph2bf8s  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ph2bf8s  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0xd7,0x74,0x72,0x80]
          vcvt2ph2bf8s  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ph2bf8s  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x47,0x20,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8s  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vcvt2ph2bf8s  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x47,0x27,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8s  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vcvt2ph2bf8s  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x47,0x30,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvt2ph2bf8s  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vcvt2ph2bf8s  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x47,0x20,0x74,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ph2bf8s  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vcvt2ph2bf8s  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0xa7,0x74,0x71,0x7f]
          vcvt2ph2bf8s  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ph2bf8s  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0xb7,0x74,0x72,0x80]
          vcvt2ph2bf8s  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ph2bf8s  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x47,0x00,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8s  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vcvt2ph2bf8s  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x47,0x07,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8s  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vcvt2ph2bf8s  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x47,0x10,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvt2ph2bf8s  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vcvt2ph2bf8s  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x47,0x00,0x74,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ph2bf8s  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vcvt2ph2bf8s  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0x87,0x74,0x71,0x7f]
          vcvt2ph2bf8s  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvt2ph2bf8s  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0x97,0x74,0x72,0x80]
          vcvt2ph2bf8s  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvt2ph2hf8 %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x47,0x20,0x18,0xf0]
          vcvt2ph2hf8 %ymm24, %ymm23, %ymm22

// CHECK: vcvt2ph2hf8 %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x47,0x27,0x18,0xf0]
          vcvt2ph2hf8 %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vcvt2ph2hf8 %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x47,0xa7,0x18,0xf0]
          vcvt2ph2hf8 %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ph2hf8 %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x47,0x40,0x18,0xf0]
          vcvt2ph2hf8 %zmm24, %zmm23, %zmm22

// CHECK: vcvt2ph2hf8 %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x47,0x47,0x18,0xf0]
          vcvt2ph2hf8 %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vcvt2ph2hf8 %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x47,0xc7,0x18,0xf0]
          vcvt2ph2hf8 %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ph2hf8 %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x47,0x00,0x18,0xf0]
          vcvt2ph2hf8 %xmm24, %xmm23, %xmm22

// CHECK: vcvt2ph2hf8 %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x47,0x07,0x18,0xf0]
          vcvt2ph2hf8 %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vcvt2ph2hf8 %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x47,0x87,0x18,0xf0]
          vcvt2ph2hf8 %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvt2ph2hf8  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x47,0x40,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vcvt2ph2hf8  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x47,0x47,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vcvt2ph2hf8  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x47,0x50,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvt2ph2hf8  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vcvt2ph2hf8  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x47,0x40,0x18,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ph2hf8  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vcvt2ph2hf8  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0xc7,0x18,0x71,0x7f]
          vcvt2ph2hf8  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ph2hf8  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0xd7,0x18,0x72,0x80]
          vcvt2ph2hf8  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ph2hf8  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x47,0x20,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vcvt2ph2hf8  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x47,0x27,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vcvt2ph2hf8  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x47,0x30,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvt2ph2hf8  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vcvt2ph2hf8  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x47,0x20,0x18,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ph2hf8  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vcvt2ph2hf8  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0xa7,0x18,0x71,0x7f]
          vcvt2ph2hf8  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ph2hf8  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0xb7,0x18,0x72,0x80]
          vcvt2ph2hf8  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ph2hf8  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x47,0x00,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vcvt2ph2hf8  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x47,0x07,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vcvt2ph2hf8  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x47,0x10,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvt2ph2hf8  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vcvt2ph2hf8  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x47,0x00,0x18,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ph2hf8  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vcvt2ph2hf8  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0x87,0x18,0x71,0x7f]
          vcvt2ph2hf8  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvt2ph2hf8  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0x97,0x18,0x72,0x80]
          vcvt2ph2hf8  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvt2ph2hf8s %ymm24, %ymm23, %ymm22
// CHECK: encoding: [0x62,0x85,0x47,0x20,0x1b,0xf0]
          vcvt2ph2hf8s %ymm24, %ymm23, %ymm22

// CHECK: vcvt2ph2hf8s %ymm24, %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x47,0x27,0x1b,0xf0]
          vcvt2ph2hf8s %ymm24, %ymm23, %ymm22 {%k7}

// CHECK: vcvt2ph2hf8s %ymm24, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x47,0xa7,0x1b,0xf0]
          vcvt2ph2hf8s %ymm24, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ph2hf8s %zmm24, %zmm23, %zmm22
// CHECK: encoding: [0x62,0x85,0x47,0x40,0x1b,0xf0]
          vcvt2ph2hf8s %zmm24, %zmm23, %zmm22

// CHECK: vcvt2ph2hf8s %zmm24, %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x47,0x47,0x1b,0xf0]
          vcvt2ph2hf8s %zmm24, %zmm23, %zmm22 {%k7}

// CHECK: vcvt2ph2hf8s %zmm24, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x47,0xc7,0x1b,0xf0]
          vcvt2ph2hf8s %zmm24, %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ph2hf8s %xmm24, %xmm23, %xmm22
// CHECK: encoding: [0x62,0x85,0x47,0x00,0x1b,0xf0]
          vcvt2ph2hf8s %xmm24, %xmm23, %xmm22

// CHECK: vcvt2ph2hf8s %xmm24, %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0x85,0x47,0x07,0x1b,0xf0]
          vcvt2ph2hf8s %xmm24, %xmm23, %xmm22 {%k7}

// CHECK: vcvt2ph2hf8s %xmm24, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0x85,0x47,0x87,0x1b,0xf0]
          vcvt2ph2hf8s %xmm24, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvt2ph2hf8s  268435456(%rbp,%r14,8), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xa5,0x47,0x40,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8s  268435456(%rbp,%r14,8), %zmm23, %zmm22

// CHECK: vcvt2ph2hf8s  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x47,0x47,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8s  291(%r8,%rax,4), %zmm23, %zmm22 {%k7}

// CHECK: vcvt2ph2hf8s  (%rip){1to32}, %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x47,0x50,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvt2ph2hf8s  (%rip){1to32}, %zmm23, %zmm22

// CHECK: vcvt2ph2hf8s  -2048(,%rbp,2), %zmm23, %zmm22
// CHECK: encoding: [0x62,0xe5,0x47,0x40,0x1b,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ph2hf8s  -2048(,%rbp,2), %zmm23, %zmm22

// CHECK: vcvt2ph2hf8s  8128(%rcx), %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0xc7,0x1b,0x71,0x7f]
          vcvt2ph2hf8s  8128(%rcx), %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ph2hf8s  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0xd7,0x1b,0x72,0x80]
          vcvt2ph2hf8s  -256(%rdx){1to32}, %zmm23, %zmm22 {%k7} {z}

// CHECK: vcvt2ph2hf8s  268435456(%rbp,%r14,8), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x47,0x20,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8s  268435456(%rbp,%r14,8), %ymm23, %ymm22

// CHECK: vcvt2ph2hf8s  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x47,0x27,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8s  291(%r8,%rax,4), %ymm23, %ymm22 {%k7}

// CHECK: vcvt2ph2hf8s  (%rip){1to16}, %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x47,0x30,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvt2ph2hf8s  (%rip){1to16}, %ymm23, %ymm22

// CHECK: vcvt2ph2hf8s  -1024(,%rbp,2), %ymm23, %ymm22
// CHECK: encoding: [0x62,0xe5,0x47,0x20,0x1b,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ph2hf8s  -1024(,%rbp,2), %ymm23, %ymm22

// CHECK: vcvt2ph2hf8s  4064(%rcx), %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0xa7,0x1b,0x71,0x7f]
          vcvt2ph2hf8s  4064(%rcx), %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ph2hf8s  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0xb7,0x1b,0x72,0x80]
          vcvt2ph2hf8s  -256(%rdx){1to16}, %ymm23, %ymm22 {%k7} {z}

// CHECK: vcvt2ph2hf8s  268435456(%rbp,%r14,8), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x47,0x00,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8s  268435456(%rbp,%r14,8), %xmm23, %xmm22

// CHECK: vcvt2ph2hf8s  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x47,0x07,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8s  291(%r8,%rax,4), %xmm23, %xmm22 {%k7}

// CHECK: vcvt2ph2hf8s  (%rip){1to8}, %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x47,0x10,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvt2ph2hf8s  (%rip){1to8}, %xmm23, %xmm22

// CHECK: vcvt2ph2hf8s  -512(,%rbp,2), %xmm23, %xmm22
// CHECK: encoding: [0x62,0xe5,0x47,0x00,0x1b,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ph2hf8s  -512(,%rbp,2), %xmm23, %xmm22

// CHECK: vcvt2ph2hf8s  2032(%rcx), %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0x87,0x1b,0x71,0x7f]
          vcvt2ph2hf8s  2032(%rcx), %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvt2ph2hf8s  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x47,0x97,0x1b,0x72,0x80]
          vcvt2ph2hf8s  -256(%rdx){1to8}, %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtph2bf8 %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x7e,0x08,0x74,0xf7]
          vcvtph2bf8 %xmm23, %xmm22

// CHECK: vcvtph2bf8 %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa2,0x7e,0x0f,0x74,0xf7]
          vcvtph2bf8 %xmm23, %xmm22 {%k7}

// CHECK: vcvtph2bf8 %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa2,0x7e,0x8f,0x74,0xf7]
          vcvtph2bf8 %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtph2bf8 %zmm23, %ymm22
// CHECK: encoding: [0x62,0xa2,0x7e,0x48,0x74,0xf7]
          vcvtph2bf8 %zmm23, %ymm22

// CHECK: vcvtph2bf8 %zmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa2,0x7e,0x4f,0x74,0xf7]
          vcvtph2bf8 %zmm23, %ymm22 {%k7}

// CHECK: vcvtph2bf8 %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa2,0x7e,0xcf,0x74,0xf7]
          vcvtph2bf8 %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtph2bf8 %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa2,0x7e,0x28,0x74,0xf7]
          vcvtph2bf8 %ymm23, %xmm22

// CHECK: vcvtph2bf8 %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa2,0x7e,0x2f,0x74,0xf7]
          vcvtph2bf8 %ymm23, %xmm22 {%k7}

// CHECK: vcvtph2bf8 %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa2,0x7e,0xaf,0x74,0xf7]
          vcvtph2bf8 %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtph2bf8x  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa2,0x7e,0x08,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2bf8x  268435456(%rbp,%r14,8), %xmm22

// CHECK: vcvtph2bf8x  291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x7e,0x0f,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2bf8x  291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vcvtph2bf8  (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe2,0x7e,0x18,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtph2bf8  (%rip){1to8}, %xmm22

// CHECK: vcvtph2bf8x  -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe2,0x7e,0x08,0x74,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2bf8x  -512(,%rbp,2), %xmm22

// CHECK: vcvtph2bf8x  2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x7e,0x8f,0x74,0x71,0x7f]
          vcvtph2bf8x  2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vcvtph2bf8  -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x7e,0x9f,0x74,0x72,0x80]
          vcvtph2bf8  -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vcvtph2bf8  (%rip){1to16}, %xmm22
// CHECK: encoding: [0x62,0xe2,0x7e,0x38,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtph2bf8  (%rip){1to16}, %xmm22

// CHECK: vcvtph2bf8y  -1024(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe2,0x7e,0x28,0x74,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2bf8y  -1024(,%rbp,2), %xmm22

// CHECK: vcvtph2bf8y  4064(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x7e,0xaf,0x74,0x71,0x7f]
          vcvtph2bf8y  4064(%rcx), %xmm22 {%k7} {z}

// CHECK: vcvtph2bf8  -256(%rdx){1to16}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x7e,0xbf,0x74,0x72,0x80]
          vcvtph2bf8  -256(%rdx){1to16}, %xmm22 {%k7} {z}

// CHECK: vcvtph2bf8  268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa2,0x7e,0x48,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2bf8  268435456(%rbp,%r14,8), %ymm22

// CHECK: vcvtph2bf8  291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc2,0x7e,0x4f,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2bf8  291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vcvtph2bf8  (%rip){1to32}, %ymm22
// CHECK: encoding: [0x62,0xe2,0x7e,0x58,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtph2bf8  (%rip){1to32}, %ymm22

// CHECK: vcvtph2bf8  -2048(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe2,0x7e,0x48,0x74,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2bf8  -2048(,%rbp,2), %ymm22

// CHECK: vcvtph2bf8  8128(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x7e,0xcf,0x74,0x71,0x7f]
          vcvtph2bf8  8128(%rcx), %ymm22 {%k7} {z}

// CHECK: vcvtph2bf8  -256(%rdx){1to32}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe2,0x7e,0xdf,0x74,0x72,0x80]
          vcvtph2bf8  -256(%rdx){1to32}, %ymm22 {%k7} {z}

// CHECK: vcvtph2bf8s %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x74,0xf7]
          vcvtph2bf8s %xmm23, %xmm22

// CHECK: vcvtph2bf8s %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7e,0x0f,0x74,0xf7]
          vcvtph2bf8s %xmm23, %xmm22 {%k7}

// CHECK: vcvtph2bf8s %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7e,0x8f,0x74,0xf7]
          vcvtph2bf8s %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtph2bf8s %zmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x48,0x74,0xf7]
          vcvtph2bf8s %zmm23, %ymm22

// CHECK: vcvtph2bf8s %zmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7e,0x4f,0x74,0xf7]
          vcvtph2bf8s %zmm23, %ymm22 {%k7}

// CHECK: vcvtph2bf8s %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7e,0xcf,0x74,0xf7]
          vcvtph2bf8s %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtph2bf8s %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x28,0x74,0xf7]
          vcvtph2bf8s %ymm23, %xmm22

// CHECK: vcvtph2bf8s %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7e,0x2f,0x74,0xf7]
          vcvtph2bf8s %ymm23, %xmm22 {%k7}

// CHECK: vcvtph2bf8s %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7e,0xaf,0x74,0xf7]
          vcvtph2bf8s %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtph2bf8sx  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2bf8sx  268435456(%rbp,%r14,8), %xmm22

// CHECK: vcvtph2bf8sx  291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7e,0x0f,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2bf8sx  291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vcvtph2bf8s  (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x18,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtph2bf8s  (%rip){1to8}, %xmm22

// CHECK: vcvtph2bf8sx  -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x08,0x74,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2bf8sx  -512(,%rbp,2), %xmm22

// CHECK: vcvtph2bf8sx  2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0x8f,0x74,0x71,0x7f]
          vcvtph2bf8sx  2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vcvtph2bf8s  -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0x9f,0x74,0x72,0x80]
          vcvtph2bf8s  -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vcvtph2bf8s  (%rip){1to16}, %xmm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x38,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtph2bf8s  (%rip){1to16}, %xmm22

// CHECK: vcvtph2bf8sy  -1024(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x28,0x74,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2bf8sy  -1024(,%rbp,2), %xmm22

// CHECK: vcvtph2bf8sy  4064(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0xaf,0x74,0x71,0x7f]
          vcvtph2bf8sy  4064(%rcx), %xmm22 {%k7} {z}

// CHECK: vcvtph2bf8s  -256(%rdx){1to16}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0xbf,0x74,0x72,0x80]
          vcvtph2bf8s  -256(%rdx){1to16}, %xmm22 {%k7} {z}

// CHECK: vcvtph2bf8s  268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x48,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2bf8s  268435456(%rbp,%r14,8), %ymm22

// CHECK: vcvtph2bf8s  291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7e,0x4f,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2bf8s  291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vcvtph2bf8s  (%rip){1to32}, %ymm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x58,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtph2bf8s  (%rip){1to32}, %ymm22

// CHECK: vcvtph2bf8s  -2048(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x48,0x74,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2bf8s  -2048(,%rbp,2), %ymm22

// CHECK: vcvtph2bf8s  8128(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0xcf,0x74,0x71,0x7f]
          vcvtph2bf8s  8128(%rcx), %ymm22 {%k7} {z}

// CHECK: vcvtph2bf8s  -256(%rdx){1to32}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0xdf,0x74,0x72,0x80]
          vcvtph2bf8s  -256(%rdx){1to32}, %ymm22 {%k7} {z}

// CHECK: vcvtph2hf8 %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x18,0xf7]
          vcvtph2hf8 %xmm23, %xmm22

// CHECK: vcvtph2hf8 %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7e,0x0f,0x18,0xf7]
          vcvtph2hf8 %xmm23, %xmm22 {%k7}

// CHECK: vcvtph2hf8 %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7e,0x8f,0x18,0xf7]
          vcvtph2hf8 %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtph2hf8 %zmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x48,0x18,0xf7]
          vcvtph2hf8 %zmm23, %ymm22

// CHECK: vcvtph2hf8 %zmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7e,0x4f,0x18,0xf7]
          vcvtph2hf8 %zmm23, %ymm22 {%k7}

// CHECK: vcvtph2hf8 %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7e,0xcf,0x18,0xf7]
          vcvtph2hf8 %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtph2hf8 %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x28,0x18,0xf7]
          vcvtph2hf8 %ymm23, %xmm22

// CHECK: vcvtph2hf8 %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7e,0x2f,0x18,0xf7]
          vcvtph2hf8 %ymm23, %xmm22 {%k7}

// CHECK: vcvtph2hf8 %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7e,0xaf,0x18,0xf7]
          vcvtph2hf8 %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtph2hf8x  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2hf8x  268435456(%rbp,%r14,8), %xmm22

// CHECK: vcvtph2hf8x  291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7e,0x0f,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2hf8x  291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vcvtph2hf8  (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x18,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtph2hf8  (%rip){1to8}, %xmm22

// CHECK: vcvtph2hf8x  -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x08,0x18,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2hf8x  -512(,%rbp,2), %xmm22

// CHECK: vcvtph2hf8x  2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0x8f,0x18,0x71,0x7f]
          vcvtph2hf8x  2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vcvtph2hf8  -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0x9f,0x18,0x72,0x80]
          vcvtph2hf8  -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vcvtph2hf8  (%rip){1to16}, %xmm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x38,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtph2hf8  (%rip){1to16}, %xmm22

// CHECK: vcvtph2hf8y  -1024(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x28,0x18,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2hf8y  -1024(,%rbp,2), %xmm22

// CHECK: vcvtph2hf8y  4064(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0xaf,0x18,0x71,0x7f]
          vcvtph2hf8y  4064(%rcx), %xmm22 {%k7} {z}

// CHECK: vcvtph2hf8  -256(%rdx){1to16}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0xbf,0x18,0x72,0x80]
          vcvtph2hf8  -256(%rdx){1to16}, %xmm22 {%k7} {z}

// CHECK: vcvtph2hf8  268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x48,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2hf8  268435456(%rbp,%r14,8), %ymm22

// CHECK: vcvtph2hf8  291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7e,0x4f,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2hf8  291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vcvtph2hf8  (%rip){1to32}, %ymm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x58,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtph2hf8  (%rip){1to32}, %ymm22

// CHECK: vcvtph2hf8  -2048(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x48,0x18,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2hf8  -2048(,%rbp,2), %ymm22

// CHECK: vcvtph2hf8  8128(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0xcf,0x18,0x71,0x7f]
          vcvtph2hf8  8128(%rcx), %ymm22 {%k7} {z}

// CHECK: vcvtph2hf8  -256(%rdx){1to32}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0xdf,0x18,0x72,0x80]
          vcvtph2hf8  -256(%rdx){1to32}, %ymm22 {%k7} {z}

// CHECK: vcvtph2hf8s %xmm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x1b,0xf7]
          vcvtph2hf8s %xmm23, %xmm22

// CHECK: vcvtph2hf8s %xmm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7e,0x0f,0x1b,0xf7]
          vcvtph2hf8s %xmm23, %xmm22 {%k7}

// CHECK: vcvtph2hf8s %xmm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7e,0x8f,0x1b,0xf7]
          vcvtph2hf8s %xmm23, %xmm22 {%k7} {z}

// CHECK: vcvtph2hf8s %zmm23, %ymm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x48,0x1b,0xf7]
          vcvtph2hf8s %zmm23, %ymm22

// CHECK: vcvtph2hf8s %zmm23, %ymm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7e,0x4f,0x1b,0xf7]
          vcvtph2hf8s %zmm23, %ymm22 {%k7}

// CHECK: vcvtph2hf8s %zmm23, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7e,0xcf,0x1b,0xf7]
          vcvtph2hf8s %zmm23, %ymm22 {%k7} {z}

// CHECK: vcvtph2hf8s %ymm23, %xmm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x28,0x1b,0xf7]
          vcvtph2hf8s %ymm23, %xmm22

// CHECK: vcvtph2hf8s %ymm23, %xmm22 {%k7}
// CHECK: encoding: [0x62,0xa5,0x7e,0x2f,0x1b,0xf7]
          vcvtph2hf8s %ymm23, %xmm22 {%k7}

// CHECK: vcvtph2hf8s %ymm23, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xa5,0x7e,0xaf,0x1b,0xf7]
          vcvtph2hf8s %ymm23, %xmm22 {%k7} {z}

// CHECK: vcvtph2hf8sx  268435456(%rbp,%r14,8), %xmm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2hf8sx  268435456(%rbp,%r14,8), %xmm22

// CHECK: vcvtph2hf8sx  291(%r8,%rax,4), %xmm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7e,0x0f,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2hf8sx  291(%r8,%rax,4), %xmm22 {%k7}

// CHECK: vcvtph2hf8s  (%rip){1to8}, %xmm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x18,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtph2hf8s  (%rip){1to8}, %xmm22

// CHECK: vcvtph2hf8sx  -512(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x08,0x1b,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2hf8sx  -512(,%rbp,2), %xmm22

// CHECK: vcvtph2hf8sx  2032(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0x8f,0x1b,0x71,0x7f]
          vcvtph2hf8sx  2032(%rcx), %xmm22 {%k7} {z}

// CHECK: vcvtph2hf8s  -256(%rdx){1to8}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0x9f,0x1b,0x72,0x80]
          vcvtph2hf8s  -256(%rdx){1to8}, %xmm22 {%k7} {z}

// CHECK: vcvtph2hf8s  (%rip){1to16}, %xmm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x38,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtph2hf8s  (%rip){1to16}, %xmm22

// CHECK: vcvtph2hf8sy  -1024(,%rbp,2), %xmm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x28,0x1b,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2hf8sy  -1024(,%rbp,2), %xmm22

// CHECK: vcvtph2hf8sy  4064(%rcx), %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0xaf,0x1b,0x71,0x7f]
          vcvtph2hf8sy  4064(%rcx), %xmm22 {%k7} {z}

// CHECK: vcvtph2hf8s  -256(%rdx){1to16}, %xmm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0xbf,0x1b,0x72,0x80]
          vcvtph2hf8s  -256(%rdx){1to16}, %xmm22 {%k7} {z}

// CHECK: vcvtph2hf8s  268435456(%rbp,%r14,8), %ymm22
// CHECK: encoding: [0x62,0xa5,0x7e,0x48,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2hf8s  268435456(%rbp,%r14,8), %ymm22

// CHECK: vcvtph2hf8s  291(%r8,%rax,4), %ymm22 {%k7}
// CHECK: encoding: [0x62,0xc5,0x7e,0x4f,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2hf8s  291(%r8,%rax,4), %ymm22 {%k7}

// CHECK: vcvtph2hf8s  (%rip){1to32}, %ymm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x58,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtph2hf8s  (%rip){1to32}, %ymm22

// CHECK: vcvtph2hf8s  -2048(,%rbp,2), %ymm22
// CHECK: encoding: [0x62,0xe5,0x7e,0x48,0x1b,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2hf8s  -2048(,%rbp,2), %ymm22

// CHECK: vcvtph2hf8s  8128(%rcx), %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0xcf,0x1b,0x71,0x7f]
          vcvtph2hf8s  8128(%rcx), %ymm22 {%k7} {z}

// CHECK: vcvtph2hf8s  -256(%rdx){1to32}, %ymm22 {%k7} {z}
// CHECK: encoding: [0x62,0xe5,0x7e,0xdf,0x1b,0x72,0x80]
          vcvtph2hf8s  -256(%rdx){1to32}, %ymm22 {%k7} {z}

