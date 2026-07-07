// RUN: llvm-mc -triple x86_64-unknown-unknown -mattr=+acev1 -show-encoding %s | FileCheck %s

// CHECK: bsrinit %bsr0
// CHECK: encoding: [0xc4,0xe2,0xfb,0x49,0xc0]
          bsrinit %bsr0

// CHECK: bsrmovf %zmm2, %zmm1, %bsr0
// CHECK: encoding: [0x62,0xf6,0xf4,0x48,0x95,0xc2]
          bsrmovf %zmm2, %zmm1, %bsr0

// CHECK: bsrmovf (%rax), %zmm1, %bsr0
// CHECK: encoding: [0x62,0xf6,0xf4,0x48,0x95,0x00]
          bsrmovf (%rax), %zmm1, %bsr0

// CHECK: bsrmovh %zmm1, %bsr0
// CHECK: encoding: [0x62,0xf6,0xff,0x48,0x95,0xc1]
          bsrmovh %zmm1, %bsr0

// CHECK: bsrmovh (%rbx), %bsr0
// CHECK: encoding: [0x62,0xf6,0xff,0x48,0x95,0x03]
          bsrmovh (%rbx), %bsr0

// CHECK: bsrmovh %bsr0, %zmm1
// CHECK: encoding: [0x62,0xf6,0x7f,0x48,0x95,0xc1]
          bsrmovh %bsr0, %zmm1

// CHECK: bsrmovh %bsr0, (%rcx)
// CHECK: encoding: [0x62,0xf6,0x7f,0x48,0x95,0x01]
          bsrmovh %bsr0, (%rcx)

// CHECK: bsrmovl %zmm2, %bsr0
// CHECK: encoding: [0x62,0xf6,0xfe,0x48,0x95,0xc2]
          bsrmovl %zmm2, %bsr0

// CHECK: bsrmovl (%rdx), %bsr0
// CHECK: encoding: [0x62,0xf6,0xfe,0x48,0x95,0x02]
          bsrmovl (%rdx), %bsr0

// CHECK: bsrmovl %bsr0, %zmm3
// CHECK: encoding: [0x62,0xf6,0x7e,0x48,0x95,0xc3]
          bsrmovl %bsr0, %zmm3

// CHECK: bsrmovl %bsr0, (%rsi)
// CHECK: encoding: [0x62,0xf6,0x7e,0x48,0x95,0x06]
          bsrmovl %bsr0, (%rsi)

// CHECK: tilemovcol $5, %zmm2, %tmm1
// CHECK: encoding: [0x62,0xf3,0xfd,0x48,0x2f,0xca,0x05]
          tilemovcol $5, %zmm2, %tmm1

// CHECK: tilemovcol %ecx, %zmm2, %tmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x48,0x4b,0xca]
          tilemovcol %ecx, %zmm2, %tmm1

// CHECK: tilemovrow $3, %zmm2, %tmm1
// CHECK: encoding: [0x62,0xf3,0xfd,0x48,0x07,0xca,0x03]
          tilemovrow $3, %zmm2, %tmm1

// CHECK: tilemovrow %edx, %zmm3, %tmm2
// CHECK: encoding: [0x62,0xf2,0xed,0x48,0x4a,0xd3]
          tilemovrow %edx, %zmm3, %tmm2

// CHECK: top2bf16ps %zmm3, %zmm2, %tmm1
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0x5c,0xca]
          top2bf16ps %zmm3, %zmm2, %tmm1

// CHECK: top4buud %zmm3, %zmm2, %tmm1
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x5e,0xca]
          top4buud %zmm3, %zmm2, %tmm1

// CHECK: top4busd %zmm3, %zmm2, %tmm1
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0x5e,0xca]
          top4busd %zmm3, %zmm2, %tmm1

// CHECK: top4bssd %zmm3, %zmm2, %tmm1
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x5e,0xca]
          top4bssd %zmm3, %zmm2, %tmm1

// CHECK: top4bsud %zmm3, %zmm2, %tmm1
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0x5e,0xca]
          top4bsud %zmm3, %zmm2, %tmm1

// CHECK: top4mxbf8ps $7, %zmm3, %zmm2, %tmm1
// CHECK: encoding: [0x62,0xf3,0x64,0x48,0x8d,0xca,0x07]
          top4mxbf8ps $7, %zmm3, %zmm2, %tmm1

// CHECK: top4mxhf8ps $7, %zmm3, %zmm2, %tmm1
// CHECK: encoding: [0x62,0xf3,0x65,0x48,0x8d,0xca,0x07]
          top4mxhf8ps $7, %zmm3, %zmm2, %tmm1

// CHECK: top4mxbhf8ps $3, %zmm4, %zmm5, %tmm2
// CHECK: encoding: [0x62,0xf3,0x5f,0x48,0x8d,0xd5,0x03]
          top4mxbhf8ps $3, %zmm4, %zmm5, %tmm2

// CHECK: top4mxhbf8ps $1, %zmm6, %zmm7, %tmm3
// CHECK: encoding: [0x62,0xf3,0x4e,0x48,0x8d,0xdf,0x01]
          top4mxhbf8ps $1, %zmm6, %zmm7, %tmm3

// CHECK: top4mxbssps $15, %zmm8, %zmm9, %tmm4
// CHECK: encoding: [0x62,0xd3,0x3f,0x48,0x8f,0xe1,0x0f]
          top4mxbssps $15, %zmm8, %zmm9, %tmm4

// CHECK: top4buud %zmm31, %zmm30, %tmm7
// CHECK: encoding: [0x62,0xd2,0x04,0x40,0x5e,0xfe]
          top4buud %zmm31, %zmm30, %tmm7

// CHECK: bsrmovf 64(%rdx,%rax,4), %zmm10, %bsr0
// CHECK: encoding: [0x62,0xf6,0xac,0x48,0x95,0x44,0x82,0x01]
          bsrmovf 64(%rdx,%rax,4), %zmm10, %bsr0
