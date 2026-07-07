// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 -mattr=+acev1 --show-encoding %s | FileCheck %s

// CHECK: bsrinit bsr0
// CHECK: encoding: [0xc4,0xe2,0xfb,0x49,0xc0]
          bsrinit bsr0

// CHECK: bsrmovf bsr0, zmm1, zmm2
// CHECK: encoding: [0x62,0xf6,0xf4,0x48,0x95,0xc2]
          bsrmovf bsr0, zmm1, zmm2

// CHECK: bsrmovf bsr0, zmm1, zmmword ptr [rax]
// CHECK: encoding: [0x62,0xf6,0xf4,0x48,0x95,0x00]
          bsrmovf bsr0, zmm1, zmmword ptr [rax]

// CHECK: bsrmovf bsr0, zmm1, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xb6,0xf4,0x48,0x95,0x84,0xf5,0x00,0x00,0x00,0x10]
          bsrmovf bsr0, zmm1, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: bsrmovh bsr0, zmm1
// CHECK: encoding: [0x62,0xf6,0xff,0x48,0x95,0xc1]
          bsrmovh bsr0, zmm1

// CHECK: bsrmovh bsr0, zmmword ptr [rbx]
// CHECK: encoding: [0x62,0xf6,0xff,0x48,0x95,0x03]
          bsrmovh bsr0, zmmword ptr [rbx]

// CHECK: bsrmovh zmm1, bsr0
// CHECK: encoding: [0x62,0xf6,0x7f,0x48,0x95,0xc1]
          bsrmovh zmm1, bsr0

// CHECK: bsrmovh zmmword ptr [rcx], bsr0
// CHECK: encoding: [0x62,0xf6,0x7f,0x48,0x95,0x01]
          bsrmovh zmmword ptr [rcx], bsr0

// CHECK: bsrmovl bsr0, zmm2
// CHECK: encoding: [0x62,0xf6,0xfe,0x48,0x95,0xc2]
          bsrmovl bsr0, zmm2

// CHECK: bsrmovl bsr0, zmmword ptr [rdx]
// CHECK: encoding: [0x62,0xf6,0xfe,0x48,0x95,0x02]
          bsrmovl bsr0, zmmword ptr [rdx]

// CHECK: bsrmovl zmm3, bsr0
// CHECK: encoding: [0x62,0xf6,0x7e,0x48,0x95,0xc3]
          bsrmovl zmm3, bsr0

// CHECK: bsrmovl zmmword ptr [rsi], bsr0
// CHECK: encoding: [0x62,0xf6,0x7e,0x48,0x95,0x06]
          bsrmovl zmmword ptr [rsi], bsr0

// CHECK: tilemovcol tmm1, zmm2, 5
// CHECK: encoding: [0x62,0xf3,0xfd,0x48,0x2f,0xca,0x05]
          tilemovcol tmm1, zmm2, 5

// CHECK: tilemovcol tmm1, zmm2, ecx
// CHECK: encoding: [0x62,0xf2,0xf5,0x48,0x4b,0xca]
          tilemovcol tmm1, zmm2, ecx

// CHECK: tilemovrow tmm1, zmm2, 3
// CHECK: encoding: [0x62,0xf3,0xfd,0x48,0x07,0xca,0x03]
          tilemovrow tmm1, zmm2, 3

// CHECK: tilemovrow tmm2, zmm3, edx
// CHECK: encoding: [0x62,0xf2,0xed,0x48,0x4a,0xd3]
          tilemovrow tmm2, zmm3, edx

// CHECK: top2bf16ps tmm1, zmm2, zmm3
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0x5c,0xca]
          top2bf16ps tmm1, zmm2, zmm3

// CHECK: top4buud tmm1, zmm2, zmm3
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x5e,0xca]
          top4buud tmm1, zmm2, zmm3

// CHECK: top4busd tmm1, zmm2, zmm3
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0x5e,0xca]
          top4busd tmm1, zmm2, zmm3

// CHECK: top4bssd tmm1, zmm2, zmm3
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x5e,0xca]
          top4bssd tmm1, zmm2, zmm3

// CHECK: top4bsud tmm1, zmm2, zmm3
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0x5e,0xca]
          top4bsud tmm1, zmm2, zmm3

// CHECK: top4mxbf8ps tmm1, zmm2, zmm3, 7
// CHECK: encoding: [0x62,0xf3,0x64,0x48,0x8d,0xca,0x07]
          top4mxbf8ps tmm1, zmm2, zmm3, 7

// CHECK: top4mxhf8ps tmm1, zmm2, zmm3, 7
// CHECK: encoding: [0x62,0xf3,0x65,0x48,0x8d,0xca,0x07]
          top4mxhf8ps tmm1, zmm2, zmm3, 7

// CHECK: top4mxbhf8ps tmm2, zmm4, zmm5, 3
// CHECK: encoding: [0x62,0xf3,0x57,0x48,0x8d,0xd4,0x03]
          top4mxbhf8ps tmm2, zmm4, zmm5, 3

// CHECK: top4mxhbf8ps tmm3, zmm6, zmm7, 1
// CHECK: encoding: [0x62,0xf3,0x46,0x48,0x8d,0xde,0x01]
          top4mxhbf8ps tmm3, zmm6, zmm7, 1

// CHECK: top4mxbssps tmm4, zmm8, zmm9, 15
// CHECK: encoding: [0x62,0xd3,0x37,0x48,0x8f,0xe0,0x0f]
          top4mxbssps tmm4, zmm8, zmm9, 15

// CHECK: top4buud tmm7, zmm30, zmm31
// CHECK: encoding: [0x62,0xd2,0x04,0x40,0x5e,0xfe]
          top4buud tmm7, zmm30, zmm31
