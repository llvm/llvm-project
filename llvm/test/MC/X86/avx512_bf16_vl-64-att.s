// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding < %s | FileCheck %s

// CHECK: vcvtne2ps2bf16 %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x02,0x17,0x00,0x72,0xf4]
          vcvtne2ps2bf16 %xmm28, %xmm29, %xmm30

// CHECK: vcvtne2ps2bf16 %xmm28, %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x02,0x17,0x07,0x72,0xf4]
          vcvtne2ps2bf16 %xmm28, %xmm29, %xmm30 {%k7}

// CHECK: vcvtne2ps2bf16 %xmm28, %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x02,0x17,0x87,0x72,0xf4]
          vcvtne2ps2bf16 %xmm28, %xmm29, %xmm30 {%k7} {z}

// CHECK: vcvtne2ps2bf16   (%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x62,0x17,0x00,0x72,0x31]
          vcvtne2ps2bf16   (%rcx), %xmm29, %xmm30

// CHECK: vcvtne2ps2bf16   291(%rax,%r14,8), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x22,0x17,0x00,0x72,0xb4,0xf0,0x23,0x01,0x00,0x00]
          vcvtne2ps2bf16   291(%rax,%r14,8), %xmm29, %xmm30

// CHECK: vcvtne2ps2bf16   268435456(%rax,%r14,8), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x22,0x17,0x00,0x72,0xb4,0xf0,0x00,0x00,0x00,0x10]
          vcvtne2ps2bf16   268435456(%rax,%r14,8), %xmm29, %xmm30

// CHECK: vcvtne2ps2bf16   -16(%rsp), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x62,0x17,0x00,0x72,0x74,0x24,0xff]
          vcvtne2ps2bf16   -16(%rsp), %xmm29, %xmm30

// CHECK: vcvtne2ps2bf16   (%rcx){1to4}, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x62,0x17,0x10,0x72,0x31]
          vcvtne2ps2bf16   (%rcx){1to4}, %xmm29, %xmm30

// CHECK: vcvtne2ps2bf16   2032(%rdx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x62,0x17,0x00,0x72,0x72,0x7f]
          vcvtne2ps2bf16   2032(%rdx), %xmm29, %xmm30

// CHECK: vcvtne2ps2bf16   -2048(%rdx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x62,0x17,0x00,0x72,0x72,0x80]
          vcvtne2ps2bf16   -2048(%rdx), %xmm29, %xmm30

// CHECK: vcvtne2ps2bf16   508(%rdx){1to4}, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x62,0x17,0x10,0x72,0x72,0x7f]
          vcvtne2ps2bf16   508(%rdx){1to4}, %xmm29, %xmm30

// CHECK: vcvtne2ps2bf16   -512(%rdx){1to4}, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x62,0x17,0x10,0x72,0x72,0x80]
          vcvtne2ps2bf16   -512(%rdx){1to4}, %xmm29, %xmm30

// CHECK: vcvtne2ps2bf16 %ymm28, %ymm29, %ymm30
// CHECK: encoding: [0x62,0x02,0x17,0x20,0x72,0xf4]
          vcvtne2ps2bf16 %ymm28, %ymm29, %ymm30

// CHECK: vcvtne2ps2bf16 %ymm28, %ymm29, %ymm30 {%k7}
// CHECK: encoding: [0x62,0x02,0x17,0x27,0x72,0xf4]
          vcvtne2ps2bf16 %ymm28, %ymm29, %ymm30 {%k7}

// CHECK: vcvtne2ps2bf16 %ymm28, %ymm29, %ymm30 {%k7} {z}
// CHECK: encoding: [0x62,0x02,0x17,0xa7,0x72,0xf4]
          vcvtne2ps2bf16 %ymm28, %ymm29, %ymm30 {%k7} {z}

// CHECK: vcvtne2ps2bf16   (%rcx), %ymm29, %ymm30
// CHECK: encoding: [0x62,0x62,0x17,0x20,0x72,0x31]
          vcvtne2ps2bf16   (%rcx), %ymm29, %ymm30

// CHECK: vcvtne2ps2bf16   291(%rax,%r14,8), %ymm29, %ymm30
// CHECK: encoding: [0x62,0x22,0x17,0x20,0x72,0xb4,0xf0,0x23,0x01,0x00,0x00]
          vcvtne2ps2bf16   291(%rax,%r14,8), %ymm29, %ymm30

// CHECK: vcvtne2ps2bf16   268435456(%rax,%r14,8), %ymm29, %ymm30
// CHECK: encoding: [0x62,0x22,0x17,0x20,0x72,0xb4,0xf0,0x00,0x00,0x00,0x10]
          vcvtne2ps2bf16   268435456(%rax,%r14,8), %ymm29, %ymm30

// CHECK: vcvtne2ps2bf16   -32(%rsp), %ymm29, %ymm30
// CHECK: encoding: [0x62,0x62,0x17,0x20,0x72,0x74,0x24,0xff]
          vcvtne2ps2bf16   -32(%rsp), %ymm29, %ymm30

// CHECK: vcvtne2ps2bf16   (%rcx){1to8}, %ymm29, %ymm30
// CHECK: encoding: [0x62,0x62,0x17,0x30,0x72,0x31]
          vcvtne2ps2bf16   (%rcx){1to8}, %ymm29, %ymm30

// CHECK: vcvtne2ps2bf16   4064(%rdx), %ymm29, %ymm30
// CHECK: encoding: [0x62,0x62,0x17,0x20,0x72,0x72,0x7f]
          vcvtne2ps2bf16   4064(%rdx), %ymm29, %ymm30

// CHECK: vcvtne2ps2bf16   -4096(%rdx), %ymm29, %ymm30
// CHECK: encoding: [0x62,0x62,0x17,0x20,0x72,0x72,0x80]
          vcvtne2ps2bf16   -4096(%rdx), %ymm29, %ymm30

// CHECK: vcvtne2ps2bf16   508(%rdx){1to8}, %ymm29, %ymm30
// CHECK: encoding: [0x62,0x62,0x17,0x30,0x72,0x72,0x7f]
          vcvtne2ps2bf16   508(%rdx){1to8}, %ymm29, %ymm30

// CHECK: vcvtne2ps2bf16   -512(%rdx){1to8}, %ymm29, %ymm30
// CHECK: encoding: [0x62,0x62,0x17,0x30,0x72,0x72,0x80]
          vcvtne2ps2bf16   -512(%rdx){1to8}, %ymm29, %ymm30

// CHECK: vcvtneps2bf16 %xmm29, %xmm30
// CHECK: encoding: [0x62,0x02,0x7e,0x08,0x72,0xf5]
          vcvtneps2bf16 %xmm29, %xmm30

// CHECK: vcvtneps2bf16x  268435456(%rbp,%r14,8), %xmm30 {%k7}
// CHECK: encoding: [0x62,0x22,0x7e,0x0f,0x72,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneps2bf16x  268435456(%rbp,%r14,8), %xmm30 {%k7}

// CHECK: vcvtneps2bf16   (%r9){1to4}, %xmm30
// CHECK: encoding: [0x62,0x42,0x7e,0x18,0x72,0x31]
          vcvtneps2bf16   (%r9){1to4}, %xmm30

// CHECK: vcvtneps2bf16x  2032(%rcx), %xmm30
// CHECK: encoding: [0x62,0x62,0x7e,0x08,0x72,0x71,0x7f]
          vcvtneps2bf16x  2032(%rcx), %xmm30

// CHECK: vcvtneps2bf16   -512(%rdx){1to4}, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x62,0x7e,0x9f,0x72,0x72,0x80]
          vcvtneps2bf16   -512(%rdx){1to4}, %xmm30 {%k7} {z}

// CHECK: vcvtneps2bf16 %ymm29, %xmm30
// CHECK: encoding: [0x62,0x02,0x7e,0x28,0x72,0xf5]
          vcvtneps2bf16 %ymm29, %xmm30

// CHECK: vcvtneps2bf16y  268435456(%rbp,%r14,8), %xmm30 {%k7}
// CHECK: encoding: [0x62,0x22,0x7e,0x2f,0x72,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneps2bf16y  268435456(%rbp,%r14,8), %xmm30 {%k7}

// CHECK: vcvtneps2bf16   (%r9){1to8}, %xmm30
// CHECK: encoding: [0x62,0x42,0x7e,0x38,0x72,0x31]
          vcvtneps2bf16   (%r9){1to8}, %xmm30

// CHECK: vcvtneps2bf16y  4064(%rcx), %xmm30
// CHECK: encoding: [0x62,0x62,0x7e,0x28,0x72,0x71,0x7f]
          vcvtneps2bf16y  4064(%rcx), %xmm30

// CHECK: vcvtneps2bf16   -512(%rdx){1to8}, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x62,0x7e,0xbf,0x72,0x72,0x80]
          vcvtneps2bf16   -512(%rdx){1to8}, %xmm30 {%k7} {z}

// CHECK: vdpbf16ps %ymm28, %ymm29, %ymm30
// CHECK: encoding: [0x62,0x02,0x16,0x20,0x52,0xf4]
          vdpbf16ps %ymm28, %ymm29, %ymm30

// CHECK: vdpbf16ps   268435456(%rbp,%r14,8), %ymm29, %ymm30 {%k7}
// CHECK: encoding: [0x62,0x22,0x16,0x27,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdpbf16ps   268435456(%rbp,%r14,8), %ymm29, %ymm30 {%k7}

// CHECK: vdpbf16ps   (%r9){1to8}, %ymm29, %ymm30
// CHECK: encoding: [0x62,0x42,0x16,0x30,0x52,0x31]
          vdpbf16ps   (%r9){1to8}, %ymm29, %ymm30

// CHECK: vdpbf16ps   4064(%rcx), %ymm29, %ymm30
// CHECK: encoding: [0x62,0x62,0x16,0x20,0x52,0x71,0x7f]
          vdpbf16ps   4064(%rcx), %ymm29, %ymm30

// CHECK: vdpbf16ps   -512(%rdx){1to8}, %ymm29, %ymm30 {%k7} {z}
// CHECK: encoding: [0x62,0x62,0x16,0xb7,0x52,0x72,0x80]
          vdpbf16ps   -512(%rdx){1to8}, %ymm29, %ymm30 {%k7} {z}

// CHECK: vdpbf16ps %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x02,0x16,0x00,0x52,0xf4]
          vdpbf16ps %xmm28, %xmm29, %xmm30

// CHECK: vdpbf16ps   268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x22,0x16,0x07,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdpbf16ps   268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vdpbf16ps   (%r9){1to4}, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x42,0x16,0x10,0x52,0x31]
          vdpbf16ps   (%r9){1to4}, %xmm29, %xmm30

// CHECK: vdpbf16ps   2032(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x62,0x16,0x00,0x52,0x71,0x7f]
          vdpbf16ps   2032(%rcx), %xmm29, %xmm30

// CHECK: vdpbf16ps   -512(%rdx){1to4}, %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x62,0x16,0x97,0x52,0x72,0x80]
          vdpbf16ps   -512(%rdx){1to4}, %xmm29, %xmm30 {%k7} {z}

