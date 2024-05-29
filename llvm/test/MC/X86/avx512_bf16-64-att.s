// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding < %s | FileCheck %s

// CHECK: vcvtne2ps2bf16 %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x02,0x17,0x40,0x72,0xf4]
          vcvtne2ps2bf16 %zmm28, %zmm29, %zmm30

// CHECK: vcvtne2ps2bf16 %zmm28, %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x02,0x17,0x47,0x72,0xf4]
          vcvtne2ps2bf16 %zmm28, %zmm29, %zmm30 {%k7}

// CHECK: vcvtne2ps2bf16 %zmm28, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x02,0x17,0xc7,0x72,0xf4]
          vcvtne2ps2bf16 %zmm28, %zmm29, %zmm30 {%k7} {z}

// CHECK: vcvtne2ps2bf16   (%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x62,0x17,0x40,0x72,0x31]
          vcvtne2ps2bf16   (%rcx), %zmm29, %zmm30

// CHECK: vcvtne2ps2bf16   291(%rax,%r14,8), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x22,0x17,0x40,0x72,0xb4,0xf0,0x23,0x01,0x00,0x00]
          vcvtne2ps2bf16   291(%rax,%r14,8), %zmm29, %zmm30

// CHECK: vcvtne2ps2bf16   268435456(%rax,%r14,8), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x22,0x17,0x40,0x72,0xb4,0xf0,0x00,0x00,0x00,0x10]
          vcvtne2ps2bf16   268435456(%rax,%r14,8), %zmm29, %zmm30

// CHECK: vcvtne2ps2bf16   -64(%rsp), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x62,0x17,0x40,0x72,0x74,0x24,0xff]
          vcvtne2ps2bf16   -64(%rsp), %zmm29, %zmm30

// CHECK: vcvtne2ps2bf16   (%rcx){1to16}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x62,0x17,0x50,0x72,0x31]
          vcvtne2ps2bf16   (%rcx){1to16}, %zmm29, %zmm30

// CHECK: vcvtne2ps2bf16   8128(%rdx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x62,0x17,0x40,0x72,0x72,0x7f]
          vcvtne2ps2bf16   8128(%rdx), %zmm29, %zmm30

// CHECK: vcvtne2ps2bf16   -8192(%rdx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x62,0x17,0x40,0x72,0x72,0x80]
          vcvtne2ps2bf16   -8192(%rdx), %zmm29, %zmm30

// CHECK: vcvtne2ps2bf16   508(%rdx){1to16}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x62,0x17,0x50,0x72,0x72,0x7f]
          vcvtne2ps2bf16   508(%rdx){1to16}, %zmm29, %zmm30

// CHECK: vcvtne2ps2bf16   -512(%rdx){1to16}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x62,0x17,0x50,0x72,0x72,0x80]
          vcvtne2ps2bf16   -512(%rdx){1to16}, %zmm29, %zmm30

// CHECK: vcvtneps2bf16 %zmm29, %ymm30
// CHECK: encoding: [0x62,0x02,0x7e,0x48,0x72,0xf5]
          vcvtneps2bf16 %zmm29, %ymm30

// CHECK: vcvtneps2bf16   268435456(%rbp,%r14,8), %ymm30 {%k7}
// CHECK: encoding: [0x62,0x22,0x7e,0x4f,0x72,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneps2bf16   268435456(%rbp,%r14,8), %ymm30 {%k7}

// CHECK: vcvtneps2bf16   (%r9){1to16}, %ymm30
// CHECK: encoding: [0x62,0x42,0x7e,0x58,0x72,0x31]
          vcvtneps2bf16   (%r9){1to16}, %ymm30

// CHECK: vcvtneps2bf16   8128(%rcx), %ymm30
// CHECK: encoding: [0x62,0x62,0x7e,0x48,0x72,0x71,0x7f]
          vcvtneps2bf16   8128(%rcx), %ymm30

// CHECK: vcvtneps2bf16   -512(%rdx){1to16}, %ymm30 {%k7} {z}
// CHECK: encoding: [0x62,0x62,0x7e,0xdf,0x72,0x72,0x80]
          vcvtneps2bf16   -512(%rdx){1to16}, %ymm30 {%k7} {z}

// CHECK: vdpbf16ps %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x02,0x16,0x40,0x52,0xf4]
          vdpbf16ps %zmm28, %zmm29, %zmm30

// CHECK: vdpbf16ps   268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x22,0x16,0x47,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdpbf16ps   268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vdpbf16ps   (%r9){1to16}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x42,0x16,0x50,0x52,0x31]
          vdpbf16ps   (%r9){1to16}, %zmm29, %zmm30

// CHECK: vdpbf16ps   8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x62,0x16,0x40,0x52,0x71,0x7f]
          vdpbf16ps   8128(%rcx), %zmm29, %zmm30

// CHECK: vdpbf16ps   -512(%rdx){1to16}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x62,0x16,0xd7,0x52,0x72,0x80]
          vdpbf16ps   -512(%rdx){1to16}, %zmm29, %zmm30 {%k7} {z}

