// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vcvtne2ps2bf16 zmm30, zmm29, zmm28
// CHECK: encoding: [0x62,0x02,0x17,0x40,0x72,0xf4]
          vcvtne2ps2bf16 zmm30, zmm29, zmm28

// CHECK: vcvtne2ps2bf16 zmm30 {k7}, zmm29, zmm28
// CHECK: encoding: [0x62,0x02,0x17,0x47,0x72,0xf4]
          vcvtne2ps2bf16 zmm30 {k7}, zmm29, zmm28

// CHECK: vcvtne2ps2bf16 zmm30 {k7} {z}, zmm29, zmm28
// CHECK: encoding: [0x62,0x02,0x17,0xc7,0x72,0xf4]
          vcvtne2ps2bf16 zmm30 {k7} {z}, zmm29, zmm28

// CHECK: vcvtne2ps2bf16 zmm30, zmm29, zmmword ptr [rcx]
// CHECK: encoding: [0x62,0x62,0x17,0x40,0x72,0x31]
          vcvtne2ps2bf16 zmm30, zmm29, zmmword ptr [rcx]

// CHECK: vcvtne2ps2bf16 zmm30, zmm29, zmmword ptr [rax + 8*r14 + 291]
// CHECK: encoding: [0x62,0x22,0x17,0x40,0x72,0xb4,0xf0,0x23,0x01,0x00,0x00]
          vcvtne2ps2bf16 zmm30, zmm29, zmmword ptr [rax + 8*r14 + 291]

// CHECK: vcvtne2ps2bf16 zmm30, zmm29, zmmword ptr [rax + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x22,0x17,0x40,0x72,0xb4,0xf0,0x00,0x00,0x00,0x10]
          vcvtne2ps2bf16 zmm30, zmm29, zmmword ptr [rax + 8*r14 + 268435456]

// CHECK: vcvtne2ps2bf16 zmm30, zmm29, zmmword ptr [rsp - 4]
// CHECK: encoding: [0x62,0x62,0x17,0x40,0x72,0xb4,0x24,0xfc,0xff,0xff,0xff]
          vcvtne2ps2bf16 zmm30, zmm29, zmmword ptr [rsp - 4]

// CHECK: vcvtne2ps2bf16 zmm30, zmm29, dword ptr [rcx]{1to16}
// CHECK: encoding: [0x62,0x62,0x17,0x50,0x72,0x31]
          vcvtne2ps2bf16 zmm30, zmm29, dword ptr [rcx]{1to16}

// CHECK: vcvtne2ps2bf16 zmm30, zmm29, zmmword ptr [rdx + 8128]
// CHECK: encoding: [0x62,0x62,0x17,0x40,0x72,0x72,0x7f]
          vcvtne2ps2bf16 zmm30, zmm29, zmmword ptr [rdx + 8128]

// CHECK: vcvtne2ps2bf16 zmm30, zmm29, zmmword ptr [rdx - 8192]
// CHECK: encoding: [0x62,0x62,0x17,0x40,0x72,0x72,0x80]
          vcvtne2ps2bf16 zmm30, zmm29, zmmword ptr [rdx - 8192]

// CHECK: vcvtne2ps2bf16 zmm30, zmm29, dword ptr [rdx + 508]{1to16}
// CHECK: encoding: [0x62,0x62,0x17,0x50,0x72,0x72,0x7f]
          vcvtne2ps2bf16 zmm30, zmm29, dword ptr [rdx + 508]{1to16}

// CHECK: vcvtne2ps2bf16 zmm30, zmm29, dword ptr [rdx - 512]{1to16}
// CHECK: encoding: [0x62,0x62,0x17,0x50,0x72,0x72,0x80]
          vcvtne2ps2bf16 zmm30, zmm29, dword ptr [rdx - 512]{1to16}

// CHECK: vcvtneps2bf16 ymm30, zmm29
// CHECK: encoding: [0x62,0x02,0x7e,0x48,0x72,0xf5]
          vcvtneps2bf16 ymm30, zmm29

// CHECK: vcvtneps2bf16 ymm30 {k7}, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x22,0x7e,0x4f,0x72,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneps2bf16 ymm30 {k7}, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtneps2bf16 ymm30, dword ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x42,0x7e,0x58,0x72,0x31]
          vcvtneps2bf16 ymm30, dword ptr [r9]{1to16}

// CHECK: vcvtneps2bf16 ymm30, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0x62,0x7e,0x48,0x72,0x71,0x7f]
          vcvtneps2bf16 ymm30, zmmword ptr [rcx + 8128]

// CHECK: vcvtneps2bf16 ymm30 {k7} {z}, dword ptr [rdx - 512]{1to16}
// CHECK: encoding: [0x62,0x62,0x7e,0xdf,0x72,0x72,0x80]
          vcvtneps2bf16 ymm30 {k7} {z}, dword ptr [rdx - 512]{1to16}

// CHECK: vdpbf16ps zmm30, zmm29, zmm28
// CHECK: encoding: [0x62,0x02,0x16,0x40,0x52,0xf4]
          vdpbf16ps zmm30, zmm29, zmm28

// CHECK: vdpbf16ps zmm30 {k7}, zmm29, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x22,0x16,0x47,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdpbf16ps zmm30 {k7}, zmm29, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vdpbf16ps zmm30, zmm29, dword ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x42,0x16,0x50,0x52,0x31]
          vdpbf16ps zmm30, zmm29, dword ptr [r9]{1to16}

// CHECK: vdpbf16ps zmm30, zmm29, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0x62,0x16,0x40,0x52,0x71,0x7f]
          vdpbf16ps zmm30, zmm29, zmmword ptr [rcx + 8128]

// CHECK: vdpbf16ps zmm30 {k7} {z}, zmm29, dword ptr [rdx - 512]{1to16}
// CHECK: encoding: [0x62,0x62,0x16,0xd7,0x52,0x72,0x80]
          vdpbf16ps zmm30 {k7} {z}, zmm29, dword ptr [rdx - 512]{1to16}

