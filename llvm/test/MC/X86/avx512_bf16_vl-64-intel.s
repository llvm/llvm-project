// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vcvtne2ps2bf16 xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x02,0x17,0x00,0x72,0xf4]
          vcvtne2ps2bf16 xmm30, xmm29, xmm28

// CHECK: vcvtne2ps2bf16 xmm30 {k7}, xmm29, xmm28
// CHECK: encoding: [0x62,0x02,0x17,0x07,0x72,0xf4]
          vcvtne2ps2bf16 xmm30 {k7}, xmm29, xmm28

// CHECK: vcvtne2ps2bf16 xmm30 {k7} {z}, xmm29, xmm28
// CHECK: encoding: [0x62,0x02,0x17,0x87,0x72,0xf4]
          vcvtne2ps2bf16 xmm30 {k7} {z}, xmm29, xmm28

// CHECK: vcvtne2ps2bf16 xmm30, xmm29, xmmword ptr [rcx]
// CHECK: encoding: [0x62,0x62,0x17,0x00,0x72,0x31]
          vcvtne2ps2bf16 xmm30, xmm29, xmmword ptr [rcx]

// CHECK: vcvtne2ps2bf16 xmm30, xmm29, xmmword ptr [rax + 8*r14 + 291]
// CHECK: encoding: [0x62,0x22,0x17,0x00,0x72,0xb4,0xf0,0x23,0x01,0x00,0x00]
          vcvtne2ps2bf16 xmm30, xmm29, xmmword ptr [rax + 8*r14 + 291]

// CHECK: vcvtne2ps2bf16 xmm30, xmm29, xmmword ptr [rax + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x22,0x17,0x00,0x72,0xb4,0xf0,0x00,0x00,0x00,0x10]
          vcvtne2ps2bf16 xmm30, xmm29, xmmword ptr [rax + 8*r14 + 268435456]

// CHECK: vcvtne2ps2bf16 xmm30, xmm29, xmmword ptr [rsp - 4]
// CHECK: encoding: [0x62,0x62,0x17,0x00,0x72,0xb4,0x24,0xfc,0xff,0xff,0xff]
          vcvtne2ps2bf16 xmm30, xmm29, xmmword ptr [rsp - 4]

// CHECK: vcvtne2ps2bf16 xmm30, xmm29, dword ptr [rcx]{1to4}
// CHECK: encoding: [0x62,0x62,0x17,0x10,0x72,0x31]
          vcvtne2ps2bf16 xmm30, xmm29, dword ptr [rcx]{1to4}

// CHECK: vcvtne2ps2bf16 xmm30, xmm29, xmmword ptr [rdx + 2032]
// CHECK: encoding: [0x62,0x62,0x17,0x00,0x72,0x72,0x7f]
          vcvtne2ps2bf16 xmm30, xmm29, xmmword ptr [rdx + 2032]

// CHECK: vcvtne2ps2bf16 xmm30, xmm29, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0x62,0x62,0x17,0x00,0x72,0x72,0x80]
          vcvtne2ps2bf16 xmm30, xmm29, xmmword ptr [rdx - 2048]

// CHECK: vcvtne2ps2bf16 xmm30, xmm29, dword ptr [rdx + 508]{1to4}
// CHECK: encoding: [0x62,0x62,0x17,0x10,0x72,0x72,0x7f]
          vcvtne2ps2bf16 xmm30, xmm29, dword ptr [rdx + 508]{1to4}

// CHECK: vcvtne2ps2bf16 xmm30, xmm29, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0x62,0x17,0x10,0x72,0x72,0x80]
          vcvtne2ps2bf16 xmm30, xmm29, dword ptr [rdx - 512]{1to4}

// CHECK: vcvtne2ps2bf16 ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x02,0x17,0x20,0x72,0xf4]
          vcvtne2ps2bf16 ymm30, ymm29, ymm28

// CHECK: vcvtne2ps2bf16 ymm30 {k7}, ymm29, ymm28
// CHECK: encoding: [0x62,0x02,0x17,0x27,0x72,0xf4]
          vcvtne2ps2bf16 ymm30 {k7}, ymm29, ymm28

// CHECK: vcvtne2ps2bf16 ymm30 {k7} {z}, ymm29, ymm28
// CHECK: encoding: [0x62,0x02,0x17,0xa7,0x72,0xf4]
          vcvtne2ps2bf16 ymm30 {k7} {z}, ymm29, ymm28

// CHECK: vcvtne2ps2bf16 ymm30, ymm29, ymmword ptr [rcx]
// CHECK: encoding: [0x62,0x62,0x17,0x20,0x72,0x31]
          vcvtne2ps2bf16 ymm30, ymm29, ymmword ptr [rcx]

// CHECK: vcvtne2ps2bf16 ymm30, ymm29, ymmword ptr [rax + 8*r14 + 291]
// CHECK: encoding: [0x62,0x22,0x17,0x20,0x72,0xb4,0xf0,0x23,0x01,0x00,0x00]
          vcvtne2ps2bf16 ymm30, ymm29, ymmword ptr [rax + 8*r14 + 291]

// CHECK: vcvtne2ps2bf16 ymm30, ymm29, ymmword ptr [rax + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x22,0x17,0x20,0x72,0xb4,0xf0,0x00,0x00,0x00,0x10]
          vcvtne2ps2bf16 ymm30, ymm29, ymmword ptr [rax + 8*r14 + 268435456]

// CHECK: vcvtne2ps2bf16 ymm30, ymm29, ymmword ptr [rsp - 4]
// CHECK: encoding: [0x62,0x62,0x17,0x20,0x72,0xb4,0x24,0xfc,0xff,0xff,0xff]
          vcvtne2ps2bf16 ymm30, ymm29, ymmword ptr [rsp - 4]

// CHECK: vcvtne2ps2bf16 ymm30, ymm29, dword ptr [rcx]{1to8}
// CHECK: encoding: [0x62,0x62,0x17,0x30,0x72,0x31]
          vcvtne2ps2bf16 ymm30, ymm29, dword ptr [rcx]{1to8}

// CHECK: vcvtne2ps2bf16 ymm30, ymm29, ymmword ptr [rdx + 4064]
// CHECK: encoding: [0x62,0x62,0x17,0x20,0x72,0x72,0x7f]
          vcvtne2ps2bf16 ymm30, ymm29, ymmword ptr [rdx + 4064]

// CHECK: vcvtne2ps2bf16 ymm30, ymm29, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0x62,0x62,0x17,0x20,0x72,0x72,0x80]
          vcvtne2ps2bf16 ymm30, ymm29, ymmword ptr [rdx - 4096]

// CHECK: vcvtne2ps2bf16 ymm30, ymm29, dword ptr [rdx + 508]{1to8}
// CHECK: encoding: [0x62,0x62,0x17,0x30,0x72,0x72,0x7f]
          vcvtne2ps2bf16 ymm30, ymm29, dword ptr [rdx + 508]{1to8}

// CHECK: vcvtne2ps2bf16 ymm30, ymm29, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0x62,0x17,0x30,0x72,0x72,0x80]
          vcvtne2ps2bf16 ymm30, ymm29, dword ptr [rdx - 512]{1to8}

// CHECK: vcvtneps2bf16 xmm30, xmm29
// CHECK: encoding: [0x62,0x02,0x7e,0x08,0x72,0xf5]
          vcvtneps2bf16 xmm30, xmm29

// CHECK: vcvtneps2bf16 xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x22,0x7e,0x0f,0x72,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneps2bf16 xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtneps2bf16 xmm30, dword ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x42,0x7e,0x18,0x72,0x31]
          vcvtneps2bf16 xmm30, dword ptr [r9]{1to4}

// CHECK: vcvtneps2bf16 xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x62,0x7e,0x08,0x72,0x71,0x7f]
          vcvtneps2bf16 xmm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtneps2bf16 xmm30 {k7} {z}, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0x62,0x7e,0x9f,0x72,0x72,0x80]
          vcvtneps2bf16 xmm30 {k7} {z}, dword ptr [rdx - 512]{1to4}

// CHECK: vcvtneps2bf16 xmm30, ymm29
// CHECK: encoding: [0x62,0x02,0x7e,0x28,0x72,0xf5]
          vcvtneps2bf16 xmm30, ymm29

// CHECK: vcvtneps2bf16 xmm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x22,0x7e,0x2f,0x72,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneps2bf16 xmm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtneps2bf16 xmm30, dword ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x42,0x7e,0x38,0x72,0x31]
          vcvtneps2bf16 xmm30, dword ptr [r9]{1to8}

// CHECK: vcvtneps2bf16 xmm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x62,0x7e,0x28,0x72,0x71,0x7f]
          vcvtneps2bf16 xmm30, ymmword ptr [rcx + 4064]

// CHECK: vcvtneps2bf16 xmm30 {k7} {z}, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0x62,0x7e,0xbf,0x72,0x72,0x80]
          vcvtneps2bf16 xmm30 {k7} {z}, dword ptr [rdx - 512]{1to8}

// CHECK: vdpbf16ps ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x02,0x16,0x20,0x52,0xf4]
          vdpbf16ps ymm30, ymm29, ymm28

// CHECK: vdpbf16ps ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x22,0x16,0x27,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdpbf16ps ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vdpbf16ps ymm30, ymm29, dword ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x42,0x16,0x30,0x52,0x31]
          vdpbf16ps ymm30, ymm29, dword ptr [r9]{1to8}

// CHECK: vdpbf16ps ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x62,0x16,0x20,0x52,0x71,0x7f]
          vdpbf16ps ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vdpbf16ps ymm30 {k7} {z}, ymm29, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0x62,0x16,0xb7,0x52,0x72,0x80]
          vdpbf16ps ymm30 {k7} {z}, ymm29, dword ptr [rdx - 512]{1to8}

// CHECK: vdpbf16ps xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x02,0x16,0x00,0x52,0xf4]
          vdpbf16ps xmm30, xmm29, xmm28

// CHECK: vdpbf16ps xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x22,0x16,0x07,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdpbf16ps xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vdpbf16ps xmm30, xmm29, dword ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x42,0x16,0x10,0x52,0x31]
          vdpbf16ps xmm30, xmm29, dword ptr [r9]{1to4}

// CHECK: vdpbf16ps xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x62,0x16,0x00,0x52,0x71,0x7f]
          vdpbf16ps xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vdpbf16ps xmm30 {k7} {z}, xmm29, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0x62,0x16,0x97,0x52,0x72,0x80]
          vdpbf16ps xmm30 {k7} {z}, xmm29, dword ptr [rdx - 512]{1to4}

