// RUN: llvm-mc -triple i686-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmm4
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0xf4]
          vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmm4

// CHECK: vcvtne2ps2bf16 xmm6 {k7} {z}, xmm5, xmm4
// CHECK: encoding: [0x62,0xf2,0x57,0x8f,0x72,0xf4]
          vcvtne2ps2bf16 xmm6 {k7} {z}, xmm5, xmm4

// CHECK: vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmmword ptr [ecx]
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0x31]
          vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmmword ptr [ecx]

// CHECK: vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmmword ptr [esp + 8*esi + 291]
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0xb4,0xf4,0x23,0x01,0x00,0x00]
          vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmmword ptr [esp + 8*esi + 291]

// CHECK: vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmmword ptr [esp - 4]
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0xb4,0x24,0xfc,0xff,0xff,0xff]
          vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmmword ptr [esp - 4]

// CHECK: vcvtne2ps2bf16 xmm6 {k7}, xmm5, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x57,0x1f,0x72,0x30]
          vcvtne2ps2bf16 xmm6 {k7}, xmm5, dword ptr [eax]{1to4}

// CHECK: vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmmword ptr [edx + 2032]
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0x72,0x7f]
          vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmmword ptr [edx + 2032]

// CHECK: vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmmword ptr [edx - 2048]
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0x72,0x80]
          vcvtne2ps2bf16 xmm6 {k7}, xmm5, xmmword ptr [edx - 2048]

// CHECK: vcvtne2ps2bf16 xmm6 {k7}, xmm5, dword ptr [edx + 508]{1to4}
// CHECK: encoding: [0x62,0xf2,0x57,0x1f,0x72,0x72,0x7f]
          vcvtne2ps2bf16 xmm6 {k7}, xmm5, dword ptr [edx + 508]{1to4}

// CHECK: vcvtne2ps2bf16 xmm6 {k7}, xmm5, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x57,0x1f,0x72,0x72,0x80]
          vcvtne2ps2bf16 xmm6 {k7}, xmm5, dword ptr [edx - 512]{1to4}

// CHECK: vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymm4
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0xf4]
          vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymm4

// CHECK: vcvtne2ps2bf16 ymm6 {k7} {z}, ymm5, ymm4
// CHECK: encoding: [0x62,0xf2,0x57,0xaf,0x72,0xf4]
          vcvtne2ps2bf16 ymm6 {k7} {z}, ymm5, ymm4

// CHECK: vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymmword ptr [ecx]
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0x31]
          vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymmword ptr [ecx]

// CHECK: vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymmword ptr [esp + 8*esi + 291]
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0xb4,0xf4,0x23,0x01,0x00,0x00]
          vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymmword ptr [esp + 8*esi + 291]

// CHECK: vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymmword ptr [esp - 4]
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0xb4,0x24,0xfc,0xff,0xff,0xff]
          vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymmword ptr [esp - 4]

// CHECK: vcvtne2ps2bf16 ymm6 {k7}, ymm5, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x57,0x3f,0x72,0x30]
          vcvtne2ps2bf16 ymm6 {k7}, ymm5, dword ptr [eax]{1to8}

// CHECK: vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymmword ptr [edx + 4064]
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0x72,0x7f]
          vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymmword ptr [edx + 4064]

// CHECK: vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymmword ptr [edx - 4096]
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0x72,0x80]
          vcvtne2ps2bf16 ymm6 {k7}, ymm5, ymmword ptr [edx - 4096]

// CHECK: vcvtne2ps2bf16 ymm6 {k7}, ymm5, dword ptr [edx + 508]{1to8}
// CHECK: encoding: [0x62,0xf2,0x57,0x3f,0x72,0x72,0x7f]
          vcvtne2ps2bf16 ymm6 {k7}, ymm5, dword ptr [edx + 508]{1to8}

// CHECK: vcvtne2ps2bf16 ymm6 {k7}, ymm5, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x57,0x3f,0x72,0x72,0x80]
          vcvtne2ps2bf16 ymm6 {k7}, ymm5, dword ptr [edx - 512]{1to8}

// CHECK: vcvtneps2bf16 xmm6, xmm5
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x72,0xf5]
          vcvtneps2bf16 xmm6, xmm5

// CHECK: vcvtneps2bf16 xmm6 {k7}, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x7e,0x0f,0x72,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtneps2bf16 xmm6 {k7}, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtneps2bf16 xmm6, dword ptr [ecx]{1to4}
// CHECK: encoding: [0x62,0xf2,0x7e,0x18,0x72,0x31]
          vcvtneps2bf16 xmm6, dword ptr [ecx]{1to4}

// CHECK: vcvtneps2bf16 xmm6, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x72,0x71,0x7f]
          vcvtneps2bf16 xmm6, xmmword ptr [ecx + 2032]

// CHECK: vcvtneps2bf16 xmm6 {k7} {z}, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x7e,0x9f,0x72,0x72,0x80]
          vcvtneps2bf16 xmm6 {k7} {z}, dword ptr [edx - 512]{1to4}

// CHECK: vcvtneps2bf16 xmm6, ymm5
// CHECK: encoding: [0x62,0xf2,0x7e,0x28,0x72,0xf5]
          vcvtneps2bf16 xmm6, ymm5

// CHECK: vcvtneps2bf16 xmm6 {k7}, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x7e,0x2f,0x72,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtneps2bf16 xmm6 {k7}, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtneps2bf16 xmm6, dword ptr [ecx]{1to8}
// CHECK: encoding: [0x62,0xf2,0x7e,0x38,0x72,0x31]
          vcvtneps2bf16 xmm6, dword ptr [ecx]{1to8}

// CHECK: vcvtneps2bf16 xmm6, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x7e,0x28,0x72,0x71,0x7f]
          vcvtneps2bf16 xmm6, ymmword ptr [ecx + 4064]

// CHECK: vcvtneps2bf16 xmm6 {k7} {z}, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x7e,0xbf,0x72,0x72,0x80]
          vcvtneps2bf16 xmm6 {k7} {z}, dword ptr [edx - 512]{1to8}

// CHECK: vdpbf16ps ymm6, ymm5, ymm4
// CHECK: encoding: [0x62,0xf2,0x56,0x28,0x52,0xf4]
          vdpbf16ps ymm6, ymm5, ymm4

// CHECK: vdpbf16ps ymm6 {k7}, ymm5, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x56,0x2f,0x52,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vdpbf16ps ymm6 {k7}, ymm5, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vdpbf16ps ymm6, ymm5, dword ptr [ecx]{1to8}
// CHECK: encoding: [0x62,0xf2,0x56,0x38,0x52,0x31]
          vdpbf16ps ymm6, ymm5, dword ptr [ecx]{1to8}

// CHECK: vdpbf16ps ymm6, ymm5, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x56,0x28,0x52,0x71,0x7f]
          vdpbf16ps ymm6, ymm5, ymmword ptr [ecx + 4064]

// CHECK: vdpbf16ps ymm6 {k7} {z}, ymm5, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x56,0xbf,0x52,0x72,0x80]
          vdpbf16ps ymm6 {k7} {z}, ymm5, dword ptr [edx - 512]{1to8}

// CHECK: vdpbf16ps xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf2,0x56,0x08,0x52,0xf4]
          vdpbf16ps xmm6, xmm5, xmm4

// CHECK: vdpbf16ps xmm6 {k7}, xmm5, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x56,0x0f,0x52,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vdpbf16ps xmm6 {k7}, xmm5, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vdpbf16ps xmm6, xmm5, dword ptr [ecx]{1to4}
// CHECK: encoding: [0x62,0xf2,0x56,0x18,0x52,0x31]
          vdpbf16ps xmm6, xmm5, dword ptr [ecx]{1to4}

// CHECK: vdpbf16ps xmm6, xmm5, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x56,0x08,0x52,0x71,0x7f]
          vdpbf16ps xmm6, xmm5, xmmword ptr [ecx + 2032]

// CHECK: vdpbf16ps xmm6 {k7} {z}, xmm5, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x56,0x9f,0x52,0x72,0x80]
          vdpbf16ps xmm6 {k7} {z}, xmm5, dword ptr [edx - 512]{1to4}

