// RUN: llvm-mc -triple i386 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vaddbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x58,0xd4]
          vaddbf16 ymm2, ymm3, ymm4

// CHECK: vaddbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x58,0xd4]
          vaddbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vaddbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x58,0xd4]
          vaddbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vaddbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x58,0xd4]
          vaddbf16 zmm2, zmm3, zmm4

// CHECK: vaddbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x58,0xd4]
          vaddbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vaddbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x58,0xd4]
          vaddbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vaddbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x58,0xd4]
          vaddbf16 xmm2, xmm3, xmm4

// CHECK: vaddbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x58,0xd4]
          vaddbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vaddbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x58,0xd4]
          vaddbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vaddbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x58,0x94,0xf4,0x00,0x00,0x00,0x10]
          vaddbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vaddbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x58,0x94,0x87,0x23,0x01,0x00,0x00]
          vaddbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vaddbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x58,0x10]
          vaddbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vaddbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x58,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vaddbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vaddbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x58,0x51,0x7f]
          vaddbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vaddbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x58,0x52,0x80]
          vaddbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vaddbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x58,0x94,0xf4,0x00,0x00,0x00,0x10]
          vaddbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vaddbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x58,0x94,0x87,0x23,0x01,0x00,0x00]
          vaddbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vaddbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x58,0x10]
          vaddbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vaddbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x58,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vaddbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vaddbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x58,0x51,0x7f]
          vaddbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vaddbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x58,0x52,0x80]
          vaddbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vaddbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x58,0x94,0xf4,0x00,0x00,0x00,0x10]
          vaddbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vaddbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x58,0x94,0x87,0x23,0x01,0x00,0x00]
          vaddbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vaddbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x58,0x10]
          vaddbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vaddbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x58,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vaddbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vaddbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x58,0x51,0x7f]
          vaddbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vaddbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x58,0x52,0x80]
          vaddbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vcmpbf16 k5, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0xc2,0xec,0x7b]
          vcmpbf16 k5, ymm3, ymm4, 123

// CHECK: vcmpbf16 k5 {k7}, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x2f,0xc2,0xec,0x7b]
          vcmpbf16 k5 {k7}, ymm3, ymm4, 123

// CHECK: vcmpbf16 k5, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0xc2,0xec,0x7b]
          vcmpbf16 k5, xmm3, xmm4, 123

// CHECK: vcmpbf16 k5 {k7}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x0f,0xc2,0xec,0x7b]
          vcmpbf16 k5 {k7}, xmm3, xmm4, 123

// CHECK: vcmpbf16 k5, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0xc2,0xec,0x7b]
          vcmpbf16 k5, zmm3, zmm4, 123

// CHECK: vcmpbf16 k5 {k7}, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x4f,0xc2,0xec,0x7b]
          vcmpbf16 k5 {k7}, zmm3, zmm4, 123

// CHECK: vcmpbf16 k5, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vcmpbf16 k5, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vcmpbf16 k5 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x4f,0xc2,0xac,0x87,0x23,0x01,0x00,0x00,0x7b]
          vcmpbf16 k5 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vcmpbf16 k5, zmm3, word ptr [eax]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x58,0xc2,0x28,0x7b]
          vcmpbf16 k5, zmm3, word ptr [eax]{1to32}, 123

// CHECK: vcmpbf16 k5, zmm3, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0xc2,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vcmpbf16 k5, zmm3, zmmword ptr [2*ebp - 2048], 123

// CHECK: vcmpbf16 k5 {k7}, zmm3, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x4f,0xc2,0x69,0x7f,0x7b]
          vcmpbf16 k5 {k7}, zmm3, zmmword ptr [ecx + 8128], 123

// CHECK: vcmpbf16 k5 {k7}, zmm3, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x5f,0xc2,0x6a,0x80,0x7b]
          vcmpbf16 k5 {k7}, zmm3, word ptr [edx - 256]{1to32}, 123

// CHECK: vcmpbf16 k5, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vcmpbf16 k5, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vcmpbf16 k5 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x0f,0xc2,0xac,0x87,0x23,0x01,0x00,0x00,0x7b]
          vcmpbf16 k5 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vcmpbf16 k5, xmm3, word ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x18,0xc2,0x28,0x7b]
          vcmpbf16 k5, xmm3, word ptr [eax]{1to8}, 123

// CHECK: vcmpbf16 k5, xmm3, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0xc2,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vcmpbf16 k5, xmm3, xmmword ptr [2*ebp - 512], 123

// CHECK: vcmpbf16 k5 {k7}, xmm3, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x0f,0xc2,0x69,0x7f,0x7b]
          vcmpbf16 k5 {k7}, xmm3, xmmword ptr [ecx + 2032], 123

// CHECK: vcmpbf16 k5 {k7}, xmm3, word ptr [edx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x1f,0xc2,0x6a,0x80,0x7b]
          vcmpbf16 k5 {k7}, xmm3, word ptr [edx - 256]{1to8}, 123

// CHECK: vcmpbf16 k5, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vcmpbf16 k5, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vcmpbf16 k5 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x2f,0xc2,0xac,0x87,0x23,0x01,0x00,0x00,0x7b]
          vcmpbf16 k5 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vcmpbf16 k5, ymm3, word ptr [eax]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x38,0xc2,0x28,0x7b]
          vcmpbf16 k5, ymm3, word ptr [eax]{1to16}, 123

// CHECK: vcmpbf16 k5, ymm3, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0xc2,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vcmpbf16 k5, ymm3, ymmword ptr [2*ebp - 1024], 123

// CHECK: vcmpbf16 k5 {k7}, ymm3, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x2f,0xc2,0x69,0x7f,0x7b]
          vcmpbf16 k5 {k7}, ymm3, ymmword ptr [ecx + 4064], 123

// CHECK: vcmpbf16 k5 {k7}, ymm3, word ptr [edx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x3f,0xc2,0x6a,0x80,0x7b]
          vcmpbf16 k5 {k7}, ymm3, word ptr [edx - 256]{1to16}, 123

// CHECK: vcomisbf16 xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0xd3]
          vcomisbf16 xmm2, xmm3

// CHECK: vcomisbf16 xmm2, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcomisbf16 xmm2, word ptr [esp + 8*esi + 268435456]

// CHECK: vcomisbf16 xmm2, word ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x94,0x87,0x23,0x01,0x00,0x00]
          vcomisbf16 xmm2, word ptr [edi + 4*eax + 291]

// CHECK: vcomisbf16 xmm2, word ptr [eax]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x10]
          vcomisbf16 xmm2, word ptr [eax]

// CHECK: vcomisbf16 xmm2, word ptr [2*ebp - 64]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x14,0x6d,0xc0,0xff,0xff,0xff]
          vcomisbf16 xmm2, word ptr [2*ebp - 64]

// CHECK: vcomisbf16 xmm2, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x51,0x7f]
          vcomisbf16 xmm2, word ptr [ecx + 254]

// CHECK: vcomisbf16 xmm2, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x52,0x80]
          vcomisbf16 xmm2, word ptr [edx - 256]

// CHECK: vdivbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5e,0xd4]
          vdivbf16 ymm2, ymm3, ymm4

// CHECK: vdivbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5e,0xd4]
          vdivbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vdivbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5e,0xd4]
          vdivbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vdivbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5e,0xd4]
          vdivbf16 zmm2, zmm3, zmm4

// CHECK: vdivbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5e,0xd4]
          vdivbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vdivbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5e,0xd4]
          vdivbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vdivbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5e,0xd4]
          vdivbf16 xmm2, xmm3, xmm4

// CHECK: vdivbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5e,0xd4]
          vdivbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vdivbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5e,0xd4]
          vdivbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vdivbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vdivbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vdivbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5e,0x94,0x87,0x23,0x01,0x00,0x00]
          vdivbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vdivbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x5e,0x10]
          vdivbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vdivbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5e,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vdivbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vdivbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5e,0x51,0x7f]
          vdivbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vdivbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x5e,0x52,0x80]
          vdivbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vdivbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vdivbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vdivbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5e,0x94,0x87,0x23,0x01,0x00,0x00]
          vdivbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vdivbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x5e,0x10]
          vdivbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vdivbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5e,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vdivbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vdivbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5e,0x51,0x7f]
          vdivbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vdivbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x5e,0x52,0x80]
          vdivbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vdivbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vdivbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vdivbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5e,0x94,0x87,0x23,0x01,0x00,0x00]
          vdivbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vdivbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x5e,0x10]
          vdivbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vdivbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5e,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vdivbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vdivbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5e,0x51,0x7f]
          vdivbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vdivbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x5e,0x52,0x80]
          vdivbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfmadd132bf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x98,0xd4]
          vfmadd132bf16 ymm2, ymm3, ymm4

// CHECK: vfmadd132bf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x98,0xd4]
          vfmadd132bf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfmadd132bf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x98,0xd4]
          vfmadd132bf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfmadd132bf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x98,0xd4]
          vfmadd132bf16 zmm2, zmm3, zmm4

// CHECK: vfmadd132bf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x98,0xd4]
          vfmadd132bf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfmadd132bf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x98,0xd4]
          vfmadd132bf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfmadd132bf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x98,0xd4]
          vfmadd132bf16 xmm2, xmm3, xmm4

// CHECK: vfmadd132bf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x98,0xd4]
          vfmadd132bf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfmadd132bf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x98,0xd4]
          vfmadd132bf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfmadd132bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x98,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd132bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd132bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x98,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd132bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd132bf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x98,0x10]
          vfmadd132bf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfmadd132bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x98,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd132bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfmadd132bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x98,0x51,0x7f]
          vfmadd132bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfmadd132bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x98,0x52,0x80]
          vfmadd132bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfmadd132bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x98,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd132bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd132bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x98,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd132bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd132bf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x98,0x10]
          vfmadd132bf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfmadd132bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x98,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd132bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfmadd132bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x98,0x51,0x7f]
          vfmadd132bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfmadd132bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x98,0x52,0x80]
          vfmadd132bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfmadd132bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x98,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd132bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd132bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x98,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd132bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd132bf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x98,0x10]
          vfmadd132bf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfmadd132bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x98,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd132bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfmadd132bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x98,0x51,0x7f]
          vfmadd132bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfmadd132bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x98,0x52,0x80]
          vfmadd132bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfmadd213bf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xa8,0xd4]
          vfmadd213bf16 ymm2, ymm3, ymm4

// CHECK: vfmadd213bf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xa8,0xd4]
          vfmadd213bf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfmadd213bf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xa8,0xd4]
          vfmadd213bf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfmadd213bf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xa8,0xd4]
          vfmadd213bf16 zmm2, zmm3, zmm4

// CHECK: vfmadd213bf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xa8,0xd4]
          vfmadd213bf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfmadd213bf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xa8,0xd4]
          vfmadd213bf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfmadd213bf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xa8,0xd4]
          vfmadd213bf16 xmm2, xmm3, xmm4

// CHECK: vfmadd213bf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xa8,0xd4]
          vfmadd213bf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfmadd213bf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xa8,0xd4]
          vfmadd213bf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfmadd213bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xa8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd213bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd213bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xa8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd213bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd213bf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xa8,0x10]
          vfmadd213bf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfmadd213bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xa8,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd213bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfmadd213bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xa8,0x51,0x7f]
          vfmadd213bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfmadd213bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xa8,0x52,0x80]
          vfmadd213bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfmadd213bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xa8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd213bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd213bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xa8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd213bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd213bf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xa8,0x10]
          vfmadd213bf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfmadd213bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xa8,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd213bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfmadd213bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xa8,0x51,0x7f]
          vfmadd213bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfmadd213bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xa8,0x52,0x80]
          vfmadd213bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfmadd213bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xa8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd213bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd213bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xa8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd213bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd213bf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xa8,0x10]
          vfmadd213bf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfmadd213bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xa8,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd213bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfmadd213bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xa8,0x51,0x7f]
          vfmadd213bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfmadd213bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xa8,0x52,0x80]
          vfmadd213bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfmadd231bf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xb8,0xd4]
          vfmadd231bf16 ymm2, ymm3, ymm4

// CHECK: vfmadd231bf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xb8,0xd4]
          vfmadd231bf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfmadd231bf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xb8,0xd4]
          vfmadd231bf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfmadd231bf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xb8,0xd4]
          vfmadd231bf16 zmm2, zmm3, zmm4

// CHECK: vfmadd231bf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xb8,0xd4]
          vfmadd231bf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfmadd231bf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xb8,0xd4]
          vfmadd231bf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfmadd231bf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xb8,0xd4]
          vfmadd231bf16 xmm2, xmm3, xmm4

// CHECK: vfmadd231bf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xb8,0xd4]
          vfmadd231bf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfmadd231bf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xb8,0xd4]
          vfmadd231bf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfmadd231bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xb8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd231bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd231bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xb8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd231bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd231bf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xb8,0x10]
          vfmadd231bf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfmadd231bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xb8,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd231bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfmadd231bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xb8,0x51,0x7f]
          vfmadd231bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfmadd231bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xb8,0x52,0x80]
          vfmadd231bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfmadd231bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xb8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd231bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd231bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xb8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd231bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd231bf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xb8,0x10]
          vfmadd231bf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfmadd231bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xb8,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd231bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfmadd231bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xb8,0x51,0x7f]
          vfmadd231bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfmadd231bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xb8,0x52,0x80]
          vfmadd231bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfmadd231bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xb8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd231bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd231bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xb8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd231bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd231bf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xb8,0x10]
          vfmadd231bf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfmadd231bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xb8,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd231bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfmadd231bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xb8,0x51,0x7f]
          vfmadd231bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfmadd231bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xb8,0x52,0x80]
          vfmadd231bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfmsub132bf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9a,0xd4]
          vfmsub132bf16 ymm2, ymm3, ymm4

// CHECK: vfmsub132bf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9a,0xd4]
          vfmsub132bf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfmsub132bf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9a,0xd4]
          vfmsub132bf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfmsub132bf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9a,0xd4]
          vfmsub132bf16 zmm2, zmm3, zmm4

// CHECK: vfmsub132bf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9a,0xd4]
          vfmsub132bf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfmsub132bf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9a,0xd4]
          vfmsub132bf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfmsub132bf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9a,0xd4]
          vfmsub132bf16 xmm2, xmm3, xmm4

// CHECK: vfmsub132bf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9a,0xd4]
          vfmsub132bf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfmsub132bf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9a,0xd4]
          vfmsub132bf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfmsub132bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub132bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub132bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9a,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub132bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub132bf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x9a,0x10]
          vfmsub132bf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfmsub132bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9a,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub132bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfmsub132bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9a,0x51,0x7f]
          vfmsub132bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfmsub132bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x9a,0x52,0x80]
          vfmsub132bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfmsub132bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub132bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub132bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9a,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub132bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub132bf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x9a,0x10]
          vfmsub132bf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfmsub132bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9a,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub132bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfmsub132bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9a,0x51,0x7f]
          vfmsub132bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfmsub132bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x9a,0x52,0x80]
          vfmsub132bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfmsub132bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub132bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub132bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9a,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub132bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub132bf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x9a,0x10]
          vfmsub132bf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfmsub132bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9a,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub132bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfmsub132bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9a,0x51,0x7f]
          vfmsub132bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfmsub132bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x9a,0x52,0x80]
          vfmsub132bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfmsub213bf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xaa,0xd4]
          vfmsub213bf16 ymm2, ymm3, ymm4

// CHECK: vfmsub213bf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xaa,0xd4]
          vfmsub213bf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfmsub213bf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xaa,0xd4]
          vfmsub213bf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfmsub213bf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xaa,0xd4]
          vfmsub213bf16 zmm2, zmm3, zmm4

// CHECK: vfmsub213bf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xaa,0xd4]
          vfmsub213bf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfmsub213bf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xaa,0xd4]
          vfmsub213bf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfmsub213bf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xaa,0xd4]
          vfmsub213bf16 xmm2, xmm3, xmm4

// CHECK: vfmsub213bf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xaa,0xd4]
          vfmsub213bf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfmsub213bf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xaa,0xd4]
          vfmsub213bf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfmsub213bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xaa,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub213bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub213bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xaa,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub213bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub213bf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xaa,0x10]
          vfmsub213bf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfmsub213bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xaa,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub213bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfmsub213bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xaa,0x51,0x7f]
          vfmsub213bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfmsub213bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xaa,0x52,0x80]
          vfmsub213bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfmsub213bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xaa,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub213bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub213bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xaa,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub213bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub213bf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xaa,0x10]
          vfmsub213bf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfmsub213bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xaa,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub213bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfmsub213bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xaa,0x51,0x7f]
          vfmsub213bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfmsub213bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xaa,0x52,0x80]
          vfmsub213bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfmsub213bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xaa,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub213bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub213bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xaa,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub213bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub213bf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xaa,0x10]
          vfmsub213bf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfmsub213bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xaa,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub213bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfmsub213bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xaa,0x51,0x7f]
          vfmsub213bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfmsub213bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xaa,0x52,0x80]
          vfmsub213bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfmsub231bf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xba,0xd4]
          vfmsub231bf16 ymm2, ymm3, ymm4

// CHECK: vfmsub231bf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xba,0xd4]
          vfmsub231bf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfmsub231bf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xba,0xd4]
          vfmsub231bf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfmsub231bf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xba,0xd4]
          vfmsub231bf16 zmm2, zmm3, zmm4

// CHECK: vfmsub231bf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xba,0xd4]
          vfmsub231bf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfmsub231bf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xba,0xd4]
          vfmsub231bf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfmsub231bf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xba,0xd4]
          vfmsub231bf16 xmm2, xmm3, xmm4

// CHECK: vfmsub231bf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xba,0xd4]
          vfmsub231bf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfmsub231bf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xba,0xd4]
          vfmsub231bf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfmsub231bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xba,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub231bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub231bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xba,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub231bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub231bf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xba,0x10]
          vfmsub231bf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfmsub231bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xba,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub231bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfmsub231bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xba,0x51,0x7f]
          vfmsub231bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfmsub231bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xba,0x52,0x80]
          vfmsub231bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfmsub231bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xba,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub231bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub231bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xba,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub231bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub231bf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xba,0x10]
          vfmsub231bf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfmsub231bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xba,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub231bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfmsub231bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xba,0x51,0x7f]
          vfmsub231bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfmsub231bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xba,0x52,0x80]
          vfmsub231bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfmsub231bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xba,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub231bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub231bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xba,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub231bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub231bf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xba,0x10]
          vfmsub231bf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfmsub231bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xba,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub231bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfmsub231bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xba,0x51,0x7f]
          vfmsub231bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfmsub231bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xba,0x52,0x80]
          vfmsub231bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfnmadd132bf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9c,0xd4]
          vfnmadd132bf16 ymm2, ymm3, ymm4

// CHECK: vfnmadd132bf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9c,0xd4]
          vfnmadd132bf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfnmadd132bf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9c,0xd4]
          vfnmadd132bf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfnmadd132bf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9c,0xd4]
          vfnmadd132bf16 zmm2, zmm3, zmm4

// CHECK: vfnmadd132bf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9c,0xd4]
          vfnmadd132bf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfnmadd132bf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9c,0xd4]
          vfnmadd132bf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfnmadd132bf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9c,0xd4]
          vfnmadd132bf16 xmm2, xmm3, xmm4

// CHECK: vfnmadd132bf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9c,0xd4]
          vfnmadd132bf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfnmadd132bf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9c,0xd4]
          vfnmadd132bf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfnmadd132bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd132bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd132bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9c,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd132bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd132bf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x9c,0x10]
          vfnmadd132bf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfnmadd132bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd132bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfnmadd132bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9c,0x51,0x7f]
          vfnmadd132bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfnmadd132bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x9c,0x52,0x80]
          vfnmadd132bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfnmadd132bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd132bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd132bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9c,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd132bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd132bf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x9c,0x10]
          vfnmadd132bf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfnmadd132bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd132bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfnmadd132bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9c,0x51,0x7f]
          vfnmadd132bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfnmadd132bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x9c,0x52,0x80]
          vfnmadd132bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfnmadd132bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd132bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd132bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9c,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd132bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd132bf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x9c,0x10]
          vfnmadd132bf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfnmadd132bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd132bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfnmadd132bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9c,0x51,0x7f]
          vfnmadd132bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfnmadd132bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x9c,0x52,0x80]
          vfnmadd132bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfnmadd213bf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xac,0xd4]
          vfnmadd213bf16 ymm2, ymm3, ymm4

// CHECK: vfnmadd213bf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xac,0xd4]
          vfnmadd213bf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfnmadd213bf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xac,0xd4]
          vfnmadd213bf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfnmadd213bf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xac,0xd4]
          vfnmadd213bf16 zmm2, zmm3, zmm4

// CHECK: vfnmadd213bf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xac,0xd4]
          vfnmadd213bf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfnmadd213bf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xac,0xd4]
          vfnmadd213bf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfnmadd213bf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xac,0xd4]
          vfnmadd213bf16 xmm2, xmm3, xmm4

// CHECK: vfnmadd213bf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xac,0xd4]
          vfnmadd213bf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfnmadd213bf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xac,0xd4]
          vfnmadd213bf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfnmadd213bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xac,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd213bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd213bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xac,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd213bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd213bf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xac,0x10]
          vfnmadd213bf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfnmadd213bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xac,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd213bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfnmadd213bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xac,0x51,0x7f]
          vfnmadd213bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfnmadd213bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xac,0x52,0x80]
          vfnmadd213bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfnmadd213bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xac,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd213bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd213bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xac,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd213bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd213bf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xac,0x10]
          vfnmadd213bf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfnmadd213bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xac,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd213bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfnmadd213bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xac,0x51,0x7f]
          vfnmadd213bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfnmadd213bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xac,0x52,0x80]
          vfnmadd213bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfnmadd213bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xac,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd213bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd213bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xac,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd213bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd213bf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xac,0x10]
          vfnmadd213bf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfnmadd213bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xac,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd213bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfnmadd213bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xac,0x51,0x7f]
          vfnmadd213bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfnmadd213bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xac,0x52,0x80]
          vfnmadd213bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfnmadd231bf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbc,0xd4]
          vfnmadd231bf16 ymm2, ymm3, ymm4

// CHECK: vfnmadd231bf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xbc,0xd4]
          vfnmadd231bf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfnmadd231bf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xbc,0xd4]
          vfnmadd231bf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfnmadd231bf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbc,0xd4]
          vfnmadd231bf16 zmm2, zmm3, zmm4

// CHECK: vfnmadd231bf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xbc,0xd4]
          vfnmadd231bf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfnmadd231bf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xbc,0xd4]
          vfnmadd231bf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfnmadd231bf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbc,0xd4]
          vfnmadd231bf16 xmm2, xmm3, xmm4

// CHECK: vfnmadd231bf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xbc,0xd4]
          vfnmadd231bf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfnmadd231bf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xbc,0xd4]
          vfnmadd231bf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfnmadd231bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbc,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd231bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd231bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xbc,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd231bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd231bf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xbc,0x10]
          vfnmadd231bf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfnmadd231bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbc,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd231bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfnmadd231bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xbc,0x51,0x7f]
          vfnmadd231bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfnmadd231bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xbc,0x52,0x80]
          vfnmadd231bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfnmadd231bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbc,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd231bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd231bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xbc,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd231bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd231bf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xbc,0x10]
          vfnmadd231bf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfnmadd231bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbc,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd231bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfnmadd231bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xbc,0x51,0x7f]
          vfnmadd231bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfnmadd231bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xbc,0x52,0x80]
          vfnmadd231bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfnmadd231bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbc,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd231bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd231bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xbc,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd231bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd231bf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xbc,0x10]
          vfnmadd231bf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfnmadd231bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbc,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd231bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfnmadd231bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xbc,0x51,0x7f]
          vfnmadd231bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfnmadd231bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xbc,0x52,0x80]
          vfnmadd231bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfnmsub132bf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9e,0xd4]
          vfnmsub132bf16 ymm2, ymm3, ymm4

// CHECK: vfnmsub132bf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9e,0xd4]
          vfnmsub132bf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfnmsub132bf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9e,0xd4]
          vfnmsub132bf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfnmsub132bf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9e,0xd4]
          vfnmsub132bf16 zmm2, zmm3, zmm4

// CHECK: vfnmsub132bf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9e,0xd4]
          vfnmsub132bf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfnmsub132bf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9e,0xd4]
          vfnmsub132bf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfnmsub132bf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9e,0xd4]
          vfnmsub132bf16 xmm2, xmm3, xmm4

// CHECK: vfnmsub132bf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9e,0xd4]
          vfnmsub132bf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfnmsub132bf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9e,0xd4]
          vfnmsub132bf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfnmsub132bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub132bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub132bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9e,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub132bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub132bf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x9e,0x10]
          vfnmsub132bf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfnmsub132bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9e,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub132bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfnmsub132bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9e,0x51,0x7f]
          vfnmsub132bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfnmsub132bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x9e,0x52,0x80]
          vfnmsub132bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfnmsub132bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub132bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub132bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9e,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub132bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub132bf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x9e,0x10]
          vfnmsub132bf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfnmsub132bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9e,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub132bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfnmsub132bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9e,0x51,0x7f]
          vfnmsub132bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfnmsub132bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x9e,0x52,0x80]
          vfnmsub132bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfnmsub132bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub132bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub132bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9e,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub132bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub132bf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x9e,0x10]
          vfnmsub132bf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfnmsub132bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9e,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub132bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfnmsub132bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9e,0x51,0x7f]
          vfnmsub132bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfnmsub132bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x9e,0x52,0x80]
          vfnmsub132bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfnmsub213bf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xae,0xd4]
          vfnmsub213bf16 ymm2, ymm3, ymm4

// CHECK: vfnmsub213bf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xae,0xd4]
          vfnmsub213bf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfnmsub213bf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xae,0xd4]
          vfnmsub213bf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfnmsub213bf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xae,0xd4]
          vfnmsub213bf16 zmm2, zmm3, zmm4

// CHECK: vfnmsub213bf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xae,0xd4]
          vfnmsub213bf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfnmsub213bf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xae,0xd4]
          vfnmsub213bf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfnmsub213bf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xae,0xd4]
          vfnmsub213bf16 xmm2, xmm3, xmm4

// CHECK: vfnmsub213bf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xae,0xd4]
          vfnmsub213bf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfnmsub213bf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xae,0xd4]
          vfnmsub213bf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfnmsub213bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xae,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub213bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub213bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xae,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub213bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub213bf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xae,0x10]
          vfnmsub213bf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfnmsub213bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xae,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub213bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfnmsub213bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xae,0x51,0x7f]
          vfnmsub213bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfnmsub213bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xae,0x52,0x80]
          vfnmsub213bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfnmsub213bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xae,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub213bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub213bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xae,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub213bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub213bf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xae,0x10]
          vfnmsub213bf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfnmsub213bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xae,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub213bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfnmsub213bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xae,0x51,0x7f]
          vfnmsub213bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfnmsub213bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xae,0x52,0x80]
          vfnmsub213bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfnmsub213bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xae,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub213bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub213bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xae,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub213bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub213bf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xae,0x10]
          vfnmsub213bf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfnmsub213bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xae,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub213bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfnmsub213bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xae,0x51,0x7f]
          vfnmsub213bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfnmsub213bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xae,0x52,0x80]
          vfnmsub213bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfnmsub231bf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbe,0xd4]
          vfnmsub231bf16 ymm2, ymm3, ymm4

// CHECK: vfnmsub231bf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xbe,0xd4]
          vfnmsub231bf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfnmsub231bf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xbe,0xd4]
          vfnmsub231bf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfnmsub231bf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbe,0xd4]
          vfnmsub231bf16 zmm2, zmm3, zmm4

// CHECK: vfnmsub231bf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xbe,0xd4]
          vfnmsub231bf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfnmsub231bf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xbe,0xd4]
          vfnmsub231bf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfnmsub231bf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbe,0xd4]
          vfnmsub231bf16 xmm2, xmm3, xmm4

// CHECK: vfnmsub231bf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xbe,0xd4]
          vfnmsub231bf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfnmsub231bf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xbe,0xd4]
          vfnmsub231bf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfnmsub231bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbe,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub231bf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub231bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xbe,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub231bf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub231bf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xbe,0x10]
          vfnmsub231bf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfnmsub231bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbe,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub231bf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfnmsub231bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xbe,0x51,0x7f]
          vfnmsub231bf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfnmsub231bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xbe,0x52,0x80]
          vfnmsub231bf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfnmsub231bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbe,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub231bf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub231bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xbe,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub231bf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub231bf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xbe,0x10]
          vfnmsub231bf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfnmsub231bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbe,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub231bf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfnmsub231bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xbe,0x51,0x7f]
          vfnmsub231bf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfnmsub231bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xbe,0x52,0x80]
          vfnmsub231bf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfnmsub231bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbe,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub231bf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub231bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xbe,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub231bf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub231bf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xbe,0x10]
          vfnmsub231bf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfnmsub231bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbe,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub231bf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfnmsub231bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xbe,0x51,0x7f]
          vfnmsub231bf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfnmsub231bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xbe,0x52,0x80]
          vfnmsub231bf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfpclassbf16 k5, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x66,0xeb,0x7b]
          vfpclassbf16 k5, zmm3, 123

// CHECK: vfpclassbf16 k5 {k7}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x66,0xeb,0x7b]
          vfpclassbf16 k5 {k7}, zmm3, 123

// CHECK: vfpclassbf16 k5, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x66,0xeb,0x7b]
          vfpclassbf16 k5, ymm3, 123

// CHECK: vfpclassbf16 k5 {k7}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x66,0xeb,0x7b]
          vfpclassbf16 k5 {k7}, ymm3, 123

// CHECK: vfpclassbf16 k5, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x66,0xeb,0x7b]
          vfpclassbf16 k5, xmm3, 123

// CHECK: vfpclassbf16 k5 {k7}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x66,0xeb,0x7b]
          vfpclassbf16 k5 {k7}, xmm3, 123

// CHECK: vfpclassbf16 k5, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x66,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vfpclassbf16 k5, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vfpclassbf16 k5 {k7}, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x66,0xac,0x87,0x23,0x01,0x00,0x00,0x7b]
          vfpclassbf16 k5 {k7}, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vfpclassbf16 k5, word ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x66,0x28,0x7b]
          vfpclassbf16 k5, word ptr [eax]{1to8}, 123

// CHECK: vfpclassbf16 k5, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x66,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vfpclassbf16 k5, xmmword ptr [2*ebp - 512], 123

// CHECK: vfpclassbf16 k5 {k7}, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x66,0x69,0x7f,0x7b]
          vfpclassbf16 k5 {k7}, xmmword ptr [ecx + 2032], 123

// CHECK: vfpclassbf16 k5 {k7}, word ptr [edx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x1f,0x66,0x6a,0x80,0x7b]
          vfpclassbf16 k5 {k7}, word ptr [edx - 256]{1to8}, 123

// CHECK: vfpclassbf16 k5, word ptr [eax]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x66,0x28,0x7b]
          vfpclassbf16 k5, word ptr [eax]{1to16}, 123

// CHECK: vfpclassbf16 k5, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x66,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vfpclassbf16 k5, ymmword ptr [2*ebp - 1024], 123

// CHECK: vfpclassbf16 k5 {k7}, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x66,0x69,0x7f,0x7b]
          vfpclassbf16 k5 {k7}, ymmword ptr [ecx + 4064], 123

// CHECK: vfpclassbf16 k5 {k7}, word ptr [edx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x3f,0x66,0x6a,0x80,0x7b]
          vfpclassbf16 k5 {k7}, word ptr [edx - 256]{1to16}, 123

// CHECK: vfpclassbf16 k5, word ptr [eax]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x66,0x28,0x7b]
          vfpclassbf16 k5, word ptr [eax]{1to32}, 123

// CHECK: vfpclassbf16 k5, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x66,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vfpclassbf16 k5, zmmword ptr [2*ebp - 2048], 123

// CHECK: vfpclassbf16 k5 {k7}, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x66,0x69,0x7f,0x7b]
          vfpclassbf16 k5 {k7}, zmmword ptr [ecx + 8128], 123

// CHECK: vfpclassbf16 k5 {k7}, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x5f,0x66,0x6a,0x80,0x7b]
          vfpclassbf16 k5 {k7}, word ptr [edx - 256]{1to32}, 123

// CHECK: vgetexpbf16 xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x42,0xd3]
          vgetexpbf16 xmm2, xmm3

// CHECK: vgetexpbf16 xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x42,0xd3]
          vgetexpbf16 xmm2 {k7}, xmm3

// CHECK: vgetexpbf16 xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x42,0xd3]
          vgetexpbf16 xmm2 {k7} {z}, xmm3

// CHECK: vgetexpbf16 zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x42,0xd3]
          vgetexpbf16 zmm2, zmm3

// CHECK: vgetexpbf16 zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x42,0xd3]
          vgetexpbf16 zmm2 {k7}, zmm3

// CHECK: vgetexpbf16 zmm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x42,0xd3]
          vgetexpbf16 zmm2 {k7} {z}, zmm3

// CHECK: vgetexpbf16 ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x42,0xd3]
          vgetexpbf16 ymm2, ymm3

// CHECK: vgetexpbf16 ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x42,0xd3]
          vgetexpbf16 ymm2 {k7}, ymm3

// CHECK: vgetexpbf16 ymm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x42,0xd3]
          vgetexpbf16 ymm2 {k7} {z}, ymm3

// CHECK: vgetexpbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x42,0x94,0xf4,0x00,0x00,0x00,0x10]
          vgetexpbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vgetexpbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x42,0x94,0x87,0x23,0x01,0x00,0x00]
          vgetexpbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vgetexpbf16 xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x42,0x10]
          vgetexpbf16 xmm2, word ptr [eax]{1to8}

// CHECK: vgetexpbf16 xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x42,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vgetexpbf16 xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vgetexpbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x42,0x51,0x7f]
          vgetexpbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vgetexpbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x42,0x52,0x80]
          vgetexpbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vgetexpbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x42,0x94,0xf4,0x00,0x00,0x00,0x10]
          vgetexpbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vgetexpbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x42,0x94,0x87,0x23,0x01,0x00,0x00]
          vgetexpbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vgetexpbf16 ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x42,0x10]
          vgetexpbf16 ymm2, word ptr [eax]{1to16}

// CHECK: vgetexpbf16 ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x42,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vgetexpbf16 ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vgetexpbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x42,0x51,0x7f]
          vgetexpbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vgetexpbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x42,0x52,0x80]
          vgetexpbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vgetexpbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x42,0x94,0xf4,0x00,0x00,0x00,0x10]
          vgetexpbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vgetexpbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x42,0x94,0x87,0x23,0x01,0x00,0x00]
          vgetexpbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vgetexpbf16 zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x42,0x10]
          vgetexpbf16 zmm2, word ptr [eax]{1to32}

// CHECK: vgetexpbf16 zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x42,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vgetexpbf16 zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vgetexpbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x42,0x51,0x7f]
          vgetexpbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vgetexpbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x42,0x52,0x80]
          vgetexpbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vgetmantbf16 zmm2, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x26,0xd3,0x7b]
          vgetmantbf16 zmm2, zmm3, 123

// CHECK: vgetmantbf16 zmm2 {k7}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x26,0xd3,0x7b]
          vgetmantbf16 zmm2 {k7}, zmm3, 123

// CHECK: vgetmantbf16 zmm2 {k7} {z}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x26,0xd3,0x7b]
          vgetmantbf16 zmm2 {k7} {z}, zmm3, 123

// CHECK: vgetmantbf16 ymm2, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x26,0xd3,0x7b]
          vgetmantbf16 ymm2, ymm3, 123

// CHECK: vgetmantbf16 ymm2 {k7}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x26,0xd3,0x7b]
          vgetmantbf16 ymm2 {k7}, ymm3, 123

// CHECK: vgetmantbf16 ymm2 {k7} {z}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x26,0xd3,0x7b]
          vgetmantbf16 ymm2 {k7} {z}, ymm3, 123

// CHECK: vgetmantbf16 xmm2, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x26,0xd3,0x7b]
          vgetmantbf16 xmm2, xmm3, 123

// CHECK: vgetmantbf16 xmm2 {k7}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x26,0xd3,0x7b]
          vgetmantbf16 xmm2 {k7}, xmm3, 123

// CHECK: vgetmantbf16 xmm2 {k7} {z}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x26,0xd3,0x7b]
          vgetmantbf16 xmm2 {k7} {z}, xmm3, 123

// CHECK: vgetmantbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x26,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vgetmantbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vgetmantbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x26,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vgetmantbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vgetmantbf16 xmm2, word ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x26,0x10,0x7b]
          vgetmantbf16 xmm2, word ptr [eax]{1to8}, 123

// CHECK: vgetmantbf16 xmm2, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x26,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vgetmantbf16 xmm2, xmmword ptr [2*ebp - 512], 123

// CHECK: vgetmantbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x26,0x51,0x7f,0x7b]
          vgetmantbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032], 123

// CHECK: vgetmantbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x9f,0x26,0x52,0x80,0x7b]
          vgetmantbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}, 123

// CHECK: vgetmantbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x26,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vgetmantbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vgetmantbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x26,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vgetmantbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vgetmantbf16 ymm2, word ptr [eax]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x26,0x10,0x7b]
          vgetmantbf16 ymm2, word ptr [eax]{1to16}, 123

// CHECK: vgetmantbf16 ymm2, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x26,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vgetmantbf16 ymm2, ymmword ptr [2*ebp - 1024], 123

// CHECK: vgetmantbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x26,0x51,0x7f,0x7b]
          vgetmantbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064], 123

// CHECK: vgetmantbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xbf,0x26,0x52,0x80,0x7b]
          vgetmantbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}, 123

// CHECK: vgetmantbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x26,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vgetmantbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vgetmantbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x26,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vgetmantbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vgetmantbf16 zmm2, word ptr [eax]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x26,0x10,0x7b]
          vgetmantbf16 zmm2, word ptr [eax]{1to32}, 123

// CHECK: vgetmantbf16 zmm2, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x26,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vgetmantbf16 zmm2, zmmword ptr [2*ebp - 2048], 123

// CHECK: vgetmantbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x26,0x51,0x7f,0x7b]
          vgetmantbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128], 123

// CHECK: vgetmantbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xdf,0x26,0x52,0x80,0x7b]
          vgetmantbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}, 123

// CHECK: vmaxbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5f,0xd4]
          vmaxbf16 ymm2, ymm3, ymm4

// CHECK: vmaxbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5f,0xd4]
          vmaxbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vmaxbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5f,0xd4]
          vmaxbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vmaxbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5f,0xd4]
          vmaxbf16 zmm2, zmm3, zmm4

// CHECK: vmaxbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5f,0xd4]
          vmaxbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vmaxbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5f,0xd4]
          vmaxbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vmaxbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5f,0xd4]
          vmaxbf16 xmm2, xmm3, xmm4

// CHECK: vmaxbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5f,0xd4]
          vmaxbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vmaxbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5f,0xd4]
          vmaxbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vmaxbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5f,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmaxbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vmaxbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5f,0x94,0x87,0x23,0x01,0x00,0x00]
          vmaxbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vmaxbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x5f,0x10]
          vmaxbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vmaxbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5f,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vmaxbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vmaxbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5f,0x51,0x7f]
          vmaxbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vmaxbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x5f,0x52,0x80]
          vmaxbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vmaxbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5f,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmaxbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vmaxbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5f,0x94,0x87,0x23,0x01,0x00,0x00]
          vmaxbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vmaxbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x5f,0x10]
          vmaxbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vmaxbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5f,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vmaxbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vmaxbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5f,0x51,0x7f]
          vmaxbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vmaxbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x5f,0x52,0x80]
          vmaxbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vmaxbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5f,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmaxbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vmaxbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5f,0x94,0x87,0x23,0x01,0x00,0x00]
          vmaxbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vmaxbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x5f,0x10]
          vmaxbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vmaxbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5f,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vmaxbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vmaxbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5f,0x51,0x7f]
          vmaxbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vmaxbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x5f,0x52,0x80]
          vmaxbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vminbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5d,0xd4]
          vminbf16 ymm2, ymm3, ymm4

// CHECK: vminbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5d,0xd4]
          vminbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vminbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5d,0xd4]
          vminbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vminbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5d,0xd4]
          vminbf16 zmm2, zmm3, zmm4

// CHECK: vminbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5d,0xd4]
          vminbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vminbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5d,0xd4]
          vminbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vminbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5d,0xd4]
          vminbf16 xmm2, xmm3, xmm4

// CHECK: vminbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5d,0xd4]
          vminbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vminbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5d,0xd4]
          vminbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vminbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vminbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vminbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5d,0x94,0x87,0x23,0x01,0x00,0x00]
          vminbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vminbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x5d,0x10]
          vminbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vminbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5d,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vminbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vminbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5d,0x51,0x7f]
          vminbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vminbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x5d,0x52,0x80]
          vminbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vminbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vminbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vminbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5d,0x94,0x87,0x23,0x01,0x00,0x00]
          vminbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vminbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x5d,0x10]
          vminbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vminbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5d,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vminbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vminbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5d,0x51,0x7f]
          vminbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vminbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x5d,0x52,0x80]
          vminbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vminbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vminbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vminbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5d,0x94,0x87,0x23,0x01,0x00,0x00]
          vminbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vminbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x5d,0x10]
          vminbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vminbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5d,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vminbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vminbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5d,0x51,0x7f]
          vminbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vminbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x5d,0x52,0x80]
          vminbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vmulbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x59,0xd4]
          vmulbf16 ymm2, ymm3, ymm4

// CHECK: vmulbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x59,0xd4]
          vmulbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vmulbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x59,0xd4]
          vmulbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vmulbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x59,0xd4]
          vmulbf16 zmm2, zmm3, zmm4

// CHECK: vmulbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x59,0xd4]
          vmulbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vmulbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x59,0xd4]
          vmulbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vmulbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x59,0xd4]
          vmulbf16 xmm2, xmm3, xmm4

// CHECK: vmulbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x59,0xd4]
          vmulbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vmulbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x59,0xd4]
          vmulbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vmulbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x59,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmulbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vmulbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x59,0x94,0x87,0x23,0x01,0x00,0x00]
          vmulbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vmulbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x59,0x10]
          vmulbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vmulbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x59,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vmulbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vmulbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x59,0x51,0x7f]
          vmulbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vmulbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x59,0x52,0x80]
          vmulbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vmulbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x59,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmulbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vmulbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x59,0x94,0x87,0x23,0x01,0x00,0x00]
          vmulbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vmulbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x59,0x10]
          vmulbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vmulbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x59,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vmulbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vmulbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x59,0x51,0x7f]
          vmulbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vmulbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x59,0x52,0x80]
          vmulbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vmulbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x59,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmulbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vmulbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x59,0x94,0x87,0x23,0x01,0x00,0x00]
          vmulbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vmulbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x59,0x10]
          vmulbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vmulbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x59,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vmulbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vmulbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x59,0x51,0x7f]
          vmulbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vmulbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x59,0x52,0x80]
          vmulbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vrcpbf16 xmm2, xmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4c,0xd3]
          vrcpbf16 xmm2, xmm3

// CHECK: vrcpbf16 xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x0f,0x4c,0xd3]
          vrcpbf16 xmm2 {k7}, xmm3

// CHECK: vrcpbf16 xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x8f,0x4c,0xd3]
          vrcpbf16 xmm2 {k7} {z}, xmm3

// CHECK: vrcpbf16 zmm2, zmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4c,0xd3]
          vrcpbf16 zmm2, zmm3

// CHECK: vrcpbf16 zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x4c,0xd3]
          vrcpbf16 zmm2 {k7}, zmm3

// CHECK: vrcpbf16 zmm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x4c,0xd3]
          vrcpbf16 zmm2 {k7} {z}, zmm3

// CHECK: vrcpbf16 ymm2, ymm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4c,0xd3]
          vrcpbf16 ymm2, ymm3

// CHECK: vrcpbf16 ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x2f,0x4c,0xd3]
          vrcpbf16 ymm2 {k7}, ymm3

// CHECK: vrcpbf16 ymm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf6,0x7c,0xaf,0x4c,0xd3]
          vrcpbf16 ymm2 {k7} {z}, ymm3

// CHECK: vrcpbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrcpbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vrcpbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x7c,0x0f,0x4c,0x94,0x87,0x23,0x01,0x00,0x00]
          vrcpbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vrcpbf16 xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x7c,0x18,0x4c,0x10]
          vrcpbf16 xmm2, word ptr [eax]{1to8}

// CHECK: vrcpbf16 xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vrcpbf16 xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vrcpbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x7c,0x8f,0x4c,0x51,0x7f]
          vrcpbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vrcpbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x7c,0x9f,0x4c,0x52,0x80]
          vrcpbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vrcpbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrcpbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vrcpbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x7c,0x2f,0x4c,0x94,0x87,0x23,0x01,0x00,0x00]
          vrcpbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vrcpbf16 ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x7c,0x38,0x4c,0x10]
          vrcpbf16 ymm2, word ptr [eax]{1to16}

// CHECK: vrcpbf16 ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vrcpbf16 ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vrcpbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x7c,0xaf,0x4c,0x51,0x7f]
          vrcpbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vrcpbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x7c,0xbf,0x4c,0x52,0x80]
          vrcpbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vrcpbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrcpbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vrcpbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x4c,0x94,0x87,0x23,0x01,0x00,0x00]
          vrcpbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vrcpbf16 zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7c,0x58,0x4c,0x10]
          vrcpbf16 zmm2, word ptr [eax]{1to32}

// CHECK: vrcpbf16 zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vrcpbf16 zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vrcpbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x4c,0x51,0x7f]
          vrcpbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vrcpbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7c,0xdf,0x4c,0x52,0x80]
          vrcpbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vreducebf16 zmm2, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x56,0xd3,0x7b]
          vreducebf16 zmm2, zmm3, 123

// CHECK: vreducebf16 zmm2 {k7}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x56,0xd3,0x7b]
          vreducebf16 zmm2 {k7}, zmm3, 123

// CHECK: vreducebf16 zmm2 {k7} {z}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x56,0xd3,0x7b]
          vreducebf16 zmm2 {k7} {z}, zmm3, 123

// CHECK: vreducebf16 ymm2, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x56,0xd3,0x7b]
          vreducebf16 ymm2, ymm3, 123

// CHECK: vreducebf16 ymm2 {k7}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x56,0xd3,0x7b]
          vreducebf16 ymm2 {k7}, ymm3, 123

// CHECK: vreducebf16 ymm2 {k7} {z}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x56,0xd3,0x7b]
          vreducebf16 ymm2 {k7} {z}, ymm3, 123

// CHECK: vreducebf16 xmm2, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x56,0xd3,0x7b]
          vreducebf16 xmm2, xmm3, 123

// CHECK: vreducebf16 xmm2 {k7}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x56,0xd3,0x7b]
          vreducebf16 xmm2 {k7}, xmm3, 123

// CHECK: vreducebf16 xmm2 {k7} {z}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x56,0xd3,0x7b]
          vreducebf16 xmm2 {k7} {z}, xmm3, 123

// CHECK: vreducebf16 xmm2, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x56,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vreducebf16 xmm2, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vreducebf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x56,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vreducebf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vreducebf16 xmm2, word ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x56,0x10,0x7b]
          vreducebf16 xmm2, word ptr [eax]{1to8}, 123

// CHECK: vreducebf16 xmm2, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x56,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vreducebf16 xmm2, xmmword ptr [2*ebp - 512], 123

// CHECK: vreducebf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x56,0x51,0x7f,0x7b]
          vreducebf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032], 123

// CHECK: vreducebf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x9f,0x56,0x52,0x80,0x7b]
          vreducebf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}, 123

// CHECK: vreducebf16 ymm2, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x56,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vreducebf16 ymm2, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vreducebf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x56,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vreducebf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vreducebf16 ymm2, word ptr [eax]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x56,0x10,0x7b]
          vreducebf16 ymm2, word ptr [eax]{1to16}, 123

// CHECK: vreducebf16 ymm2, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x56,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vreducebf16 ymm2, ymmword ptr [2*ebp - 1024], 123

// CHECK: vreducebf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x56,0x51,0x7f,0x7b]
          vreducebf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064], 123

// CHECK: vreducebf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xbf,0x56,0x52,0x80,0x7b]
          vreducebf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}, 123

// CHECK: vreducebf16 zmm2, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x56,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vreducebf16 zmm2, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vreducebf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x56,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vreducebf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vreducebf16 zmm2, word ptr [eax]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x56,0x10,0x7b]
          vreducebf16 zmm2, word ptr [eax]{1to32}, 123

// CHECK: vreducebf16 zmm2, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x56,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vreducebf16 zmm2, zmmword ptr [2*ebp - 2048], 123

// CHECK: vreducebf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x56,0x51,0x7f,0x7b]
          vreducebf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128], 123

// CHECK: vreducebf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xdf,0x56,0x52,0x80,0x7b]
          vreducebf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}, 123

// CHECK: vrndscalebf16 zmm2, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x08,0xd3,0x7b]
          vrndscalebf16 zmm2, zmm3, 123

// CHECK: vrndscalebf16 zmm2 {k7}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x08,0xd3,0x7b]
          vrndscalebf16 zmm2 {k7}, zmm3, 123

// CHECK: vrndscalebf16 zmm2 {k7} {z}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x08,0xd3,0x7b]
          vrndscalebf16 zmm2 {k7} {z}, zmm3, 123

// CHECK: vrndscalebf16 ymm2, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x08,0xd3,0x7b]
          vrndscalebf16 ymm2, ymm3, 123

// CHECK: vrndscalebf16 ymm2 {k7}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x08,0xd3,0x7b]
          vrndscalebf16 ymm2 {k7}, ymm3, 123

// CHECK: vrndscalebf16 ymm2 {k7} {z}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x08,0xd3,0x7b]
          vrndscalebf16 ymm2 {k7} {z}, ymm3, 123

// CHECK: vrndscalebf16 xmm2, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x08,0xd3,0x7b]
          vrndscalebf16 xmm2, xmm3, 123

// CHECK: vrndscalebf16 xmm2 {k7}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x08,0xd3,0x7b]
          vrndscalebf16 xmm2 {k7}, xmm3, 123

// CHECK: vrndscalebf16 xmm2 {k7} {z}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x08,0xd3,0x7b]
          vrndscalebf16 xmm2 {k7} {z}, xmm3, 123

// CHECK: vrndscalebf16 xmm2, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x08,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vrndscalebf16 xmm2, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vrndscalebf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x08,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vrndscalebf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vrndscalebf16 xmm2, word ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x08,0x10,0x7b]
          vrndscalebf16 xmm2, word ptr [eax]{1to8}, 123

// CHECK: vrndscalebf16 xmm2, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x08,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vrndscalebf16 xmm2, xmmword ptr [2*ebp - 512], 123

// CHECK: vrndscalebf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x08,0x51,0x7f,0x7b]
          vrndscalebf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032], 123

// CHECK: vrndscalebf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x9f,0x08,0x52,0x80,0x7b]
          vrndscalebf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}, 123

// CHECK: vrndscalebf16 ymm2, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x08,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vrndscalebf16 ymm2, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vrndscalebf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x08,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vrndscalebf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vrndscalebf16 ymm2, word ptr [eax]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x08,0x10,0x7b]
          vrndscalebf16 ymm2, word ptr [eax]{1to16}, 123

// CHECK: vrndscalebf16 ymm2, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x08,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vrndscalebf16 ymm2, ymmword ptr [2*ebp - 1024], 123

// CHECK: vrndscalebf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x08,0x51,0x7f,0x7b]
          vrndscalebf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064], 123

// CHECK: vrndscalebf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xbf,0x08,0x52,0x80,0x7b]
          vrndscalebf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}, 123

// CHECK: vrndscalebf16 zmm2, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x08,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vrndscalebf16 zmm2, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vrndscalebf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x08,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vrndscalebf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vrndscalebf16 zmm2, word ptr [eax]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x08,0x10,0x7b]
          vrndscalebf16 zmm2, word ptr [eax]{1to32}, 123

// CHECK: vrndscalebf16 zmm2, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x08,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vrndscalebf16 zmm2, zmmword ptr [2*ebp - 2048], 123

// CHECK: vrndscalebf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x08,0x51,0x7f,0x7b]
          vrndscalebf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128], 123

// CHECK: vrndscalebf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xdf,0x08,0x52,0x80,0x7b]
          vrndscalebf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}, 123

// CHECK: vrsqrtbf16 xmm2, xmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4e,0xd3]
          vrsqrtbf16 xmm2, xmm3

// CHECK: vrsqrtbf16 xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x0f,0x4e,0xd3]
          vrsqrtbf16 xmm2 {k7}, xmm3

// CHECK: vrsqrtbf16 xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x8f,0x4e,0xd3]
          vrsqrtbf16 xmm2 {k7} {z}, xmm3

// CHECK: vrsqrtbf16 zmm2, zmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4e,0xd3]
          vrsqrtbf16 zmm2, zmm3

// CHECK: vrsqrtbf16 zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x4e,0xd3]
          vrsqrtbf16 zmm2 {k7}, zmm3

// CHECK: vrsqrtbf16 zmm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x4e,0xd3]
          vrsqrtbf16 zmm2 {k7} {z}, zmm3

// CHECK: vrsqrtbf16 ymm2, ymm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4e,0xd3]
          vrsqrtbf16 ymm2, ymm3

// CHECK: vrsqrtbf16 ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x2f,0x4e,0xd3]
          vrsqrtbf16 ymm2 {k7}, ymm3

// CHECK: vrsqrtbf16 ymm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf6,0x7c,0xaf,0x4e,0xd3]
          vrsqrtbf16 ymm2 {k7} {z}, ymm3

// CHECK: vrsqrtbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrsqrtbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vrsqrtbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x7c,0x0f,0x4e,0x94,0x87,0x23,0x01,0x00,0x00]
          vrsqrtbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vrsqrtbf16 xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x7c,0x18,0x4e,0x10]
          vrsqrtbf16 xmm2, word ptr [eax]{1to8}

// CHECK: vrsqrtbf16 xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4e,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vrsqrtbf16 xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vrsqrtbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x7c,0x8f,0x4e,0x51,0x7f]
          vrsqrtbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vrsqrtbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x7c,0x9f,0x4e,0x52,0x80]
          vrsqrtbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vrsqrtbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrsqrtbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vrsqrtbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x7c,0x2f,0x4e,0x94,0x87,0x23,0x01,0x00,0x00]
          vrsqrtbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vrsqrtbf16 ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x7c,0x38,0x4e,0x10]
          vrsqrtbf16 ymm2, word ptr [eax]{1to16}

// CHECK: vrsqrtbf16 ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4e,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vrsqrtbf16 ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vrsqrtbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x7c,0xaf,0x4e,0x51,0x7f]
          vrsqrtbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vrsqrtbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x7c,0xbf,0x4e,0x52,0x80]
          vrsqrtbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vrsqrtbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrsqrtbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vrsqrtbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x4e,0x94,0x87,0x23,0x01,0x00,0x00]
          vrsqrtbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vrsqrtbf16 zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7c,0x58,0x4e,0x10]
          vrsqrtbf16 zmm2, word ptr [eax]{1to32}

// CHECK: vrsqrtbf16 zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4e,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vrsqrtbf16 zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vrsqrtbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x4e,0x51,0x7f]
          vrsqrtbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vrsqrtbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7c,0xdf,0x4e,0x52,0x80]
          vrsqrtbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vscalefbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x2c,0xd4]
          vscalefbf16 ymm2, ymm3, ymm4

// CHECK: vscalefbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x2c,0xd4]
          vscalefbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vscalefbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x2c,0xd4]
          vscalefbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vscalefbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x2c,0xd4]
          vscalefbf16 zmm2, zmm3, zmm4

// CHECK: vscalefbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x2c,0xd4]
          vscalefbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vscalefbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x2c,0xd4]
          vscalefbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vscalefbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x2c,0xd4]
          vscalefbf16 xmm2, xmm3, xmm4

// CHECK: vscalefbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x2c,0xd4]
          vscalefbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vscalefbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x2c,0xd4]
          vscalefbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vscalefbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x2c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vscalefbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vscalefbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x2c,0x94,0x87,0x23,0x01,0x00,0x00]
          vscalefbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vscalefbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x2c,0x10]
          vscalefbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vscalefbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x2c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vscalefbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vscalefbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x2c,0x51,0x7f]
          vscalefbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vscalefbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x2c,0x52,0x80]
          vscalefbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vscalefbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x2c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vscalefbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vscalefbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x2c,0x94,0x87,0x23,0x01,0x00,0x00]
          vscalefbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vscalefbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x2c,0x10]
          vscalefbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vscalefbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x2c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vscalefbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vscalefbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x2c,0x51,0x7f]
          vscalefbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vscalefbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x2c,0x52,0x80]
          vscalefbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vscalefbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x2c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vscalefbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vscalefbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x2c,0x94,0x87,0x23,0x01,0x00,0x00]
          vscalefbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vscalefbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x2c,0x10]
          vscalefbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vscalefbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x2c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vscalefbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vscalefbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x2c,0x51,0x7f]
          vscalefbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vscalefbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x2c,0x52,0x80]
          vscalefbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vsqrtbf16 xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x51,0xd3]
          vsqrtbf16 xmm2, xmm3

// CHECK: vsqrtbf16 xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x51,0xd3]
          vsqrtbf16 xmm2 {k7}, xmm3

// CHECK: vsqrtbf16 xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x51,0xd3]
          vsqrtbf16 xmm2 {k7} {z}, xmm3

// CHECK: vsqrtbf16 zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x51,0xd3]
          vsqrtbf16 zmm2, zmm3

// CHECK: vsqrtbf16 zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x51,0xd3]
          vsqrtbf16 zmm2 {k7}, zmm3

// CHECK: vsqrtbf16 zmm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x51,0xd3]
          vsqrtbf16 zmm2 {k7} {z}, zmm3

// CHECK: vsqrtbf16 ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x51,0xd3]
          vsqrtbf16 ymm2, ymm3

// CHECK: vsqrtbf16 ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x51,0xd3]
          vsqrtbf16 ymm2 {k7}, ymm3

// CHECK: vsqrtbf16 ymm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x51,0xd3]
          vsqrtbf16 ymm2 {k7} {z}, ymm3

// CHECK: vsqrtbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsqrtbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vsqrtbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vsqrtbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vsqrtbf16 xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x51,0x10]
          vsqrtbf16 xmm2, word ptr [eax]{1to8}

// CHECK: vsqrtbf16 xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x51,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vsqrtbf16 xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vsqrtbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x51,0x51,0x7f]
          vsqrtbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vsqrtbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x51,0x52,0x80]
          vsqrtbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vsqrtbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsqrtbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vsqrtbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vsqrtbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vsqrtbf16 ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x51,0x10]
          vsqrtbf16 ymm2, word ptr [eax]{1to16}

// CHECK: vsqrtbf16 ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x51,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vsqrtbf16 ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vsqrtbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x51,0x51,0x7f]
          vsqrtbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vsqrtbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x51,0x52,0x80]
          vsqrtbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vsqrtbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsqrtbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vsqrtbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vsqrtbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vsqrtbf16 zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x51,0x10]
          vsqrtbf16 zmm2, word ptr [eax]{1to32}

// CHECK: vsqrtbf16 zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x51,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vsqrtbf16 zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vsqrtbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x51,0x51,0x7f]
          vsqrtbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vsqrtbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x51,0x52,0x80]
          vsqrtbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vsubbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5c,0xd4]
          vsubbf16 ymm2, ymm3, ymm4

// CHECK: vsubbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5c,0xd4]
          vsubbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vsubbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5c,0xd4]
          vsubbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vsubbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5c,0xd4]
          vsubbf16 zmm2, zmm3, zmm4

// CHECK: vsubbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5c,0xd4]
          vsubbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vsubbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5c,0xd4]
          vsubbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vsubbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5c,0xd4]
          vsubbf16 xmm2, xmm3, xmm4

// CHECK: vsubbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5c,0xd4]
          vsubbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vsubbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5c,0xd4]
          vsubbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vsubbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsubbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vsubbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5c,0x94,0x87,0x23,0x01,0x00,0x00]
          vsubbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vsubbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x5c,0x10]
          vsubbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vsubbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vsubbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vsubbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5c,0x51,0x7f]
          vsubbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vsubbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x5c,0x52,0x80]
          vsubbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vsubbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsubbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vsubbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5c,0x94,0x87,0x23,0x01,0x00,0x00]
          vsubbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vsubbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x5c,0x10]
          vsubbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vsubbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vsubbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vsubbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5c,0x51,0x7f]
          vsubbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vsubbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x5c,0x52,0x80]
          vsubbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vsubbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsubbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vsubbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5c,0x94,0x87,0x23,0x01,0x00,0x00]
          vsubbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vsubbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x5c,0x10]
          vsubbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vsubbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vsubbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vsubbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5c,0x51,0x7f]
          vsubbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vsubbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x5c,0x52,0x80]
          vsubbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

