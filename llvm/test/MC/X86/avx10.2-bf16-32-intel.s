// RUN: llvm-mc -triple i386 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vaddnepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x58,0xd4]
          vaddnepbf16 ymm2, ymm3, ymm4

// CHECK: vaddnepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x58,0xd4]
          vaddnepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vaddnepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x58,0xd4]
          vaddnepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vaddnepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x58,0xd4]
          vaddnepbf16 zmm2, zmm3, zmm4

// CHECK: vaddnepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x58,0xd4]
          vaddnepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vaddnepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x58,0xd4]
          vaddnepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vaddnepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x58,0xd4]
          vaddnepbf16 xmm2, xmm3, xmm4

// CHECK: vaddnepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x58,0xd4]
          vaddnepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vaddnepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x58,0xd4]
          vaddnepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vaddnepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x58,0x94,0xf4,0x00,0x00,0x00,0x10]
          vaddnepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vaddnepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x58,0x94,0x87,0x23,0x01,0x00,0x00]
          vaddnepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vaddnepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x58,0x10]
          vaddnepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vaddnepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x58,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vaddnepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vaddnepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x58,0x51,0x7f]
          vaddnepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vaddnepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x58,0x52,0x80]
          vaddnepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vaddnepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x58,0x94,0xf4,0x00,0x00,0x00,0x10]
          vaddnepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vaddnepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x58,0x94,0x87,0x23,0x01,0x00,0x00]
          vaddnepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vaddnepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x58,0x10]
          vaddnepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vaddnepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x58,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vaddnepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vaddnepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x58,0x51,0x7f]
          vaddnepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vaddnepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x58,0x52,0x80]
          vaddnepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vaddnepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x58,0x94,0xf4,0x00,0x00,0x00,0x10]
          vaddnepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vaddnepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x58,0x94,0x87,0x23,0x01,0x00,0x00]
          vaddnepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vaddnepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x58,0x10]
          vaddnepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vaddnepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x58,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vaddnepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vaddnepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x58,0x51,0x7f]
          vaddnepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vaddnepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x58,0x52,0x80]
          vaddnepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vcmppbf16 k5, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0xc2,0xec,0x7b]
          vcmppbf16 k5, ymm3, ymm4, 123

// CHECK: vcmppbf16 k5 {k7}, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x2f,0xc2,0xec,0x7b]
          vcmppbf16 k5 {k7}, ymm3, ymm4, 123

// CHECK: vcmppbf16 k5, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0xc2,0xec,0x7b]
          vcmppbf16 k5, xmm3, xmm4, 123

// CHECK: vcmppbf16 k5 {k7}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x0f,0xc2,0xec,0x7b]
          vcmppbf16 k5 {k7}, xmm3, xmm4, 123

// CHECK: vcmppbf16 k5, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0xc2,0xec,0x7b]
          vcmppbf16 k5, zmm3, zmm4, 123

// CHECK: vcmppbf16 k5 {k7}, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x4f,0xc2,0xec,0x7b]
          vcmppbf16 k5 {k7}, zmm3, zmm4, 123

// CHECK: vcmppbf16 k5, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vcmppbf16 k5, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vcmppbf16 k5 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x4f,0xc2,0xac,0x87,0x23,0x01,0x00,0x00,0x7b]
          vcmppbf16 k5 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vcmppbf16 k5, zmm3, word ptr [eax]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x58,0xc2,0x28,0x7b]
          vcmppbf16 k5, zmm3, word ptr [eax]{1to32}, 123

// CHECK: vcmppbf16 k5, zmm3, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0xc2,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vcmppbf16 k5, zmm3, zmmword ptr [2*ebp - 2048], 123

// CHECK: vcmppbf16 k5 {k7}, zmm3, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x4f,0xc2,0x69,0x7f,0x7b]
          vcmppbf16 k5 {k7}, zmm3, zmmword ptr [ecx + 8128], 123

// CHECK: vcmppbf16 k5 {k7}, zmm3, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x5f,0xc2,0x6a,0x80,0x7b]
          vcmppbf16 k5 {k7}, zmm3, word ptr [edx - 256]{1to32}, 123

// CHECK: vcmppbf16 k5, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vcmppbf16 k5, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vcmppbf16 k5 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x0f,0xc2,0xac,0x87,0x23,0x01,0x00,0x00,0x7b]
          vcmppbf16 k5 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vcmppbf16 k5, xmm3, word ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x18,0xc2,0x28,0x7b]
          vcmppbf16 k5, xmm3, word ptr [eax]{1to8}, 123

// CHECK: vcmppbf16 k5, xmm3, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0xc2,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vcmppbf16 k5, xmm3, xmmword ptr [2*ebp - 512], 123

// CHECK: vcmppbf16 k5 {k7}, xmm3, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x0f,0xc2,0x69,0x7f,0x7b]
          vcmppbf16 k5 {k7}, xmm3, xmmword ptr [ecx + 2032], 123

// CHECK: vcmppbf16 k5 {k7}, xmm3, word ptr [edx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x1f,0xc2,0x6a,0x80,0x7b]
          vcmppbf16 k5 {k7}, xmm3, word ptr [edx - 256]{1to8}, 123

// CHECK: vcmppbf16 k5, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vcmppbf16 k5, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vcmppbf16 k5 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x2f,0xc2,0xac,0x87,0x23,0x01,0x00,0x00,0x7b]
          vcmppbf16 k5 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vcmppbf16 k5, ymm3, word ptr [eax]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x38,0xc2,0x28,0x7b]
          vcmppbf16 k5, ymm3, word ptr [eax]{1to16}, 123

// CHECK: vcmppbf16 k5, ymm3, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0xc2,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vcmppbf16 k5, ymm3, ymmword ptr [2*ebp - 1024], 123

// CHECK: vcmppbf16 k5 {k7}, ymm3, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x2f,0xc2,0x69,0x7f,0x7b]
          vcmppbf16 k5 {k7}, ymm3, ymmword ptr [ecx + 4064], 123

// CHECK: vcmppbf16 k5 {k7}, ymm3, word ptr [edx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x3f,0xc2,0x6a,0x80,0x7b]
          vcmppbf16 k5 {k7}, ymm3, word ptr [edx - 256]{1to16}, 123

// CHECK: vcomsbf16 xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0xd3]
          vcomsbf16 xmm2, xmm3

// CHECK: vcomsbf16 xmm2, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcomsbf16 xmm2, word ptr [esp + 8*esi + 268435456]

// CHECK: vcomsbf16 xmm2, word ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x94,0x87,0x23,0x01,0x00,0x00]
          vcomsbf16 xmm2, word ptr [edi + 4*eax + 291]

// CHECK: vcomsbf16 xmm2, word ptr [eax]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x10]
          vcomsbf16 xmm2, word ptr [eax]

// CHECK: vcomsbf16 xmm2, word ptr [2*ebp - 64]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x14,0x6d,0xc0,0xff,0xff,0xff]
          vcomsbf16 xmm2, word ptr [2*ebp - 64]

// CHECK: vcomsbf16 xmm2, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x51,0x7f]
          vcomsbf16 xmm2, word ptr [ecx + 254]

// CHECK: vcomsbf16 xmm2, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x52,0x80]
          vcomsbf16 xmm2, word ptr [edx - 256]

// CHECK: vdivnepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5e,0xd4]
          vdivnepbf16 ymm2, ymm3, ymm4

// CHECK: vdivnepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5e,0xd4]
          vdivnepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vdivnepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5e,0xd4]
          vdivnepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vdivnepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5e,0xd4]
          vdivnepbf16 zmm2, zmm3, zmm4

// CHECK: vdivnepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5e,0xd4]
          vdivnepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vdivnepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5e,0xd4]
          vdivnepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vdivnepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5e,0xd4]
          vdivnepbf16 xmm2, xmm3, xmm4

// CHECK: vdivnepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5e,0xd4]
          vdivnepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vdivnepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5e,0xd4]
          vdivnepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vdivnepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vdivnepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vdivnepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5e,0x94,0x87,0x23,0x01,0x00,0x00]
          vdivnepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vdivnepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x5e,0x10]
          vdivnepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vdivnepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5e,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vdivnepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vdivnepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5e,0x51,0x7f]
          vdivnepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vdivnepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x5e,0x52,0x80]
          vdivnepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vdivnepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vdivnepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vdivnepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5e,0x94,0x87,0x23,0x01,0x00,0x00]
          vdivnepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vdivnepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x5e,0x10]
          vdivnepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vdivnepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5e,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vdivnepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vdivnepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5e,0x51,0x7f]
          vdivnepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vdivnepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x5e,0x52,0x80]
          vdivnepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vdivnepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vdivnepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vdivnepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5e,0x94,0x87,0x23,0x01,0x00,0x00]
          vdivnepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vdivnepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x5e,0x10]
          vdivnepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vdivnepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5e,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vdivnepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vdivnepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5e,0x51,0x7f]
          vdivnepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vdivnepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x5e,0x52,0x80]
          vdivnepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfmadd132nepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x98,0xd4]
          vfmadd132nepbf16 ymm2, ymm3, ymm4

// CHECK: vfmadd132nepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x98,0xd4]
          vfmadd132nepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfmadd132nepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x98,0xd4]
          vfmadd132nepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfmadd132nepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x98,0xd4]
          vfmadd132nepbf16 zmm2, zmm3, zmm4

// CHECK: vfmadd132nepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x98,0xd4]
          vfmadd132nepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfmadd132nepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x98,0xd4]
          vfmadd132nepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfmadd132nepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x98,0xd4]
          vfmadd132nepbf16 xmm2, xmm3, xmm4

// CHECK: vfmadd132nepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x98,0xd4]
          vfmadd132nepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfmadd132nepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x98,0xd4]
          vfmadd132nepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfmadd132nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x98,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd132nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd132nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x98,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd132nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd132nepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x98,0x10]
          vfmadd132nepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfmadd132nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x98,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd132nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfmadd132nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x98,0x51,0x7f]
          vfmadd132nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfmadd132nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x98,0x52,0x80]
          vfmadd132nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfmadd132nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x98,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd132nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd132nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x98,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd132nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd132nepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x98,0x10]
          vfmadd132nepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfmadd132nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x98,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd132nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfmadd132nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x98,0x51,0x7f]
          vfmadd132nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfmadd132nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x98,0x52,0x80]
          vfmadd132nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfmadd132nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x98,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd132nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd132nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x98,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd132nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd132nepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x98,0x10]
          vfmadd132nepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfmadd132nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x98,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd132nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfmadd132nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x98,0x51,0x7f]
          vfmadd132nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfmadd132nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x98,0x52,0x80]
          vfmadd132nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfmadd213nepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xa8,0xd4]
          vfmadd213nepbf16 ymm2, ymm3, ymm4

// CHECK: vfmadd213nepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xa8,0xd4]
          vfmadd213nepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfmadd213nepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xa8,0xd4]
          vfmadd213nepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfmadd213nepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xa8,0xd4]
          vfmadd213nepbf16 zmm2, zmm3, zmm4

// CHECK: vfmadd213nepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xa8,0xd4]
          vfmadd213nepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfmadd213nepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xa8,0xd4]
          vfmadd213nepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfmadd213nepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xa8,0xd4]
          vfmadd213nepbf16 xmm2, xmm3, xmm4

// CHECK: vfmadd213nepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xa8,0xd4]
          vfmadd213nepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfmadd213nepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xa8,0xd4]
          vfmadd213nepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfmadd213nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xa8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd213nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd213nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xa8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd213nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd213nepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xa8,0x10]
          vfmadd213nepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfmadd213nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xa8,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd213nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfmadd213nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xa8,0x51,0x7f]
          vfmadd213nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfmadd213nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xa8,0x52,0x80]
          vfmadd213nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfmadd213nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xa8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd213nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd213nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xa8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd213nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd213nepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xa8,0x10]
          vfmadd213nepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfmadd213nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xa8,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd213nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfmadd213nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xa8,0x51,0x7f]
          vfmadd213nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfmadd213nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xa8,0x52,0x80]
          vfmadd213nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfmadd213nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xa8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd213nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd213nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xa8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd213nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd213nepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xa8,0x10]
          vfmadd213nepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfmadd213nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xa8,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd213nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfmadd213nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xa8,0x51,0x7f]
          vfmadd213nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfmadd213nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xa8,0x52,0x80]
          vfmadd213nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfmadd231nepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xb8,0xd4]
          vfmadd231nepbf16 ymm2, ymm3, ymm4

// CHECK: vfmadd231nepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xb8,0xd4]
          vfmadd231nepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfmadd231nepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xb8,0xd4]
          vfmadd231nepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfmadd231nepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xb8,0xd4]
          vfmadd231nepbf16 zmm2, zmm3, zmm4

// CHECK: vfmadd231nepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xb8,0xd4]
          vfmadd231nepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfmadd231nepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xb8,0xd4]
          vfmadd231nepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfmadd231nepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xb8,0xd4]
          vfmadd231nepbf16 xmm2, xmm3, xmm4

// CHECK: vfmadd231nepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xb8,0xd4]
          vfmadd231nepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfmadd231nepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xb8,0xd4]
          vfmadd231nepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfmadd231nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xb8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd231nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd231nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xb8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd231nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd231nepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xb8,0x10]
          vfmadd231nepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfmadd231nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xb8,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd231nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfmadd231nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xb8,0x51,0x7f]
          vfmadd231nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfmadd231nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xb8,0x52,0x80]
          vfmadd231nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfmadd231nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xb8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd231nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd231nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xb8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd231nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd231nepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xb8,0x10]
          vfmadd231nepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfmadd231nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xb8,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd231nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfmadd231nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xb8,0x51,0x7f]
          vfmadd231nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfmadd231nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xb8,0x52,0x80]
          vfmadd231nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfmadd231nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xb8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd231nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd231nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xb8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd231nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfmadd231nepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xb8,0x10]
          vfmadd231nepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfmadd231nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xb8,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd231nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfmadd231nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xb8,0x51,0x7f]
          vfmadd231nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfmadd231nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xb8,0x52,0x80]
          vfmadd231nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfmsub132nepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9a,0xd4]
          vfmsub132nepbf16 ymm2, ymm3, ymm4

// CHECK: vfmsub132nepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9a,0xd4]
          vfmsub132nepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfmsub132nepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9a,0xd4]
          vfmsub132nepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfmsub132nepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9a,0xd4]
          vfmsub132nepbf16 zmm2, zmm3, zmm4

// CHECK: vfmsub132nepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9a,0xd4]
          vfmsub132nepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfmsub132nepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9a,0xd4]
          vfmsub132nepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfmsub132nepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9a,0xd4]
          vfmsub132nepbf16 xmm2, xmm3, xmm4

// CHECK: vfmsub132nepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9a,0xd4]
          vfmsub132nepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfmsub132nepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9a,0xd4]
          vfmsub132nepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfmsub132nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub132nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub132nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9a,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub132nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub132nepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x9a,0x10]
          vfmsub132nepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfmsub132nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9a,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub132nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfmsub132nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9a,0x51,0x7f]
          vfmsub132nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfmsub132nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x9a,0x52,0x80]
          vfmsub132nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfmsub132nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub132nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub132nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9a,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub132nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub132nepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x9a,0x10]
          vfmsub132nepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfmsub132nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9a,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub132nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfmsub132nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9a,0x51,0x7f]
          vfmsub132nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfmsub132nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x9a,0x52,0x80]
          vfmsub132nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfmsub132nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub132nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub132nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9a,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub132nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub132nepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x9a,0x10]
          vfmsub132nepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfmsub132nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9a,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub132nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfmsub132nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9a,0x51,0x7f]
          vfmsub132nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfmsub132nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x9a,0x52,0x80]
          vfmsub132nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfmsub213nepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xaa,0xd4]
          vfmsub213nepbf16 ymm2, ymm3, ymm4

// CHECK: vfmsub213nepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xaa,0xd4]
          vfmsub213nepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfmsub213nepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xaa,0xd4]
          vfmsub213nepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfmsub213nepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xaa,0xd4]
          vfmsub213nepbf16 zmm2, zmm3, zmm4

// CHECK: vfmsub213nepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xaa,0xd4]
          vfmsub213nepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfmsub213nepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xaa,0xd4]
          vfmsub213nepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfmsub213nepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xaa,0xd4]
          vfmsub213nepbf16 xmm2, xmm3, xmm4

// CHECK: vfmsub213nepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xaa,0xd4]
          vfmsub213nepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfmsub213nepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xaa,0xd4]
          vfmsub213nepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfmsub213nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xaa,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub213nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub213nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xaa,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub213nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub213nepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xaa,0x10]
          vfmsub213nepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfmsub213nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xaa,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub213nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfmsub213nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xaa,0x51,0x7f]
          vfmsub213nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfmsub213nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xaa,0x52,0x80]
          vfmsub213nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfmsub213nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xaa,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub213nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub213nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xaa,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub213nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub213nepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xaa,0x10]
          vfmsub213nepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfmsub213nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xaa,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub213nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfmsub213nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xaa,0x51,0x7f]
          vfmsub213nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfmsub213nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xaa,0x52,0x80]
          vfmsub213nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfmsub213nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xaa,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub213nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub213nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xaa,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub213nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub213nepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xaa,0x10]
          vfmsub213nepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfmsub213nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xaa,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub213nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfmsub213nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xaa,0x51,0x7f]
          vfmsub213nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfmsub213nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xaa,0x52,0x80]
          vfmsub213nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfmsub231nepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xba,0xd4]
          vfmsub231nepbf16 ymm2, ymm3, ymm4

// CHECK: vfmsub231nepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xba,0xd4]
          vfmsub231nepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfmsub231nepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xba,0xd4]
          vfmsub231nepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfmsub231nepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xba,0xd4]
          vfmsub231nepbf16 zmm2, zmm3, zmm4

// CHECK: vfmsub231nepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xba,0xd4]
          vfmsub231nepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfmsub231nepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xba,0xd4]
          vfmsub231nepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfmsub231nepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xba,0xd4]
          vfmsub231nepbf16 xmm2, xmm3, xmm4

// CHECK: vfmsub231nepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xba,0xd4]
          vfmsub231nepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfmsub231nepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xba,0xd4]
          vfmsub231nepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfmsub231nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xba,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub231nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub231nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xba,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub231nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub231nepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xba,0x10]
          vfmsub231nepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfmsub231nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xba,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub231nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfmsub231nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xba,0x51,0x7f]
          vfmsub231nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfmsub231nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xba,0x52,0x80]
          vfmsub231nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfmsub231nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xba,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub231nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub231nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xba,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub231nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub231nepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xba,0x10]
          vfmsub231nepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfmsub231nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xba,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub231nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfmsub231nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xba,0x51,0x7f]
          vfmsub231nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfmsub231nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xba,0x52,0x80]
          vfmsub231nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfmsub231nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xba,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub231nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub231nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xba,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub231nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfmsub231nepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xba,0x10]
          vfmsub231nepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfmsub231nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xba,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub231nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfmsub231nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xba,0x51,0x7f]
          vfmsub231nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfmsub231nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xba,0x52,0x80]
          vfmsub231nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfnmadd132nepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9c,0xd4]
          vfnmadd132nepbf16 ymm2, ymm3, ymm4

// CHECK: vfnmadd132nepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9c,0xd4]
          vfnmadd132nepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfnmadd132nepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9c,0xd4]
          vfnmadd132nepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfnmadd132nepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9c,0xd4]
          vfnmadd132nepbf16 zmm2, zmm3, zmm4

// CHECK: vfnmadd132nepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9c,0xd4]
          vfnmadd132nepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfnmadd132nepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9c,0xd4]
          vfnmadd132nepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfnmadd132nepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9c,0xd4]
          vfnmadd132nepbf16 xmm2, xmm3, xmm4

// CHECK: vfnmadd132nepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9c,0xd4]
          vfnmadd132nepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfnmadd132nepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9c,0xd4]
          vfnmadd132nepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfnmadd132nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd132nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd132nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9c,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd132nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd132nepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x9c,0x10]
          vfnmadd132nepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfnmadd132nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd132nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfnmadd132nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9c,0x51,0x7f]
          vfnmadd132nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfnmadd132nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x9c,0x52,0x80]
          vfnmadd132nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfnmadd132nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd132nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd132nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9c,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd132nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd132nepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x9c,0x10]
          vfnmadd132nepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfnmadd132nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd132nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfnmadd132nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9c,0x51,0x7f]
          vfnmadd132nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfnmadd132nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x9c,0x52,0x80]
          vfnmadd132nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfnmadd132nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd132nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd132nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9c,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd132nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd132nepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x9c,0x10]
          vfnmadd132nepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfnmadd132nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd132nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfnmadd132nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9c,0x51,0x7f]
          vfnmadd132nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfnmadd132nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x9c,0x52,0x80]
          vfnmadd132nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfnmadd213nepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xac,0xd4]
          vfnmadd213nepbf16 ymm2, ymm3, ymm4

// CHECK: vfnmadd213nepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xac,0xd4]
          vfnmadd213nepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfnmadd213nepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xac,0xd4]
          vfnmadd213nepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfnmadd213nepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xac,0xd4]
          vfnmadd213nepbf16 zmm2, zmm3, zmm4

// CHECK: vfnmadd213nepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xac,0xd4]
          vfnmadd213nepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfnmadd213nepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xac,0xd4]
          vfnmadd213nepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfnmadd213nepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xac,0xd4]
          vfnmadd213nepbf16 xmm2, xmm3, xmm4

// CHECK: vfnmadd213nepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xac,0xd4]
          vfnmadd213nepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfnmadd213nepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xac,0xd4]
          vfnmadd213nepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfnmadd213nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xac,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd213nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd213nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xac,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd213nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd213nepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xac,0x10]
          vfnmadd213nepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfnmadd213nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xac,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd213nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfnmadd213nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xac,0x51,0x7f]
          vfnmadd213nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfnmadd213nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xac,0x52,0x80]
          vfnmadd213nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfnmadd213nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xac,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd213nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd213nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xac,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd213nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd213nepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xac,0x10]
          vfnmadd213nepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfnmadd213nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xac,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd213nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfnmadd213nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xac,0x51,0x7f]
          vfnmadd213nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfnmadd213nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xac,0x52,0x80]
          vfnmadd213nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfnmadd213nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xac,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd213nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd213nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xac,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd213nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd213nepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xac,0x10]
          vfnmadd213nepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfnmadd213nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xac,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd213nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfnmadd213nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xac,0x51,0x7f]
          vfnmadd213nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfnmadd213nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xac,0x52,0x80]
          vfnmadd213nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfnmadd231nepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbc,0xd4]
          vfnmadd231nepbf16 ymm2, ymm3, ymm4

// CHECK: vfnmadd231nepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xbc,0xd4]
          vfnmadd231nepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfnmadd231nepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xbc,0xd4]
          vfnmadd231nepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfnmadd231nepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbc,0xd4]
          vfnmadd231nepbf16 zmm2, zmm3, zmm4

// CHECK: vfnmadd231nepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xbc,0xd4]
          vfnmadd231nepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfnmadd231nepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xbc,0xd4]
          vfnmadd231nepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfnmadd231nepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbc,0xd4]
          vfnmadd231nepbf16 xmm2, xmm3, xmm4

// CHECK: vfnmadd231nepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xbc,0xd4]
          vfnmadd231nepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfnmadd231nepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xbc,0xd4]
          vfnmadd231nepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfnmadd231nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbc,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd231nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd231nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xbc,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd231nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd231nepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xbc,0x10]
          vfnmadd231nepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfnmadd231nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbc,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd231nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfnmadd231nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xbc,0x51,0x7f]
          vfnmadd231nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfnmadd231nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xbc,0x52,0x80]
          vfnmadd231nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfnmadd231nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbc,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd231nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd231nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xbc,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd231nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd231nepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xbc,0x10]
          vfnmadd231nepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfnmadd231nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbc,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd231nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfnmadd231nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xbc,0x51,0x7f]
          vfnmadd231nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfnmadd231nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xbc,0x52,0x80]
          vfnmadd231nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfnmadd231nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbc,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd231nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd231nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xbc,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd231nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmadd231nepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xbc,0x10]
          vfnmadd231nepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfnmadd231nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbc,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd231nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfnmadd231nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xbc,0x51,0x7f]
          vfnmadd231nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfnmadd231nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xbc,0x52,0x80]
          vfnmadd231nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfnmsub132nepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9e,0xd4]
          vfnmsub132nepbf16 ymm2, ymm3, ymm4

// CHECK: vfnmsub132nepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9e,0xd4]
          vfnmsub132nepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfnmsub132nepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9e,0xd4]
          vfnmsub132nepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfnmsub132nepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9e,0xd4]
          vfnmsub132nepbf16 zmm2, zmm3, zmm4

// CHECK: vfnmsub132nepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9e,0xd4]
          vfnmsub132nepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfnmsub132nepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9e,0xd4]
          vfnmsub132nepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfnmsub132nepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9e,0xd4]
          vfnmsub132nepbf16 xmm2, xmm3, xmm4

// CHECK: vfnmsub132nepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9e,0xd4]
          vfnmsub132nepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfnmsub132nepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9e,0xd4]
          vfnmsub132nepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfnmsub132nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub132nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub132nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9e,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub132nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub132nepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x9e,0x10]
          vfnmsub132nepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfnmsub132nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9e,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub132nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfnmsub132nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9e,0x51,0x7f]
          vfnmsub132nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfnmsub132nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x9e,0x52,0x80]
          vfnmsub132nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfnmsub132nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub132nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub132nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9e,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub132nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub132nepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x9e,0x10]
          vfnmsub132nepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfnmsub132nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9e,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub132nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfnmsub132nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9e,0x51,0x7f]
          vfnmsub132nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfnmsub132nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x9e,0x52,0x80]
          vfnmsub132nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfnmsub132nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub132nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub132nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9e,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub132nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub132nepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x9e,0x10]
          vfnmsub132nepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfnmsub132nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9e,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub132nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfnmsub132nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9e,0x51,0x7f]
          vfnmsub132nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfnmsub132nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x9e,0x52,0x80]
          vfnmsub132nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfnmsub213nepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xae,0xd4]
          vfnmsub213nepbf16 ymm2, ymm3, ymm4

// CHECK: vfnmsub213nepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xae,0xd4]
          vfnmsub213nepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfnmsub213nepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xae,0xd4]
          vfnmsub213nepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfnmsub213nepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xae,0xd4]
          vfnmsub213nepbf16 zmm2, zmm3, zmm4

// CHECK: vfnmsub213nepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xae,0xd4]
          vfnmsub213nepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfnmsub213nepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xae,0xd4]
          vfnmsub213nepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfnmsub213nepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xae,0xd4]
          vfnmsub213nepbf16 xmm2, xmm3, xmm4

// CHECK: vfnmsub213nepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xae,0xd4]
          vfnmsub213nepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfnmsub213nepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xae,0xd4]
          vfnmsub213nepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfnmsub213nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xae,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub213nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub213nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xae,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub213nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub213nepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xae,0x10]
          vfnmsub213nepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfnmsub213nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xae,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub213nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfnmsub213nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xae,0x51,0x7f]
          vfnmsub213nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfnmsub213nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xae,0x52,0x80]
          vfnmsub213nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfnmsub213nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xae,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub213nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub213nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xae,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub213nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub213nepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xae,0x10]
          vfnmsub213nepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfnmsub213nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xae,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub213nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfnmsub213nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xae,0x51,0x7f]
          vfnmsub213nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfnmsub213nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xae,0x52,0x80]
          vfnmsub213nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfnmsub213nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xae,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub213nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub213nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xae,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub213nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub213nepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xae,0x10]
          vfnmsub213nepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfnmsub213nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xae,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub213nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfnmsub213nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xae,0x51,0x7f]
          vfnmsub213nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfnmsub213nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xae,0x52,0x80]
          vfnmsub213nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfnmsub231nepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbe,0xd4]
          vfnmsub231nepbf16 ymm2, ymm3, ymm4

// CHECK: vfnmsub231nepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xbe,0xd4]
          vfnmsub231nepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vfnmsub231nepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xbe,0xd4]
          vfnmsub231nepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vfnmsub231nepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbe,0xd4]
          vfnmsub231nepbf16 zmm2, zmm3, zmm4

// CHECK: vfnmsub231nepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xbe,0xd4]
          vfnmsub231nepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vfnmsub231nepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xbe,0xd4]
          vfnmsub231nepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vfnmsub231nepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbe,0xd4]
          vfnmsub231nepbf16 xmm2, xmm3, xmm4

// CHECK: vfnmsub231nepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xbe,0xd4]
          vfnmsub231nepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vfnmsub231nepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xbe,0xd4]
          vfnmsub231nepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vfnmsub231nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbe,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub231nepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub231nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xbe,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub231nepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub231nepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xbe,0x10]
          vfnmsub231nepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vfnmsub231nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbe,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub231nepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vfnmsub231nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xbe,0x51,0x7f]
          vfnmsub231nepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vfnmsub231nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xbe,0x52,0x80]
          vfnmsub231nepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vfnmsub231nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbe,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub231nepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub231nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xbe,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub231nepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub231nepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xbe,0x10]
          vfnmsub231nepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vfnmsub231nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbe,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub231nepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vfnmsub231nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xbe,0x51,0x7f]
          vfnmsub231nepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vfnmsub231nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xbe,0x52,0x80]
          vfnmsub231nepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vfnmsub231nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbe,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub231nepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub231nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xbe,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub231nepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vfnmsub231nepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xbe,0x10]
          vfnmsub231nepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vfnmsub231nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbe,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub231nepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vfnmsub231nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xbe,0x51,0x7f]
          vfnmsub231nepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vfnmsub231nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xbe,0x52,0x80]
          vfnmsub231nepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vfpclasspbf16 k5, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x66,0xeb,0x7b]
          vfpclasspbf16 k5, zmm3, 123

// CHECK: vfpclasspbf16 k5 {k7}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x66,0xeb,0x7b]
          vfpclasspbf16 k5 {k7}, zmm3, 123

// CHECK: vfpclasspbf16 k5, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x66,0xeb,0x7b]
          vfpclasspbf16 k5, ymm3, 123

// CHECK: vfpclasspbf16 k5 {k7}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x66,0xeb,0x7b]
          vfpclasspbf16 k5 {k7}, ymm3, 123

// CHECK: vfpclasspbf16 k5, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x66,0xeb,0x7b]
          vfpclasspbf16 k5, xmm3, 123

// CHECK: vfpclasspbf16 k5 {k7}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x66,0xeb,0x7b]
          vfpclasspbf16 k5 {k7}, xmm3, 123

// CHECK: vfpclasspbf16 k5, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x66,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vfpclasspbf16 k5, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vfpclasspbf16 k5 {k7}, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x66,0xac,0x87,0x23,0x01,0x00,0x00,0x7b]
          vfpclasspbf16 k5 {k7}, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vfpclasspbf16 k5, word ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x66,0x28,0x7b]
          vfpclasspbf16 k5, word ptr [eax]{1to8}, 123

// CHECK: vfpclasspbf16 k5, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x66,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vfpclasspbf16 k5, xmmword ptr [2*ebp - 512], 123

// CHECK: vfpclasspbf16 k5 {k7}, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x66,0x69,0x7f,0x7b]
          vfpclasspbf16 k5 {k7}, xmmword ptr [ecx + 2032], 123

// CHECK: vfpclasspbf16 k5 {k7}, word ptr [edx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x1f,0x66,0x6a,0x80,0x7b]
          vfpclasspbf16 k5 {k7}, word ptr [edx - 256]{1to8}, 123

// CHECK: vfpclasspbf16 k5, word ptr [eax]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x66,0x28,0x7b]
          vfpclasspbf16 k5, word ptr [eax]{1to16}, 123

// CHECK: vfpclasspbf16 k5, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x66,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vfpclasspbf16 k5, ymmword ptr [2*ebp - 1024], 123

// CHECK: vfpclasspbf16 k5 {k7}, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x66,0x69,0x7f,0x7b]
          vfpclasspbf16 k5 {k7}, ymmword ptr [ecx + 4064], 123

// CHECK: vfpclasspbf16 k5 {k7}, word ptr [edx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x3f,0x66,0x6a,0x80,0x7b]
          vfpclasspbf16 k5 {k7}, word ptr [edx - 256]{1to16}, 123

// CHECK: vfpclasspbf16 k5, word ptr [eax]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x66,0x28,0x7b]
          vfpclasspbf16 k5, word ptr [eax]{1to32}, 123

// CHECK: vfpclasspbf16 k5, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x66,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vfpclasspbf16 k5, zmmword ptr [2*ebp - 2048], 123

// CHECK: vfpclasspbf16 k5 {k7}, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x66,0x69,0x7f,0x7b]
          vfpclasspbf16 k5 {k7}, zmmword ptr [ecx + 8128], 123

// CHECK: vfpclasspbf16 k5 {k7}, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x5f,0x66,0x6a,0x80,0x7b]
          vfpclasspbf16 k5 {k7}, word ptr [edx - 256]{1to32}, 123

// CHECK: vgetexppbf16 xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x42,0xd3]
          vgetexppbf16 xmm2, xmm3

// CHECK: vgetexppbf16 xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x42,0xd3]
          vgetexppbf16 xmm2 {k7}, xmm3

// CHECK: vgetexppbf16 xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x42,0xd3]
          vgetexppbf16 xmm2 {k7} {z}, xmm3

// CHECK: vgetexppbf16 zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x42,0xd3]
          vgetexppbf16 zmm2, zmm3

// CHECK: vgetexppbf16 zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x42,0xd3]
          vgetexppbf16 zmm2 {k7}, zmm3

// CHECK: vgetexppbf16 zmm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x42,0xd3]
          vgetexppbf16 zmm2 {k7} {z}, zmm3

// CHECK: vgetexppbf16 ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x42,0xd3]
          vgetexppbf16 ymm2, ymm3

// CHECK: vgetexppbf16 ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x42,0xd3]
          vgetexppbf16 ymm2 {k7}, ymm3

// CHECK: vgetexppbf16 ymm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x42,0xd3]
          vgetexppbf16 ymm2 {k7} {z}, ymm3

// CHECK: vgetexppbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x42,0x94,0xf4,0x00,0x00,0x00,0x10]
          vgetexppbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vgetexppbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x42,0x94,0x87,0x23,0x01,0x00,0x00]
          vgetexppbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vgetexppbf16 xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x42,0x10]
          vgetexppbf16 xmm2, word ptr [eax]{1to8}

// CHECK: vgetexppbf16 xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x42,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vgetexppbf16 xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vgetexppbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x42,0x51,0x7f]
          vgetexppbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vgetexppbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x42,0x52,0x80]
          vgetexppbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vgetexppbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x42,0x94,0xf4,0x00,0x00,0x00,0x10]
          vgetexppbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vgetexppbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x42,0x94,0x87,0x23,0x01,0x00,0x00]
          vgetexppbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vgetexppbf16 ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x42,0x10]
          vgetexppbf16 ymm2, word ptr [eax]{1to16}

// CHECK: vgetexppbf16 ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x42,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vgetexppbf16 ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vgetexppbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x42,0x51,0x7f]
          vgetexppbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vgetexppbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x42,0x52,0x80]
          vgetexppbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vgetexppbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x42,0x94,0xf4,0x00,0x00,0x00,0x10]
          vgetexppbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vgetexppbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x42,0x94,0x87,0x23,0x01,0x00,0x00]
          vgetexppbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vgetexppbf16 zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x42,0x10]
          vgetexppbf16 zmm2, word ptr [eax]{1to32}

// CHECK: vgetexppbf16 zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x42,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vgetexppbf16 zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vgetexppbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x42,0x51,0x7f]
          vgetexppbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vgetexppbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x42,0x52,0x80]
          vgetexppbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vgetmantpbf16 zmm2, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x26,0xd3,0x7b]
          vgetmantpbf16 zmm2, zmm3, 123

// CHECK: vgetmantpbf16 zmm2 {k7}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x26,0xd3,0x7b]
          vgetmantpbf16 zmm2 {k7}, zmm3, 123

// CHECK: vgetmantpbf16 zmm2 {k7} {z}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x26,0xd3,0x7b]
          vgetmantpbf16 zmm2 {k7} {z}, zmm3, 123

// CHECK: vgetmantpbf16 ymm2, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x26,0xd3,0x7b]
          vgetmantpbf16 ymm2, ymm3, 123

// CHECK: vgetmantpbf16 ymm2 {k7}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x26,0xd3,0x7b]
          vgetmantpbf16 ymm2 {k7}, ymm3, 123

// CHECK: vgetmantpbf16 ymm2 {k7} {z}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x26,0xd3,0x7b]
          vgetmantpbf16 ymm2 {k7} {z}, ymm3, 123

// CHECK: vgetmantpbf16 xmm2, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x26,0xd3,0x7b]
          vgetmantpbf16 xmm2, xmm3, 123

// CHECK: vgetmantpbf16 xmm2 {k7}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x26,0xd3,0x7b]
          vgetmantpbf16 xmm2 {k7}, xmm3, 123

// CHECK: vgetmantpbf16 xmm2 {k7} {z}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x26,0xd3,0x7b]
          vgetmantpbf16 xmm2 {k7} {z}, xmm3, 123

// CHECK: vgetmantpbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x26,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vgetmantpbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vgetmantpbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x26,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vgetmantpbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vgetmantpbf16 xmm2, word ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x26,0x10,0x7b]
          vgetmantpbf16 xmm2, word ptr [eax]{1to8}, 123

// CHECK: vgetmantpbf16 xmm2, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x26,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vgetmantpbf16 xmm2, xmmword ptr [2*ebp - 512], 123

// CHECK: vgetmantpbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x26,0x51,0x7f,0x7b]
          vgetmantpbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032], 123

// CHECK: vgetmantpbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x9f,0x26,0x52,0x80,0x7b]
          vgetmantpbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}, 123

// CHECK: vgetmantpbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x26,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vgetmantpbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vgetmantpbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x26,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vgetmantpbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vgetmantpbf16 ymm2, word ptr [eax]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x26,0x10,0x7b]
          vgetmantpbf16 ymm2, word ptr [eax]{1to16}, 123

// CHECK: vgetmantpbf16 ymm2, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x26,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vgetmantpbf16 ymm2, ymmword ptr [2*ebp - 1024], 123

// CHECK: vgetmantpbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x26,0x51,0x7f,0x7b]
          vgetmantpbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064], 123

// CHECK: vgetmantpbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xbf,0x26,0x52,0x80,0x7b]
          vgetmantpbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}, 123

// CHECK: vgetmantpbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x26,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vgetmantpbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vgetmantpbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x26,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vgetmantpbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vgetmantpbf16 zmm2, word ptr [eax]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x26,0x10,0x7b]
          vgetmantpbf16 zmm2, word ptr [eax]{1to32}, 123

// CHECK: vgetmantpbf16 zmm2, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x26,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vgetmantpbf16 zmm2, zmmword ptr [2*ebp - 2048], 123

// CHECK: vgetmantpbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x26,0x51,0x7f,0x7b]
          vgetmantpbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128], 123

// CHECK: vgetmantpbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xdf,0x26,0x52,0x80,0x7b]
          vgetmantpbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}, 123

// CHECK: vmaxpbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5f,0xd4]
          vmaxpbf16 ymm2, ymm3, ymm4

// CHECK: vmaxpbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5f,0xd4]
          vmaxpbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vmaxpbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5f,0xd4]
          vmaxpbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vmaxpbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5f,0xd4]
          vmaxpbf16 zmm2, zmm3, zmm4

// CHECK: vmaxpbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5f,0xd4]
          vmaxpbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vmaxpbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5f,0xd4]
          vmaxpbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vmaxpbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5f,0xd4]
          vmaxpbf16 xmm2, xmm3, xmm4

// CHECK: vmaxpbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5f,0xd4]
          vmaxpbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vmaxpbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5f,0xd4]
          vmaxpbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vmaxpbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5f,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmaxpbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vmaxpbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5f,0x94,0x87,0x23,0x01,0x00,0x00]
          vmaxpbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vmaxpbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x5f,0x10]
          vmaxpbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vmaxpbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5f,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vmaxpbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vmaxpbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5f,0x51,0x7f]
          vmaxpbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vmaxpbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x5f,0x52,0x80]
          vmaxpbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vmaxpbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5f,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmaxpbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vmaxpbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5f,0x94,0x87,0x23,0x01,0x00,0x00]
          vmaxpbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vmaxpbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x5f,0x10]
          vmaxpbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vmaxpbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5f,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vmaxpbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vmaxpbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5f,0x51,0x7f]
          vmaxpbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vmaxpbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x5f,0x52,0x80]
          vmaxpbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vmaxpbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5f,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmaxpbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vmaxpbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5f,0x94,0x87,0x23,0x01,0x00,0x00]
          vmaxpbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vmaxpbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x5f,0x10]
          vmaxpbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vmaxpbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5f,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vmaxpbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vmaxpbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5f,0x51,0x7f]
          vmaxpbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vmaxpbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x5f,0x52,0x80]
          vmaxpbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vminpbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5d,0xd4]
          vminpbf16 ymm2, ymm3, ymm4

// CHECK: vminpbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5d,0xd4]
          vminpbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vminpbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5d,0xd4]
          vminpbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vminpbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5d,0xd4]
          vminpbf16 zmm2, zmm3, zmm4

// CHECK: vminpbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5d,0xd4]
          vminpbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vminpbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5d,0xd4]
          vminpbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vminpbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5d,0xd4]
          vminpbf16 xmm2, xmm3, xmm4

// CHECK: vminpbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5d,0xd4]
          vminpbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vminpbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5d,0xd4]
          vminpbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vminpbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vminpbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vminpbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5d,0x94,0x87,0x23,0x01,0x00,0x00]
          vminpbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vminpbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x5d,0x10]
          vminpbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vminpbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5d,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vminpbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vminpbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5d,0x51,0x7f]
          vminpbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vminpbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x5d,0x52,0x80]
          vminpbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vminpbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vminpbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vminpbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5d,0x94,0x87,0x23,0x01,0x00,0x00]
          vminpbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vminpbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x5d,0x10]
          vminpbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vminpbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5d,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vminpbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vminpbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5d,0x51,0x7f]
          vminpbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vminpbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x5d,0x52,0x80]
          vminpbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vminpbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vminpbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vminpbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5d,0x94,0x87,0x23,0x01,0x00,0x00]
          vminpbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vminpbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x5d,0x10]
          vminpbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vminpbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5d,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vminpbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vminpbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5d,0x51,0x7f]
          vminpbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vminpbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x5d,0x52,0x80]
          vminpbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vmulnepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x59,0xd4]
          vmulnepbf16 ymm2, ymm3, ymm4

// CHECK: vmulnepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x59,0xd4]
          vmulnepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vmulnepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x59,0xd4]
          vmulnepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vmulnepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x59,0xd4]
          vmulnepbf16 zmm2, zmm3, zmm4

// CHECK: vmulnepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x59,0xd4]
          vmulnepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vmulnepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x59,0xd4]
          vmulnepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vmulnepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x59,0xd4]
          vmulnepbf16 xmm2, xmm3, xmm4

// CHECK: vmulnepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x59,0xd4]
          vmulnepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vmulnepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x59,0xd4]
          vmulnepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vmulnepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x59,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmulnepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vmulnepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x59,0x94,0x87,0x23,0x01,0x00,0x00]
          vmulnepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vmulnepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x59,0x10]
          vmulnepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vmulnepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x59,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vmulnepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vmulnepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x59,0x51,0x7f]
          vmulnepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vmulnepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x59,0x52,0x80]
          vmulnepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vmulnepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x59,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmulnepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vmulnepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x59,0x94,0x87,0x23,0x01,0x00,0x00]
          vmulnepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vmulnepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x59,0x10]
          vmulnepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vmulnepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x59,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vmulnepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vmulnepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x59,0x51,0x7f]
          vmulnepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vmulnepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x59,0x52,0x80]
          vmulnepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vmulnepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x59,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmulnepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vmulnepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x59,0x94,0x87,0x23,0x01,0x00,0x00]
          vmulnepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vmulnepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x59,0x10]
          vmulnepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vmulnepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x59,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vmulnepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vmulnepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x59,0x51,0x7f]
          vmulnepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vmulnepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x59,0x52,0x80]
          vmulnepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vrcppbf16 xmm2, xmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4c,0xd3]
          vrcppbf16 xmm2, xmm3

// CHECK: vrcppbf16 xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x0f,0x4c,0xd3]
          vrcppbf16 xmm2 {k7}, xmm3

// CHECK: vrcppbf16 xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x8f,0x4c,0xd3]
          vrcppbf16 xmm2 {k7} {z}, xmm3

// CHECK: vrcppbf16 zmm2, zmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4c,0xd3]
          vrcppbf16 zmm2, zmm3

// CHECK: vrcppbf16 zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x4c,0xd3]
          vrcppbf16 zmm2 {k7}, zmm3

// CHECK: vrcppbf16 zmm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x4c,0xd3]
          vrcppbf16 zmm2 {k7} {z}, zmm3

// CHECK: vrcppbf16 ymm2, ymm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4c,0xd3]
          vrcppbf16 ymm2, ymm3

// CHECK: vrcppbf16 ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x2f,0x4c,0xd3]
          vrcppbf16 ymm2 {k7}, ymm3

// CHECK: vrcppbf16 ymm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf6,0x7c,0xaf,0x4c,0xd3]
          vrcppbf16 ymm2 {k7} {z}, ymm3

// CHECK: vrcppbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrcppbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vrcppbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x7c,0x0f,0x4c,0x94,0x87,0x23,0x01,0x00,0x00]
          vrcppbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vrcppbf16 xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x7c,0x18,0x4c,0x10]
          vrcppbf16 xmm2, word ptr [eax]{1to8}

// CHECK: vrcppbf16 xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vrcppbf16 xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vrcppbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x7c,0x8f,0x4c,0x51,0x7f]
          vrcppbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vrcppbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x7c,0x9f,0x4c,0x52,0x80]
          vrcppbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vrcppbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrcppbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vrcppbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x7c,0x2f,0x4c,0x94,0x87,0x23,0x01,0x00,0x00]
          vrcppbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vrcppbf16 ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x7c,0x38,0x4c,0x10]
          vrcppbf16 ymm2, word ptr [eax]{1to16}

// CHECK: vrcppbf16 ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vrcppbf16 ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vrcppbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x7c,0xaf,0x4c,0x51,0x7f]
          vrcppbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vrcppbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x7c,0xbf,0x4c,0x52,0x80]
          vrcppbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vrcppbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrcppbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vrcppbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x4c,0x94,0x87,0x23,0x01,0x00,0x00]
          vrcppbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vrcppbf16 zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7c,0x58,0x4c,0x10]
          vrcppbf16 zmm2, word ptr [eax]{1to32}

// CHECK: vrcppbf16 zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vrcppbf16 zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vrcppbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x4c,0x51,0x7f]
          vrcppbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vrcppbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7c,0xdf,0x4c,0x52,0x80]
          vrcppbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vreducenepbf16 zmm2, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x56,0xd3,0x7b]
          vreducenepbf16 zmm2, zmm3, 123

// CHECK: vreducenepbf16 zmm2 {k7}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x56,0xd3,0x7b]
          vreducenepbf16 zmm2 {k7}, zmm3, 123

// CHECK: vreducenepbf16 zmm2 {k7} {z}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x56,0xd3,0x7b]
          vreducenepbf16 zmm2 {k7} {z}, zmm3, 123

// CHECK: vreducenepbf16 ymm2, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x56,0xd3,0x7b]
          vreducenepbf16 ymm2, ymm3, 123

// CHECK: vreducenepbf16 ymm2 {k7}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x56,0xd3,0x7b]
          vreducenepbf16 ymm2 {k7}, ymm3, 123

// CHECK: vreducenepbf16 ymm2 {k7} {z}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x56,0xd3,0x7b]
          vreducenepbf16 ymm2 {k7} {z}, ymm3, 123

// CHECK: vreducenepbf16 xmm2, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x56,0xd3,0x7b]
          vreducenepbf16 xmm2, xmm3, 123

// CHECK: vreducenepbf16 xmm2 {k7}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x56,0xd3,0x7b]
          vreducenepbf16 xmm2 {k7}, xmm3, 123

// CHECK: vreducenepbf16 xmm2 {k7} {z}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x56,0xd3,0x7b]
          vreducenepbf16 xmm2 {k7} {z}, xmm3, 123

// CHECK: vreducenepbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x56,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vreducenepbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vreducenepbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x56,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vreducenepbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vreducenepbf16 xmm2, word ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x56,0x10,0x7b]
          vreducenepbf16 xmm2, word ptr [eax]{1to8}, 123

// CHECK: vreducenepbf16 xmm2, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x56,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vreducenepbf16 xmm2, xmmword ptr [2*ebp - 512], 123

// CHECK: vreducenepbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x56,0x51,0x7f,0x7b]
          vreducenepbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032], 123

// CHECK: vreducenepbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x9f,0x56,0x52,0x80,0x7b]
          vreducenepbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}, 123

// CHECK: vreducenepbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x56,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vreducenepbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vreducenepbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x56,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vreducenepbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vreducenepbf16 ymm2, word ptr [eax]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x56,0x10,0x7b]
          vreducenepbf16 ymm2, word ptr [eax]{1to16}, 123

// CHECK: vreducenepbf16 ymm2, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x56,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vreducenepbf16 ymm2, ymmword ptr [2*ebp - 1024], 123

// CHECK: vreducenepbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x56,0x51,0x7f,0x7b]
          vreducenepbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064], 123

// CHECK: vreducenepbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xbf,0x56,0x52,0x80,0x7b]
          vreducenepbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}, 123

// CHECK: vreducenepbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x56,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vreducenepbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vreducenepbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x56,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vreducenepbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vreducenepbf16 zmm2, word ptr [eax]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x56,0x10,0x7b]
          vreducenepbf16 zmm2, word ptr [eax]{1to32}, 123

// CHECK: vreducenepbf16 zmm2, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x56,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vreducenepbf16 zmm2, zmmword ptr [2*ebp - 2048], 123

// CHECK: vreducenepbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x56,0x51,0x7f,0x7b]
          vreducenepbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128], 123

// CHECK: vreducenepbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xdf,0x56,0x52,0x80,0x7b]
          vreducenepbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}, 123

// CHECK: vrndscalenepbf16 zmm2, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x08,0xd3,0x7b]
          vrndscalenepbf16 zmm2, zmm3, 123

// CHECK: vrndscalenepbf16 zmm2 {k7}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x08,0xd3,0x7b]
          vrndscalenepbf16 zmm2 {k7}, zmm3, 123

// CHECK: vrndscalenepbf16 zmm2 {k7} {z}, zmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x08,0xd3,0x7b]
          vrndscalenepbf16 zmm2 {k7} {z}, zmm3, 123

// CHECK: vrndscalenepbf16 ymm2, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x08,0xd3,0x7b]
          vrndscalenepbf16 ymm2, ymm3, 123

// CHECK: vrndscalenepbf16 ymm2 {k7}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x08,0xd3,0x7b]
          vrndscalenepbf16 ymm2 {k7}, ymm3, 123

// CHECK: vrndscalenepbf16 ymm2 {k7} {z}, ymm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x08,0xd3,0x7b]
          vrndscalenepbf16 ymm2 {k7} {z}, ymm3, 123

// CHECK: vrndscalenepbf16 xmm2, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x08,0xd3,0x7b]
          vrndscalenepbf16 xmm2, xmm3, 123

// CHECK: vrndscalenepbf16 xmm2 {k7}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x08,0xd3,0x7b]
          vrndscalenepbf16 xmm2 {k7}, xmm3, 123

// CHECK: vrndscalenepbf16 xmm2 {k7} {z}, xmm3, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x08,0xd3,0x7b]
          vrndscalenepbf16 xmm2 {k7} {z}, xmm3, 123

// CHECK: vrndscalenepbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x08,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vrndscalenepbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vrndscalenepbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x08,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vrndscalenepbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vrndscalenepbf16 xmm2, word ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x08,0x10,0x7b]
          vrndscalenepbf16 xmm2, word ptr [eax]{1to8}, 123

// CHECK: vrndscalenepbf16 xmm2, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x08,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vrndscalenepbf16 xmm2, xmmword ptr [2*ebp - 512], 123

// CHECK: vrndscalenepbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x08,0x51,0x7f,0x7b]
          vrndscalenepbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032], 123

// CHECK: vrndscalenepbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x9f,0x08,0x52,0x80,0x7b]
          vrndscalenepbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}, 123

// CHECK: vrndscalenepbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x08,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vrndscalenepbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vrndscalenepbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x08,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vrndscalenepbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vrndscalenepbf16 ymm2, word ptr [eax]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x08,0x10,0x7b]
          vrndscalenepbf16 ymm2, word ptr [eax]{1to16}, 123

// CHECK: vrndscalenepbf16 ymm2, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x08,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vrndscalenepbf16 ymm2, ymmword ptr [2*ebp - 1024], 123

// CHECK: vrndscalenepbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x08,0x51,0x7f,0x7b]
          vrndscalenepbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064], 123

// CHECK: vrndscalenepbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xbf,0x08,0x52,0x80,0x7b]
          vrndscalenepbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}, 123

// CHECK: vrndscalenepbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x08,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vrndscalenepbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vrndscalenepbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x08,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vrndscalenepbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vrndscalenepbf16 zmm2, word ptr [eax]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x08,0x10,0x7b]
          vrndscalenepbf16 zmm2, word ptr [eax]{1to32}, 123

// CHECK: vrndscalenepbf16 zmm2, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x08,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vrndscalenepbf16 zmm2, zmmword ptr [2*ebp - 2048], 123

// CHECK: vrndscalenepbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x08,0x51,0x7f,0x7b]
          vrndscalenepbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128], 123

// CHECK: vrndscalenepbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0xdf,0x08,0x52,0x80,0x7b]
          vrndscalenepbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}, 123

// CHECK: vrsqrtpbf16 xmm2, xmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4e,0xd3]
          vrsqrtpbf16 xmm2, xmm3

// CHECK: vrsqrtpbf16 xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x0f,0x4e,0xd3]
          vrsqrtpbf16 xmm2 {k7}, xmm3

// CHECK: vrsqrtpbf16 xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x8f,0x4e,0xd3]
          vrsqrtpbf16 xmm2 {k7} {z}, xmm3

// CHECK: vrsqrtpbf16 zmm2, zmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4e,0xd3]
          vrsqrtpbf16 zmm2, zmm3

// CHECK: vrsqrtpbf16 zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x4e,0xd3]
          vrsqrtpbf16 zmm2 {k7}, zmm3

// CHECK: vrsqrtpbf16 zmm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x4e,0xd3]
          vrsqrtpbf16 zmm2 {k7} {z}, zmm3

// CHECK: vrsqrtpbf16 ymm2, ymm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4e,0xd3]
          vrsqrtpbf16 ymm2, ymm3

// CHECK: vrsqrtpbf16 ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf6,0x7c,0x2f,0x4e,0xd3]
          vrsqrtpbf16 ymm2 {k7}, ymm3

// CHECK: vrsqrtpbf16 ymm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf6,0x7c,0xaf,0x4e,0xd3]
          vrsqrtpbf16 ymm2 {k7} {z}, ymm3

// CHECK: vrsqrtpbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrsqrtpbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vrsqrtpbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x7c,0x0f,0x4e,0x94,0x87,0x23,0x01,0x00,0x00]
          vrsqrtpbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vrsqrtpbf16 xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x7c,0x18,0x4e,0x10]
          vrsqrtpbf16 xmm2, word ptr [eax]{1to8}

// CHECK: vrsqrtpbf16 xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4e,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vrsqrtpbf16 xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vrsqrtpbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x7c,0x8f,0x4e,0x51,0x7f]
          vrsqrtpbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vrsqrtpbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x7c,0x9f,0x4e,0x52,0x80]
          vrsqrtpbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vrsqrtpbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrsqrtpbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vrsqrtpbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x7c,0x2f,0x4e,0x94,0x87,0x23,0x01,0x00,0x00]
          vrsqrtpbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vrsqrtpbf16 ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x7c,0x38,0x4e,0x10]
          vrsqrtpbf16 ymm2, word ptr [eax]{1to16}

// CHECK: vrsqrtpbf16 ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4e,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vrsqrtpbf16 ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vrsqrtpbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x7c,0xaf,0x4e,0x51,0x7f]
          vrsqrtpbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vrsqrtpbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x7c,0xbf,0x4e,0x52,0x80]
          vrsqrtpbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vrsqrtpbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrsqrtpbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vrsqrtpbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x4e,0x94,0x87,0x23,0x01,0x00,0x00]
          vrsqrtpbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vrsqrtpbf16 zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7c,0x58,0x4e,0x10]
          vrsqrtpbf16 zmm2, word ptr [eax]{1to32}

// CHECK: vrsqrtpbf16 zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4e,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vrsqrtpbf16 zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vrsqrtpbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x4e,0x51,0x7f]
          vrsqrtpbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vrsqrtpbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7c,0xdf,0x4e,0x52,0x80]
          vrsqrtpbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vscalefpbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x2c,0xd4]
          vscalefpbf16 ymm2, ymm3, ymm4

// CHECK: vscalefpbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x2c,0xd4]
          vscalefpbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vscalefpbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x2c,0xd4]
          vscalefpbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vscalefpbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x2c,0xd4]
          vscalefpbf16 zmm2, zmm3, zmm4

// CHECK: vscalefpbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x2c,0xd4]
          vscalefpbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vscalefpbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x2c,0xd4]
          vscalefpbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vscalefpbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x2c,0xd4]
          vscalefpbf16 xmm2, xmm3, xmm4

// CHECK: vscalefpbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x2c,0xd4]
          vscalefpbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vscalefpbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x2c,0xd4]
          vscalefpbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vscalefpbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x2c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vscalefpbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vscalefpbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x2c,0x94,0x87,0x23,0x01,0x00,0x00]
          vscalefpbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vscalefpbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x2c,0x10]
          vscalefpbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vscalefpbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x2c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vscalefpbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vscalefpbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x2c,0x51,0x7f]
          vscalefpbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vscalefpbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x2c,0x52,0x80]
          vscalefpbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vscalefpbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x2c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vscalefpbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vscalefpbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x2c,0x94,0x87,0x23,0x01,0x00,0x00]
          vscalefpbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vscalefpbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x2c,0x10]
          vscalefpbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vscalefpbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x2c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vscalefpbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vscalefpbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x2c,0x51,0x7f]
          vscalefpbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vscalefpbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x2c,0x52,0x80]
          vscalefpbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vscalefpbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x2c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vscalefpbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vscalefpbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x2c,0x94,0x87,0x23,0x01,0x00,0x00]
          vscalefpbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vscalefpbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x2c,0x10]
          vscalefpbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vscalefpbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x2c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vscalefpbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vscalefpbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x2c,0x51,0x7f]
          vscalefpbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vscalefpbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x2c,0x52,0x80]
          vscalefpbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vsqrtnepbf16 xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x51,0xd3]
          vsqrtnepbf16 xmm2, xmm3

// CHECK: vsqrtnepbf16 xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x51,0xd3]
          vsqrtnepbf16 xmm2 {k7}, xmm3

// CHECK: vsqrtnepbf16 xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x51,0xd3]
          vsqrtnepbf16 xmm2 {k7} {z}, xmm3

// CHECK: vsqrtnepbf16 zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x51,0xd3]
          vsqrtnepbf16 zmm2, zmm3

// CHECK: vsqrtnepbf16 zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x51,0xd3]
          vsqrtnepbf16 zmm2 {k7}, zmm3

// CHECK: vsqrtnepbf16 zmm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x51,0xd3]
          vsqrtnepbf16 zmm2 {k7} {z}, zmm3

// CHECK: vsqrtnepbf16 ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x51,0xd3]
          vsqrtnepbf16 ymm2, ymm3

// CHECK: vsqrtnepbf16 ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x51,0xd3]
          vsqrtnepbf16 ymm2 {k7}, ymm3

// CHECK: vsqrtnepbf16 ymm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x51,0xd3]
          vsqrtnepbf16 ymm2 {k7} {z}, ymm3

// CHECK: vsqrtnepbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsqrtnepbf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vsqrtnepbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vsqrtnepbf16 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vsqrtnepbf16 xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x51,0x10]
          vsqrtnepbf16 xmm2, word ptr [eax]{1to8}

// CHECK: vsqrtnepbf16 xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x51,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vsqrtnepbf16 xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vsqrtnepbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x51,0x51,0x7f]
          vsqrtnepbf16 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vsqrtnepbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x51,0x52,0x80]
          vsqrtnepbf16 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vsqrtnepbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsqrtnepbf16 ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vsqrtnepbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vsqrtnepbf16 ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vsqrtnepbf16 ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x51,0x10]
          vsqrtnepbf16 ymm2, word ptr [eax]{1to16}

// CHECK: vsqrtnepbf16 ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x51,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vsqrtnepbf16 ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vsqrtnepbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x51,0x51,0x7f]
          vsqrtnepbf16 ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vsqrtnepbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x51,0x52,0x80]
          vsqrtnepbf16 ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vsqrtnepbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsqrtnepbf16 zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vsqrtnepbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vsqrtnepbf16 zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vsqrtnepbf16 zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x51,0x10]
          vsqrtnepbf16 zmm2, word ptr [eax]{1to32}

// CHECK: vsqrtnepbf16 zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x51,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vsqrtnepbf16 zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vsqrtnepbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x51,0x51,0x7f]
          vsqrtnepbf16 zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vsqrtnepbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x51,0x52,0x80]
          vsqrtnepbf16 zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vsubnepbf16 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5c,0xd4]
          vsubnepbf16 ymm2, ymm3, ymm4

// CHECK: vsubnepbf16 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5c,0xd4]
          vsubnepbf16 ymm2 {k7}, ymm3, ymm4

// CHECK: vsubnepbf16 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5c,0xd4]
          vsubnepbf16 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vsubnepbf16 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5c,0xd4]
          vsubnepbf16 zmm2, zmm3, zmm4

// CHECK: vsubnepbf16 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5c,0xd4]
          vsubnepbf16 zmm2 {k7}, zmm3, zmm4

// CHECK: vsubnepbf16 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5c,0xd4]
          vsubnepbf16 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vsubnepbf16 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5c,0xd4]
          vsubnepbf16 xmm2, xmm3, xmm4

// CHECK: vsubnepbf16 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5c,0xd4]
          vsubnepbf16 xmm2 {k7}, xmm3, xmm4

// CHECK: vsubnepbf16 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5c,0xd4]
          vsubnepbf16 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vsubnepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsubnepbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vsubnepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5c,0x94,0x87,0x23,0x01,0x00,0x00]
          vsubnepbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vsubnepbf16 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x5c,0x10]
          vsubnepbf16 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vsubnepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vsubnepbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vsubnepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5c,0x51,0x7f]
          vsubnepbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vsubnepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x5c,0x52,0x80]
          vsubnepbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vsubnepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsubnepbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vsubnepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5c,0x94,0x87,0x23,0x01,0x00,0x00]
          vsubnepbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vsubnepbf16 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x5c,0x10]
          vsubnepbf16 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vsubnepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vsubnepbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vsubnepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5c,0x51,0x7f]
          vsubnepbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vsubnepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x5c,0x52,0x80]
          vsubnepbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vsubnepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsubnepbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vsubnepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5c,0x94,0x87,0x23,0x01,0x00,0x00]
          vsubnepbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vsubnepbf16 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x5c,0x10]
          vsubnepbf16 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vsubnepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vsubnepbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vsubnepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5c,0x51,0x7f]
          vsubnepbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vsubnepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x5c,0x52,0x80]
          vsubnepbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

