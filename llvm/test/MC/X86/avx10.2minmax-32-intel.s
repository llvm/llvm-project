// RUN: llvm-mc -triple i386 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vminmaxbf16 xmm2, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0x52,0xd4,0x7b]
          vminmaxbf16 xmm2, xmm3, xmm4, 123

// CHECK: vminmaxbf16 xmm2 {k7}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x0f,0x52,0xd4,0x7b]
          vminmaxbf16 xmm2 {k7}, xmm3, xmm4, 123

// CHECK: vminmaxbf16 xmm2 {k7} {z}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x8f,0x52,0xd4,0x7b]
          vminmaxbf16 xmm2 {k7} {z}, xmm3, xmm4, 123

// CHECK: vminmaxbf16 zmm2, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0x52,0xd4,0x7b]
          vminmaxbf16 zmm2, zmm3, zmm4, 123

// CHECK: vminmaxbf16 zmm2 {k7}, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x4f,0x52,0xd4,0x7b]
          vminmaxbf16 zmm2 {k7}, zmm3, zmm4, 123

// CHECK: vminmaxbf16 zmm2 {k7} {z}, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0xcf,0x52,0xd4,0x7b]
          vminmaxbf16 zmm2 {k7} {z}, zmm3, zmm4, 123

// CHECK: vminmaxbf16 ymm2, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0x52,0xd4,0x7b]
          vminmaxbf16 ymm2, ymm3, ymm4, 123

// CHECK: vminmaxbf16 ymm2 {k7}, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x2f,0x52,0xd4,0x7b]
          vminmaxbf16 ymm2 {k7}, ymm3, ymm4, 123

// CHECK: vminmaxbf16 ymm2 {k7} {z}, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x67,0xaf,0x52,0xd4,0x7b]
          vminmaxbf16 ymm2 {k7} {z}, ymm3, ymm4, 123

// CHECK: vminmaxbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxbf16 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x2f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxbf16 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxbf16 ymm2, ymm3, word ptr [eax]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x38,0x52,0x10,0x7b]
          vminmaxbf16 ymm2, ymm3, word ptr [eax]{1to16}, 123

// CHECK: vminmaxbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0x52,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxbf16 ymm2, ymm3, ymmword ptr [2*ebp - 1024], 123

// CHECK: vminmaxbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x67,0xaf,0x52,0x51,0x7f,0x7b]
          vminmaxbf16 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064], 123

// CHECK: vminmaxbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0xbf,0x52,0x52,0x80,0x7b]
          vminmaxbf16 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}, 123

// CHECK: vminmaxbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxbf16 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x0f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxbf16 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxbf16 xmm2, xmm3, word ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x18,0x52,0x10,0x7b]
          vminmaxbf16 xmm2, xmm3, word ptr [eax]{1to8}, 123

// CHECK: vminmaxbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0x52,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxbf16 xmm2, xmm3, xmmword ptr [2*ebp - 512], 123

// CHECK: vminmaxbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x8f,0x52,0x51,0x7f,0x7b]
          vminmaxbf16 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032], 123

// CHECK: vminmaxbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x9f,0x52,0x52,0x80,0x7b]
          vminmaxbf16 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}, 123

// CHECK: vminmaxbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxbf16 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x4f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxbf16 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxbf16 zmm2, zmm3, word ptr [eax]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0x58,0x52,0x10,0x7b]
          vminmaxbf16 zmm2, zmm3, word ptr [eax]{1to32}, 123

// CHECK: vminmaxbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0x52,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxbf16 zmm2, zmm3, zmmword ptr [2*ebp - 2048], 123

// CHECK: vminmaxbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x67,0xcf,0x52,0x51,0x7f,0x7b]
          vminmaxbf16 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128], 123

// CHECK: vminmaxbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x67,0xdf,0x52,0x52,0x80,0x7b]
          vminmaxbf16 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}, 123

// CHECK: vminmaxpd xmm2, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x52,0xd4,0x7b]
          vminmaxpd xmm2, xmm3, xmm4, 123

// CHECK: vminmaxpd xmm2 {k7}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x0f,0x52,0xd4,0x7b]
          vminmaxpd xmm2 {k7}, xmm3, xmm4, 123

// CHECK: vminmaxpd xmm2 {k7} {z}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x8f,0x52,0xd4,0x7b]
          vminmaxpd xmm2 {k7} {z}, xmm3, xmm4, 123

// CHECK: vminmaxpd zmm2, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x48,0x52,0xd4,0x7b]
          vminmaxpd zmm2, zmm3, zmm4, 123

// CHECK: vminmaxpd zmm2, zmm3, zmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x18,0x52,0xd4,0x7b]
          vminmaxpd zmm2, zmm3, zmm4, {sae}, 123

// CHECK: vminmaxpd zmm2 {k7}, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x4f,0x52,0xd4,0x7b]
          vminmaxpd zmm2 {k7}, zmm3, zmm4, 123

// CHECK: vminmaxpd zmm2 {k7} {z}, zmm3, zmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x9f,0x52,0xd4,0x7b]
          vminmaxpd zmm2 {k7} {z}, zmm3, zmm4, {sae}, 123

// CHECK: vminmaxpd ymm2, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x28,0x52,0xd4,0x7b]
          vminmaxpd ymm2, ymm3, ymm4, 123

// CHECK: vminmaxpd ymm2, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x18,0x52,0xd4,0x7b]
          vminmaxpd ymm2, ymm3, ymm4, {sae}, 123

// CHECK: vminmaxpd ymm2 {k7}, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x2f,0x52,0xd4,0x7b]
          vminmaxpd ymm2 {k7}, ymm3, ymm4, 123

// CHECK: vminmaxpd ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x9f,0x52,0xd4,0x7b]
          vminmaxpd ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123

// CHECK: vminmaxpd ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x28,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxpd ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxpd ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x2f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxpd ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxpd ymm2, ymm3, qword ptr [eax]{1to4}, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x38,0x52,0x10,0x7b]
          vminmaxpd ymm2, ymm3, qword ptr [eax]{1to4}, 123

// CHECK: vminmaxpd ymm2, ymm3, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x28,0x52,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxpd ymm2, ymm3, ymmword ptr [2*ebp - 1024], 123

// CHECK: vminmaxpd ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0xaf,0x52,0x51,0x7f,0x7b]
          vminmaxpd ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064], 123

// CHECK: vminmaxpd ymm2 {k7} {z}, ymm3, qword ptr [edx - 1024]{1to4}, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0xbf,0x52,0x52,0x80,0x7b]
          vminmaxpd ymm2 {k7} {z}, ymm3, qword ptr [edx - 1024]{1to4}, 123

// CHECK: vminmaxpd xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxpd xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxpd xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x0f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxpd xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxpd xmm2, xmm3, qword ptr [eax]{1to2}, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x18,0x52,0x10,0x7b]
          vminmaxpd xmm2, xmm3, qword ptr [eax]{1to2}, 123

// CHECK: vminmaxpd xmm2, xmm3, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x52,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxpd xmm2, xmm3, xmmword ptr [2*ebp - 512], 123

// CHECK: vminmaxpd xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x8f,0x52,0x51,0x7f,0x7b]
          vminmaxpd xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032], 123

// CHECK: vminmaxpd xmm2 {k7} {z}, xmm3, qword ptr [edx - 1024]{1to2}, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x9f,0x52,0x52,0x80,0x7b]
          vminmaxpd xmm2 {k7} {z}, xmm3, qword ptr [edx - 1024]{1to2}, 123

// CHECK: vminmaxpd zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x48,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxpd zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxpd zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x4f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxpd zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxpd zmm2, zmm3, qword ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x58,0x52,0x10,0x7b]
          vminmaxpd zmm2, zmm3, qword ptr [eax]{1to8}, 123

// CHECK: vminmaxpd zmm2, zmm3, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x48,0x52,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxpd zmm2, zmm3, zmmword ptr [2*ebp - 2048], 123

// CHECK: vminmaxpd zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0xcf,0x52,0x51,0x7f,0x7b]
          vminmaxpd zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128], 123

// CHECK: vminmaxpd zmm2 {k7} {z}, zmm3, qword ptr [edx - 1024]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0xdf,0x52,0x52,0x80,0x7b]
          vminmaxpd zmm2 {k7} {z}, zmm3, qword ptr [edx - 1024]{1to8}, 123

// CHECK: vminmaxph xmm2, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x52,0xd4,0x7b]
          vminmaxph xmm2, xmm3, xmm4, 123

// CHECK: vminmaxph xmm2 {k7}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x0f,0x52,0xd4,0x7b]
          vminmaxph xmm2 {k7}, xmm3, xmm4, 123

// CHECK: vminmaxph xmm2 {k7} {z}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x8f,0x52,0xd4,0x7b]
          vminmaxph xmm2 {k7} {z}, xmm3, xmm4, 123

// CHECK: vminmaxph zmm2, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x48,0x52,0xd4,0x7b]
          vminmaxph zmm2, zmm3, zmm4, 123

// CHECK: vminmaxph zmm2, zmm3, zmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x18,0x52,0xd4,0x7b]
          vminmaxph zmm2, zmm3, zmm4, {sae}, 123

// CHECK: vminmaxph zmm2 {k7}, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x4f,0x52,0xd4,0x7b]
          vminmaxph zmm2 {k7}, zmm3, zmm4, 123

// CHECK: vminmaxph zmm2 {k7} {z}, zmm3, zmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x9f,0x52,0xd4,0x7b]
          vminmaxph zmm2 {k7} {z}, zmm3, zmm4, {sae}, 123

// CHECK: vminmaxph ymm2, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x28,0x52,0xd4,0x7b]
          vminmaxph ymm2, ymm3, ymm4, 123

// CHECK: vminmaxph ymm2, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x60,0x18,0x52,0xd4,0x7b]
          vminmaxph ymm2, ymm3, ymm4, {sae}, 123

// CHECK: vminmaxph ymm2 {k7}, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x2f,0x52,0xd4,0x7b]
          vminmaxph ymm2 {k7}, ymm3, ymm4, 123

// CHECK: vminmaxph ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x60,0x9f,0x52,0xd4,0x7b]
          vminmaxph ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123

// CHECK: vminmaxph ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x28,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxph ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxph ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x2f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxph ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxph ymm2, ymm3, word ptr [eax]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x38,0x52,0x10,0x7b]
          vminmaxph ymm2, ymm3, word ptr [eax]{1to16}, 123

// CHECK: vminmaxph ymm2, ymm3, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x28,0x52,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxph ymm2, ymm3, ymmword ptr [2*ebp - 1024], 123

// CHECK: vminmaxph ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x64,0xaf,0x52,0x51,0x7f,0x7b]
          vminmaxph ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064], 123

// CHECK: vminmaxph ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x64,0xbf,0x52,0x52,0x80,0x7b]
          vminmaxph ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}, 123

// CHECK: vminmaxph xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxph xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxph xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x0f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxph xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxph xmm2, xmm3, word ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x18,0x52,0x10,0x7b]
          vminmaxph xmm2, xmm3, word ptr [eax]{1to8}, 123

// CHECK: vminmaxph xmm2, xmm3, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x52,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxph xmm2, xmm3, xmmword ptr [2*ebp - 512], 123

// CHECK: vminmaxph xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x8f,0x52,0x51,0x7f,0x7b]
          vminmaxph xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032], 123

// CHECK: vminmaxph xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x9f,0x52,0x52,0x80,0x7b]
          vminmaxph xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}, 123

// CHECK: vminmaxph zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x48,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxph zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxph zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x4f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxph zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxph zmm2, zmm3, word ptr [eax]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x58,0x52,0x10,0x7b]
          vminmaxph zmm2, zmm3, word ptr [eax]{1to32}, 123

// CHECK: vminmaxph zmm2, zmm3, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x48,0x52,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxph zmm2, zmm3, zmmword ptr [2*ebp - 2048], 123

// CHECK: vminmaxph zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x64,0xcf,0x52,0x51,0x7f,0x7b]
          vminmaxph zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128], 123

// CHECK: vminmaxph zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x64,0xdf,0x52,0x52,0x80,0x7b]
          vminmaxph zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}, 123

// CHECK: vminmaxps xmm2, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x52,0xd4,0x7b]
          vminmaxps xmm2, xmm3, xmm4, 123

// CHECK: vminmaxps xmm2 {k7}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x0f,0x52,0xd4,0x7b]
          vminmaxps xmm2 {k7}, xmm3, xmm4, 123

// CHECK: vminmaxps xmm2 {k7} {z}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x8f,0x52,0xd4,0x7b]
          vminmaxps xmm2 {k7} {z}, xmm3, xmm4, 123

// CHECK: vminmaxps zmm2, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x48,0x52,0xd4,0x7b]
          vminmaxps zmm2, zmm3, zmm4, 123

// CHECK: vminmaxps zmm2, zmm3, zmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x18,0x52,0xd4,0x7b]
          vminmaxps zmm2, zmm3, zmm4, {sae}, 123

// CHECK: vminmaxps zmm2 {k7}, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x4f,0x52,0xd4,0x7b]
          vminmaxps zmm2 {k7}, zmm3, zmm4, 123

// CHECK: vminmaxps zmm2 {k7} {z}, zmm3, zmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x9f,0x52,0xd4,0x7b]
          vminmaxps zmm2 {k7} {z}, zmm3, zmm4, {sae}, 123

// CHECK: vminmaxps ymm2, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x28,0x52,0xd4,0x7b]
          vminmaxps ymm2, ymm3, ymm4, 123

// CHECK: vminmaxps ymm2, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x18,0x52,0xd4,0x7b]
          vminmaxps ymm2, ymm3, ymm4, {sae}, 123

// CHECK: vminmaxps ymm2 {k7}, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x2f,0x52,0xd4,0x7b]
          vminmaxps ymm2 {k7}, ymm3, ymm4, 123

// CHECK: vminmaxps ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x9f,0x52,0xd4,0x7b]
          vminmaxps ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123

// CHECK: vminmaxps ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x28,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxps ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxps ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x2f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxps ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxps ymm2, ymm3, dword ptr [eax]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x38,0x52,0x10,0x7b]
          vminmaxps ymm2, ymm3, dword ptr [eax]{1to8}, 123

// CHECK: vminmaxps ymm2, ymm3, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x28,0x52,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxps ymm2, ymm3, ymmword ptr [2*ebp - 1024], 123

// CHECK: vminmaxps ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x65,0xaf,0x52,0x51,0x7f,0x7b]
          vminmaxps ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064], 123

// CHECK: vminmaxps ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x65,0xbf,0x52,0x52,0x80,0x7b]
          vminmaxps ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}, 123

// CHECK: vminmaxps xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxps xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxps xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x0f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxps xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxps xmm2, xmm3, dword ptr [eax]{1to4}, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x18,0x52,0x10,0x7b]
          vminmaxps xmm2, xmm3, dword ptr [eax]{1to4}, 123

// CHECK: vminmaxps xmm2, xmm3, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x52,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxps xmm2, xmm3, xmmword ptr [2*ebp - 512], 123

// CHECK: vminmaxps xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x8f,0x52,0x51,0x7f,0x7b]
          vminmaxps xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032], 123

// CHECK: vminmaxps xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x9f,0x52,0x52,0x80,0x7b]
          vminmaxps xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}, 123

// CHECK: vminmaxps zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x48,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxps zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxps zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x4f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxps zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxps zmm2, zmm3, dword ptr [eax]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x58,0x52,0x10,0x7b]
          vminmaxps zmm2, zmm3, dword ptr [eax]{1to16}, 123

// CHECK: vminmaxps zmm2, zmm3, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x48,0x52,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxps zmm2, zmm3, zmmword ptr [2*ebp - 2048], 123

// CHECK: vminmaxps zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x65,0xcf,0x52,0x51,0x7f,0x7b]
          vminmaxps zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128], 123

// CHECK: vminmaxps zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x65,0xdf,0x52,0x52,0x80,0x7b]
          vminmaxps zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}, 123

// CHECK: vminmaxsd xmm2, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x53,0xd4,0x7b]
          vminmaxsd xmm2, xmm3, xmm4, 123

// CHECK: vminmaxsd xmm2, xmm3, xmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x18,0x53,0xd4,0x7b]
          vminmaxsd xmm2, xmm3, xmm4, {sae}, 123

// CHECK: vminmaxsd xmm2 {k7}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x0f,0x53,0xd4,0x7b]
          vminmaxsd xmm2 {k7}, xmm3, xmm4, 123

// CHECK: vminmaxsd xmm2 {k7} {z}, xmm3, xmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x9f,0x53,0xd4,0x7b]
          vminmaxsd xmm2 {k7} {z}, xmm3, xmm4, {sae}, 123

// CHECK: vminmaxsd xmm2, xmm3, qword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x53,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxsd xmm2, xmm3, qword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxsd xmm2 {k7}, xmm3, qword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x0f,0x53,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxsd xmm2 {k7}, xmm3, qword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxsd xmm2, xmm3, qword ptr [eax], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x53,0x10,0x7b]
          vminmaxsd xmm2, xmm3, qword ptr [eax], 123

// CHECK: vminmaxsd xmm2, xmm3, qword ptr [2*ebp - 256], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x53,0x14,0x6d,0x00,0xff,0xff,0xff,0x7b]
          vminmaxsd xmm2, xmm3, qword ptr [2*ebp - 256], 123

// CHECK: vminmaxsd xmm2 {k7} {z}, xmm3, qword ptr [ecx + 1016], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x8f,0x53,0x51,0x7f,0x7b]
          vminmaxsd xmm2 {k7} {z}, xmm3, qword ptr [ecx + 1016], 123

// CHECK: vminmaxsd xmm2 {k7} {z}, xmm3, qword ptr [edx - 1024], 123
// CHECK: encoding: [0x62,0xf3,0xe5,0x8f,0x53,0x52,0x80,0x7b]
          vminmaxsd xmm2 {k7} {z}, xmm3, qword ptr [edx - 1024], 123

// CHECK: vminmaxsh xmm2, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x53,0xd4,0x7b]
          vminmaxsh xmm2, xmm3, xmm4, 123

// CHECK: vminmaxsh xmm2, xmm3, xmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x18,0x53,0xd4,0x7b]
          vminmaxsh xmm2, xmm3, xmm4, {sae}, 123

// CHECK: vminmaxsh xmm2 {k7}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x0f,0x53,0xd4,0x7b]
          vminmaxsh xmm2 {k7}, xmm3, xmm4, 123

// CHECK: vminmaxsh xmm2 {k7} {z}, xmm3, xmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x64,0x9f,0x53,0xd4,0x7b]
          vminmaxsh xmm2 {k7} {z}, xmm3, xmm4, {sae}, 123

// CHECK: vminmaxsh xmm2, xmm3, word ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x53,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxsh xmm2, xmm3, word ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxsh xmm2 {k7}, xmm3, word ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x0f,0x53,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxsh xmm2 {k7}, xmm3, word ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxsh xmm2, xmm3, word ptr [eax], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x53,0x10,0x7b]
          vminmaxsh xmm2, xmm3, word ptr [eax], 123

// CHECK: vminmaxsh xmm2, xmm3, word ptr [2*ebp - 64], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x53,0x14,0x6d,0xc0,0xff,0xff,0xff,0x7b]
          vminmaxsh xmm2, xmm3, word ptr [2*ebp - 64], 123

// CHECK: vminmaxsh xmm2 {k7} {z}, xmm3, word ptr [ecx + 254], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x8f,0x53,0x51,0x7f,0x7b]
          vminmaxsh xmm2 {k7} {z}, xmm3, word ptr [ecx + 254], 123

// CHECK: vminmaxsh xmm2 {k7} {z}, xmm3, word ptr [edx - 256], 123
// CHECK: encoding: [0x62,0xf3,0x64,0x8f,0x53,0x52,0x80,0x7b]
          vminmaxsh xmm2 {k7} {z}, xmm3, word ptr [edx - 256], 123

// CHECK: vminmaxss xmm2, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x53,0xd4,0x7b]
          vminmaxss xmm2, xmm3, xmm4, 123

// CHECK: vminmaxss xmm2, xmm3, xmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x18,0x53,0xd4,0x7b]
          vminmaxss xmm2, xmm3, xmm4, {sae}, 123

// CHECK: vminmaxss xmm2 {k7}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x0f,0x53,0xd4,0x7b]
          vminmaxss xmm2 {k7}, xmm3, xmm4, 123

// CHECK: vminmaxss xmm2 {k7} {z}, xmm3, xmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x65,0x9f,0x53,0xd4,0x7b]
          vminmaxss xmm2 {k7} {z}, xmm3, xmm4, {sae}, 123

// CHECK: vminmaxss xmm2, xmm3, dword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x53,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxss xmm2, xmm3, dword ptr [esp + 8*esi + 268435456], 123

// CHECK: vminmaxss xmm2 {k7}, xmm3, dword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x0f,0x53,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxss xmm2 {k7}, xmm3, dword ptr [edi + 4*eax + 291], 123

// CHECK: vminmaxss xmm2, xmm3, dword ptr [eax], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x53,0x10,0x7b]
          vminmaxss xmm2, xmm3, dword ptr [eax], 123

// CHECK: vminmaxss xmm2, xmm3, dword ptr [2*ebp - 128], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x53,0x14,0x6d,0x80,0xff,0xff,0xff,0x7b]
          vminmaxss xmm2, xmm3, dword ptr [2*ebp - 128], 123

// CHECK: vminmaxss xmm2 {k7} {z}, xmm3, dword ptr [ecx + 508], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x8f,0x53,0x51,0x7f,0x7b]
          vminmaxss xmm2 {k7} {z}, xmm3, dword ptr [ecx + 508], 123

// CHECK: vminmaxss xmm2 {k7} {z}, xmm3, dword ptr [edx - 512], 123
// CHECK: encoding: [0x62,0xf3,0x65,0x8f,0x53,0x52,0x80,0x7b]
          vminmaxss xmm2 {k7} {z}, xmm3, dword ptr [edx - 512], 123

