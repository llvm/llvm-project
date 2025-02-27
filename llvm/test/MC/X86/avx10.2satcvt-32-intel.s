// RUN: llvm-mc -triple i386 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vcvtbf162ibs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x69,0xd3]
          vcvtbf162ibs xmm2, xmm3

// CHECK: vcvtbf162ibs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x69,0xd3]
          vcvtbf162ibs xmm2 {k7}, xmm3

// CHECK: vcvtbf162ibs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x69,0xd3]
          vcvtbf162ibs xmm2 {k7} {z}, xmm3

// CHECK: vcvtbf162ibs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x69,0xd3]
          vcvtbf162ibs zmm2, zmm3

// CHECK: vcvtbf162ibs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x69,0xd3]
          vcvtbf162ibs zmm2 {k7}, zmm3

// CHECK: vcvtbf162ibs zmm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x69,0xd3]
          vcvtbf162ibs zmm2 {k7} {z}, zmm3

// CHECK: vcvtbf162ibs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x69,0xd3]
          vcvtbf162ibs ymm2, ymm3

// CHECK: vcvtbf162ibs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x69,0xd3]
          vcvtbf162ibs ymm2 {k7}, ymm3

// CHECK: vcvtbf162ibs ymm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x69,0xd3]
          vcvtbf162ibs ymm2 {k7} {z}, ymm3

// CHECK: vcvtbf162ibs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbf162ibs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbf162ibs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbf162ibs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbf162ibs xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x69,0x10]
          vcvtbf162ibs xmm2, word ptr [eax]{1to8}

// CHECK: vcvtbf162ibs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x69,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbf162ibs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvtbf162ibs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x69,0x51,0x7f]
          vcvtbf162ibs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvtbf162ibs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7f,0x9f,0x69,0x52,0x80]
          vcvtbf162ibs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvtbf162ibs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbf162ibs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbf162ibs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbf162ibs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbf162ibs ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7f,0x38,0x69,0x10]
          vcvtbf162ibs ymm2, word ptr [eax]{1to16}

// CHECK: vcvtbf162ibs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x69,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbf162ibs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtbf162ibs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x69,0x51,0x7f]
          vcvtbf162ibs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvtbf162ibs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7f,0xbf,0x69,0x52,0x80]
          vcvtbf162ibs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvtbf162ibs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbf162ibs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbf162ibs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbf162ibs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbf162ibs zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7f,0x58,0x69,0x10]
          vcvtbf162ibs zmm2, word ptr [eax]{1to32}

// CHECK: vcvtbf162ibs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x69,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbf162ibs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtbf162ibs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x69,0x51,0x7f]
          vcvtbf162ibs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvtbf162ibs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7f,0xdf,0x69,0x52,0x80]
          vcvtbf162ibs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvtbf162iubs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6b,0xd3]
          vcvtbf162iubs xmm2, xmm3

// CHECK: vcvtbf162iubs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x6b,0xd3]
          vcvtbf162iubs xmm2 {k7}, xmm3

// CHECK: vcvtbf162iubs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x6b,0xd3]
          vcvtbf162iubs xmm2 {k7} {z}, xmm3

// CHECK: vcvtbf162iubs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x6b,0xd3]
          vcvtbf162iubs zmm2, zmm3

// CHECK: vcvtbf162iubs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x6b,0xd3]
          vcvtbf162iubs zmm2 {k7}, zmm3

// CHECK: vcvtbf162iubs zmm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x6b,0xd3]
          vcvtbf162iubs zmm2 {k7} {z}, zmm3

// CHECK: vcvtbf162iubs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x6b,0xd3]
          vcvtbf162iubs ymm2, ymm3

// CHECK: vcvtbf162iubs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x6b,0xd3]
          vcvtbf162iubs ymm2 {k7}, ymm3

// CHECK: vcvtbf162iubs ymm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x6b,0xd3]
          vcvtbf162iubs ymm2 {k7} {z}, ymm3

// CHECK: vcvtbf162iubs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbf162iubs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbf162iubs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbf162iubs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbf162iubs xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x6b,0x10]
          vcvtbf162iubs xmm2, word ptr [eax]{1to8}

// CHECK: vcvtbf162iubs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6b,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbf162iubs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvtbf162iubs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x6b,0x51,0x7f]
          vcvtbf162iubs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvtbf162iubs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7f,0x9f,0x6b,0x52,0x80]
          vcvtbf162iubs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvtbf162iubs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbf162iubs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbf162iubs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbf162iubs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbf162iubs ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7f,0x38,0x6b,0x10]
          vcvtbf162iubs ymm2, word ptr [eax]{1to16}

// CHECK: vcvtbf162iubs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x6b,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbf162iubs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtbf162iubs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x6b,0x51,0x7f]
          vcvtbf162iubs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvtbf162iubs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7f,0xbf,0x6b,0x52,0x80]
          vcvtbf162iubs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvtbf162iubs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbf162iubs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbf162iubs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbf162iubs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbf162iubs zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7f,0x58,0x6b,0x10]
          vcvtbf162iubs zmm2, word ptr [eax]{1to32}

// CHECK: vcvtbf162iubs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x6b,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbf162iubs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtbf162iubs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x6b,0x51,0x7f]
          vcvtbf162iubs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvtbf162iubs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7f,0xdf,0x6b,0x52,0x80]
          vcvtbf162iubs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvtph2ibs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x69,0xd3]
          vcvtph2ibs xmm2, xmm3

// CHECK: vcvtph2ibs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x69,0xd3]
          vcvtph2ibs xmm2 {k7}, xmm3

// CHECK: vcvtph2ibs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x69,0xd3]
          vcvtph2ibs xmm2 {k7} {z}, xmm3

// CHECK: vcvtph2ibs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x69,0xd3]
          vcvtph2ibs zmm2, zmm3

// CHECK: vcvtph2ibs zmm2, zmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x69,0xd3]
          vcvtph2ibs zmm2, zmm3, {rn-sae}

// CHECK: vcvtph2ibs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x69,0xd3]
          vcvtph2ibs zmm2 {k7}, zmm3

// CHECK: vcvtph2ibs zmm2 {k7} {z}, zmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0xff,0x69,0xd3]
          vcvtph2ibs zmm2 {k7} {z}, zmm3, {rz-sae}

// CHECK: vcvtph2ibs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x69,0xd3]
          vcvtph2ibs ymm2, ymm3

// CHECK: vcvtph2ibs ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x69,0xd3]
          vcvtph2ibs ymm2, ymm3, {rn-sae}

// CHECK: vcvtph2ibs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x69,0xd3]
          vcvtph2ibs ymm2 {k7}, ymm3

// CHECK: vcvtph2ibs ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0xff,0x69,0xd3]
          vcvtph2ibs ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtph2ibs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2ibs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2ibs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2ibs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2ibs xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x69,0x10]
          vcvtph2ibs xmm2, word ptr [eax]{1to8}

// CHECK: vcvtph2ibs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x69,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2ibs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvtph2ibs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x69,0x51,0x7f]
          vcvtph2ibs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvtph2ibs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x69,0x52,0x80]
          vcvtph2ibs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvtph2ibs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2ibs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2ibs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2ibs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2ibs ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x69,0x10]
          vcvtph2ibs ymm2, word ptr [eax]{1to16}

// CHECK: vcvtph2ibs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x69,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2ibs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtph2ibs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7c,0xaf,0x69,0x51,0x7f]
          vcvtph2ibs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvtph2ibs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x69,0x52,0x80]
          vcvtph2ibs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvtph2ibs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2ibs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2ibs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2ibs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2ibs zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x69,0x10]
          vcvtph2ibs zmm2, word ptr [eax]{1to32}

// CHECK: vcvtph2ibs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x69,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2ibs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtph2ibs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7c,0xcf,0x69,0x51,0x7f]
          vcvtph2ibs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvtph2ibs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x69,0x52,0x80]
          vcvtph2ibs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvtph2iubs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6b,0xd3]
          vcvtph2iubs xmm2, xmm3

// CHECK: vcvtph2iubs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6b,0xd3]
          vcvtph2iubs xmm2 {k7}, xmm3

// CHECK: vcvtph2iubs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6b,0xd3]
          vcvtph2iubs xmm2 {k7} {z}, xmm3

// CHECK: vcvtph2iubs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6b,0xd3]
          vcvtph2iubs zmm2, zmm3

// CHECK: vcvtph2iubs zmm2, zmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6b,0xd3]
          vcvtph2iubs zmm2, zmm3, {rn-sae}

// CHECK: vcvtph2iubs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6b,0xd3]
          vcvtph2iubs zmm2 {k7}, zmm3

// CHECK: vcvtph2iubs zmm2 {k7} {z}, zmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0xff,0x6b,0xd3]
          vcvtph2iubs zmm2 {k7} {z}, zmm3, {rz-sae}

// CHECK: vcvtph2iubs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6b,0xd3]
          vcvtph2iubs ymm2, ymm3

// CHECK: vcvtph2iubs ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x6b,0xd3]
          vcvtph2iubs ymm2, ymm3, {rn-sae}

// CHECK: vcvtph2iubs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6b,0xd3]
          vcvtph2iubs ymm2 {k7}, ymm3

// CHECK: vcvtph2iubs ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0xff,0x6b,0xd3]
          vcvtph2iubs ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtph2iubs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2iubs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2iubs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2iubs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2iubs xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6b,0x10]
          vcvtph2iubs xmm2, word ptr [eax]{1to8}

// CHECK: vcvtph2iubs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6b,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2iubs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvtph2iubs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6b,0x51,0x7f]
          vcvtph2iubs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvtph2iubs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6b,0x52,0x80]
          vcvtph2iubs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvtph2iubs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2iubs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2iubs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2iubs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2iubs ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x6b,0x10]
          vcvtph2iubs ymm2, word ptr [eax]{1to16}

// CHECK: vcvtph2iubs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6b,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2iubs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtph2iubs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7c,0xaf,0x6b,0x51,0x7f]
          vcvtph2iubs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvtph2iubs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x6b,0x52,0x80]
          vcvtph2iubs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvtph2iubs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2iubs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2iubs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2iubs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2iubs zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x6b,0x10]
          vcvtph2iubs zmm2, word ptr [eax]{1to32}

// CHECK: vcvtph2iubs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6b,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2iubs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtph2iubs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7c,0xcf,0x6b,0x51,0x7f]
          vcvtph2iubs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvtph2iubs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x6b,0x52,0x80]
          vcvtph2iubs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvtps2ibs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x69,0xd3]
          vcvtps2ibs xmm2, xmm3

// CHECK: vcvtps2ibs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x69,0xd3]
          vcvtps2ibs xmm2 {k7}, xmm3

// CHECK: vcvtps2ibs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x69,0xd3]
          vcvtps2ibs xmm2 {k7} {z}, xmm3

// CHECK: vcvtps2ibs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x69,0xd3]
          vcvtps2ibs zmm2, zmm3

// CHECK: vcvtps2ibs zmm2, zmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x69,0xd3]
          vcvtps2ibs zmm2, zmm3, {rn-sae}

// CHECK: vcvtps2ibs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x69,0xd3]
          vcvtps2ibs zmm2 {k7}, zmm3

// CHECK: vcvtps2ibs zmm2 {k7} {z}, zmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0xff,0x69,0xd3]
          vcvtps2ibs zmm2 {k7} {z}, zmm3, {rz-sae}

// CHECK: vcvtps2ibs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x69,0xd3]
          vcvtps2ibs ymm2, ymm3

// CHECK: vcvtps2ibs ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x69,0xd3]
          vcvtps2ibs ymm2, ymm3, {rn-sae}

// CHECK: vcvtps2ibs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x69,0xd3]
          vcvtps2ibs ymm2 {k7}, ymm3

// CHECK: vcvtps2ibs ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x69,0xd3]
          vcvtps2ibs ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtps2ibs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2ibs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtps2ibs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtps2ibs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtps2ibs xmm2, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x69,0x10]
          vcvtps2ibs xmm2, dword ptr [eax]{1to4}

// CHECK: vcvtps2ibs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x69,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtps2ibs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvtps2ibs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x69,0x51,0x7f]
          vcvtps2ibs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvtps2ibs xmm2 {k7} {z}, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x69,0x52,0x80]
          vcvtps2ibs xmm2 {k7} {z}, dword ptr [edx - 512]{1to4}

// CHECK: vcvtps2ibs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2ibs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtps2ibs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtps2ibs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvtps2ibs ymm2, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x69,0x10]
          vcvtps2ibs ymm2, dword ptr [eax]{1to8}

// CHECK: vcvtps2ibs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x69,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtps2ibs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtps2ibs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x69,0x51,0x7f]
          vcvtps2ibs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvtps2ibs ymm2 {k7} {z}, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x69,0x52,0x80]
          vcvtps2ibs ymm2 {k7} {z}, dword ptr [edx - 512]{1to8}

// CHECK: vcvtps2ibs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2ibs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtps2ibs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtps2ibs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtps2ibs zmm2, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x69,0x10]
          vcvtps2ibs zmm2, dword ptr [eax]{1to16}

// CHECK: vcvtps2ibs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x69,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtps2ibs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtps2ibs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x69,0x51,0x7f]
          vcvtps2ibs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvtps2ibs zmm2 {k7} {z}, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x69,0x52,0x80]
          vcvtps2ibs zmm2 {k7} {z}, dword ptr [edx - 512]{1to16}

// CHECK: vcvtps2iubs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6b,0xd3]
          vcvtps2iubs xmm2, xmm3

// CHECK: vcvtps2iubs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6b,0xd3]
          vcvtps2iubs xmm2 {k7}, xmm3

// CHECK: vcvtps2iubs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6b,0xd3]
          vcvtps2iubs xmm2 {k7} {z}, xmm3

// CHECK: vcvtps2iubs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6b,0xd3]
          vcvtps2iubs zmm2, zmm3

// CHECK: vcvtps2iubs zmm2, zmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6b,0xd3]
          vcvtps2iubs zmm2, zmm3, {rn-sae}

// CHECK: vcvtps2iubs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6b,0xd3]
          vcvtps2iubs zmm2 {k7}, zmm3

// CHECK: vcvtps2iubs zmm2 {k7} {z}, zmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0xff,0x6b,0xd3]
          vcvtps2iubs zmm2 {k7} {z}, zmm3, {rz-sae}

// CHECK: vcvtps2iubs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6b,0xd3]
          vcvtps2iubs ymm2, ymm3

// CHECK: vcvtps2iubs ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x6b,0xd3]
          vcvtps2iubs ymm2, ymm3, {rn-sae}

// CHECK: vcvtps2iubs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6b,0xd3]
          vcvtps2iubs ymm2 {k7}, ymm3

// CHECK: vcvtps2iubs ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x6b,0xd3]
          vcvtps2iubs ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtps2iubs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2iubs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtps2iubs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtps2iubs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtps2iubs xmm2, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6b,0x10]
          vcvtps2iubs xmm2, dword ptr [eax]{1to4}

// CHECK: vcvtps2iubs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6b,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtps2iubs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvtps2iubs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6b,0x51,0x7f]
          vcvtps2iubs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvtps2iubs xmm2 {k7} {z}, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6b,0x52,0x80]
          vcvtps2iubs xmm2 {k7} {z}, dword ptr [edx - 512]{1to4}

// CHECK: vcvtps2iubs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2iubs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtps2iubs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtps2iubs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvtps2iubs ymm2, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x6b,0x10]
          vcvtps2iubs ymm2, dword ptr [eax]{1to8}

// CHECK: vcvtps2iubs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6b,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtps2iubs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtps2iubs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x6b,0x51,0x7f]
          vcvtps2iubs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvtps2iubs ymm2 {k7} {z}, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x6b,0x52,0x80]
          vcvtps2iubs ymm2 {k7} {z}, dword ptr [edx - 512]{1to8}

// CHECK: vcvtps2iubs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2iubs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtps2iubs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtps2iubs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtps2iubs zmm2, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x6b,0x10]
          vcvtps2iubs zmm2, dword ptr [eax]{1to16}

// CHECK: vcvtps2iubs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6b,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtps2iubs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtps2iubs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x6b,0x51,0x7f]
          vcvtps2iubs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvtps2iubs zmm2 {k7} {z}, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x6b,0x52,0x80]
          vcvtps2iubs zmm2 {k7} {z}, dword ptr [edx - 512]{1to16}

// CHECK: vcvttbf162ibs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x68,0xd3]
          vcvttbf162ibs xmm2, xmm3

// CHECK: vcvttbf162ibs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x68,0xd3]
          vcvttbf162ibs xmm2 {k7}, xmm3

// CHECK: vcvttbf162ibs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x68,0xd3]
          vcvttbf162ibs xmm2 {k7} {z}, xmm3

// CHECK: vcvttbf162ibs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x68,0xd3]
          vcvttbf162ibs zmm2, zmm3

// CHECK: vcvttbf162ibs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x68,0xd3]
          vcvttbf162ibs zmm2 {k7}, zmm3

// CHECK: vcvttbf162ibs zmm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x68,0xd3]
          vcvttbf162ibs zmm2 {k7} {z}, zmm3

// CHECK: vcvttbf162ibs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x68,0xd3]
          vcvttbf162ibs ymm2, ymm3

// CHECK: vcvttbf162ibs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x68,0xd3]
          vcvttbf162ibs ymm2 {k7}, ymm3

// CHECK: vcvttbf162ibs ymm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x68,0xd3]
          vcvttbf162ibs ymm2 {k7} {z}, ymm3

// CHECK: vcvttbf162ibs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttbf162ibs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttbf162ibs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttbf162ibs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttbf162ibs xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x68,0x10]
          vcvttbf162ibs xmm2, word ptr [eax]{1to8}

// CHECK: vcvttbf162ibs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x68,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttbf162ibs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttbf162ibs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x68,0x51,0x7f]
          vcvttbf162ibs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttbf162ibs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7f,0x9f,0x68,0x52,0x80]
          vcvttbf162ibs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvttbf162ibs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttbf162ibs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttbf162ibs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttbf162ibs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvttbf162ibs ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7f,0x38,0x68,0x10]
          vcvttbf162ibs ymm2, word ptr [eax]{1to16}

// CHECK: vcvttbf162ibs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x68,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttbf162ibs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttbf162ibs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x68,0x51,0x7f]
          vcvttbf162ibs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttbf162ibs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7f,0xbf,0x68,0x52,0x80]
          vcvttbf162ibs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvttbf162ibs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttbf162ibs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttbf162ibs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttbf162ibs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttbf162ibs zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7f,0x58,0x68,0x10]
          vcvttbf162ibs zmm2, word ptr [eax]{1to32}

// CHECK: vcvttbf162ibs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x68,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttbf162ibs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvttbf162ibs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x68,0x51,0x7f]
          vcvttbf162ibs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvttbf162ibs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7f,0xdf,0x68,0x52,0x80]
          vcvttbf162ibs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvttbf162iubs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6a,0xd3]
          vcvttbf162iubs xmm2, xmm3

// CHECK: vcvttbf162iubs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x6a,0xd3]
          vcvttbf162iubs xmm2 {k7}, xmm3

// CHECK: vcvttbf162iubs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x6a,0xd3]
          vcvttbf162iubs xmm2 {k7} {z}, xmm3

// CHECK: vcvttbf162iubs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x6a,0xd3]
          vcvttbf162iubs zmm2, zmm3

// CHECK: vcvttbf162iubs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x6a,0xd3]
          vcvttbf162iubs zmm2 {k7}, zmm3

// CHECK: vcvttbf162iubs zmm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x6a,0xd3]
          vcvttbf162iubs zmm2 {k7} {z}, zmm3

// CHECK: vcvttbf162iubs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x6a,0xd3]
          vcvttbf162iubs ymm2, ymm3

// CHECK: vcvttbf162iubs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x6a,0xd3]
          vcvttbf162iubs ymm2 {k7}, ymm3

// CHECK: vcvttbf162iubs ymm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x6a,0xd3]
          vcvttbf162iubs ymm2 {k7} {z}, ymm3

// CHECK: vcvttbf162iubs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttbf162iubs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttbf162iubs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttbf162iubs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttbf162iubs xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x6a,0x10]
          vcvttbf162iubs xmm2, word ptr [eax]{1to8}

// CHECK: vcvttbf162iubs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6a,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttbf162iubs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttbf162iubs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x6a,0x51,0x7f]
          vcvttbf162iubs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttbf162iubs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7f,0x9f,0x6a,0x52,0x80]
          vcvttbf162iubs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvttbf162iubs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttbf162iubs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttbf162iubs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttbf162iubs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvttbf162iubs ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7f,0x38,0x6a,0x10]
          vcvttbf162iubs ymm2, word ptr [eax]{1to16}

// CHECK: vcvttbf162iubs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x6a,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttbf162iubs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttbf162iubs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x6a,0x51,0x7f]
          vcvttbf162iubs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttbf162iubs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7f,0xbf,0x6a,0x52,0x80]
          vcvttbf162iubs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvttbf162iubs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttbf162iubs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttbf162iubs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttbf162iubs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttbf162iubs zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7f,0x58,0x6a,0x10]
          vcvttbf162iubs zmm2, word ptr [eax]{1to32}

// CHECK: vcvttbf162iubs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x6a,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttbf162iubs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvttbf162iubs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x6a,0x51,0x7f]
          vcvttbf162iubs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvttbf162iubs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7f,0xdf,0x6a,0x52,0x80]
          vcvttbf162iubs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvttph2ibs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x68,0xd3]
          vcvttph2ibs xmm2, xmm3

// CHECK: vcvttph2ibs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x68,0xd3]
          vcvttph2ibs xmm2 {k7}, xmm3

// CHECK: vcvttph2ibs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x68,0xd3]
          vcvttph2ibs xmm2 {k7} {z}, xmm3

// CHECK: vcvttph2ibs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x68,0xd3]
          vcvttph2ibs zmm2, zmm3

// CHECK: vcvttph2ibs zmm2, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x68,0xd3]
          vcvttph2ibs zmm2, zmm3, {sae}

// CHECK: vcvttph2ibs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x68,0xd3]
          vcvttph2ibs zmm2 {k7}, zmm3

// CHECK: vcvttph2ibs zmm2 {k7} {z}, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x68,0xd3]
          vcvttph2ibs zmm2 {k7} {z}, zmm3, {sae}

// CHECK: vcvttph2ibs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x68,0xd3]
          vcvttph2ibs ymm2, ymm3

// CHECK: vcvttph2ibs ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x68,0xd3]
          vcvttph2ibs ymm2, ymm3, {sae}

// CHECK: vcvttph2ibs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x68,0xd3]
          vcvttph2ibs ymm2 {k7}, ymm3

// CHECK: vcvttph2ibs ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x68,0xd3]
          vcvttph2ibs ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttph2ibs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2ibs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttph2ibs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttph2ibs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttph2ibs xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x68,0x10]
          vcvttph2ibs xmm2, word ptr [eax]{1to8}

// CHECK: vcvttph2ibs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x68,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttph2ibs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttph2ibs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x68,0x51,0x7f]
          vcvttph2ibs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttph2ibs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x68,0x52,0x80]
          vcvttph2ibs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvttph2ibs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2ibs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttph2ibs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttph2ibs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvttph2ibs ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x68,0x10]
          vcvttph2ibs ymm2, word ptr [eax]{1to16}

// CHECK: vcvttph2ibs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x68,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttph2ibs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttph2ibs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7c,0xaf,0x68,0x51,0x7f]
          vcvttph2ibs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttph2ibs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x68,0x52,0x80]
          vcvttph2ibs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvttph2ibs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2ibs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttph2ibs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttph2ibs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttph2ibs zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x68,0x10]
          vcvttph2ibs zmm2, word ptr [eax]{1to32}

// CHECK: vcvttph2ibs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x68,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttph2ibs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvttph2ibs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7c,0xcf,0x68,0x51,0x7f]
          vcvttph2ibs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvttph2ibs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x68,0x52,0x80]
          vcvttph2ibs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvttph2iubs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6a,0xd3]
          vcvttph2iubs xmm2, xmm3

// CHECK: vcvttph2iubs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6a,0xd3]
          vcvttph2iubs xmm2 {k7}, xmm3

// CHECK: vcvttph2iubs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6a,0xd3]
          vcvttph2iubs xmm2 {k7} {z}, xmm3

// CHECK: vcvttph2iubs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6a,0xd3]
          vcvttph2iubs zmm2, zmm3

// CHECK: vcvttph2iubs zmm2, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6a,0xd3]
          vcvttph2iubs zmm2, zmm3, {sae}

// CHECK: vcvttph2iubs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6a,0xd3]
          vcvttph2iubs zmm2 {k7}, zmm3

// CHECK: vcvttph2iubs zmm2 {k7} {z}, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6a,0xd3]
          vcvttph2iubs zmm2 {k7} {z}, zmm3, {sae}

// CHECK: vcvttph2iubs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6a,0xd3]
          vcvttph2iubs ymm2, ymm3

// CHECK: vcvttph2iubs ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x6a,0xd3]
          vcvttph2iubs ymm2, ymm3, {sae}

// CHECK: vcvttph2iubs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6a,0xd3]
          vcvttph2iubs ymm2 {k7}, ymm3

// CHECK: vcvttph2iubs ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x6a,0xd3]
          vcvttph2iubs ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttph2iubs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2iubs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttph2iubs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttph2iubs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttph2iubs xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6a,0x10]
          vcvttph2iubs xmm2, word ptr [eax]{1to8}

// CHECK: vcvttph2iubs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6a,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttph2iubs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttph2iubs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6a,0x51,0x7f]
          vcvttph2iubs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttph2iubs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6a,0x52,0x80]
          vcvttph2iubs xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvttph2iubs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2iubs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttph2iubs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttph2iubs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvttph2iubs ymm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x6a,0x10]
          vcvttph2iubs ymm2, word ptr [eax]{1to16}

// CHECK: vcvttph2iubs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6a,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttph2iubs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttph2iubs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7c,0xaf,0x6a,0x51,0x7f]
          vcvttph2iubs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttph2iubs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x6a,0x52,0x80]
          vcvttph2iubs ymm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvttph2iubs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2iubs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttph2iubs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttph2iubs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttph2iubs zmm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x6a,0x10]
          vcvttph2iubs zmm2, word ptr [eax]{1to32}

// CHECK: vcvttph2iubs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6a,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttph2iubs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvttph2iubs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7c,0xcf,0x6a,0x51,0x7f]
          vcvttph2iubs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvttph2iubs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x6a,0x52,0x80]
          vcvttph2iubs zmm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvttps2ibs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x68,0xd3]
          vcvttps2ibs xmm2, xmm3

// CHECK: vcvttps2ibs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x68,0xd3]
          vcvttps2ibs xmm2 {k7}, xmm3

// CHECK: vcvttps2ibs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x68,0xd3]
          vcvttps2ibs xmm2 {k7} {z}, xmm3

// CHECK: vcvttps2ibs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x68,0xd3]
          vcvttps2ibs zmm2, zmm3

// CHECK: vcvttps2ibs zmm2, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x68,0xd3]
          vcvttps2ibs zmm2, zmm3, {sae}

// CHECK: vcvttps2ibs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x68,0xd3]
          vcvttps2ibs zmm2 {k7}, zmm3

// CHECK: vcvttps2ibs zmm2 {k7} {z}, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x68,0xd3]
          vcvttps2ibs zmm2 {k7} {z}, zmm3, {sae}

// CHECK: vcvttps2ibs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x68,0xd3]
          vcvttps2ibs ymm2, ymm3

// CHECK: vcvttps2ibs ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x68,0xd3]
          vcvttps2ibs ymm2, ymm3, {sae}

// CHECK: vcvttps2ibs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x68,0xd3]
          vcvttps2ibs ymm2 {k7}, ymm3

// CHECK: vcvttps2ibs ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x68,0xd3]
          vcvttps2ibs ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttps2ibs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2ibs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2ibs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2ibs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2ibs xmm2, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x68,0x10]
          vcvttps2ibs xmm2, dword ptr [eax]{1to4}

// CHECK: vcvttps2ibs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x68,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2ibs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttps2ibs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x68,0x51,0x7f]
          vcvttps2ibs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttps2ibs xmm2 {k7} {z}, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x68,0x52,0x80]
          vcvttps2ibs xmm2 {k7} {z}, dword ptr [edx - 512]{1to4}

// CHECK: vcvttps2ibs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2ibs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2ibs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2ibs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2ibs ymm2, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x68,0x10]
          vcvttps2ibs ymm2, dword ptr [eax]{1to8}

// CHECK: vcvttps2ibs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x68,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2ibs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttps2ibs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x68,0x51,0x7f]
          vcvttps2ibs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttps2ibs ymm2 {k7} {z}, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x68,0x52,0x80]
          vcvttps2ibs ymm2 {k7} {z}, dword ptr [edx - 512]{1to8}

// CHECK: vcvttps2ibs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2ibs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2ibs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2ibs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2ibs zmm2, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x68,0x10]
          vcvttps2ibs zmm2, dword ptr [eax]{1to16}

// CHECK: vcvttps2ibs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x68,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttps2ibs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvttps2ibs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x68,0x51,0x7f]
          vcvttps2ibs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvttps2ibs zmm2 {k7} {z}, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x68,0x52,0x80]
          vcvttps2ibs zmm2 {k7} {z}, dword ptr [edx - 512]{1to16}

// CHECK: vcvttps2iubs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6a,0xd3]
          vcvttps2iubs xmm2, xmm3

// CHECK: vcvttps2iubs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6a,0xd3]
          vcvttps2iubs xmm2 {k7}, xmm3

// CHECK: vcvttps2iubs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6a,0xd3]
          vcvttps2iubs xmm2 {k7} {z}, xmm3

// CHECK: vcvttps2iubs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6a,0xd3]
          vcvttps2iubs zmm2, zmm3

// CHECK: vcvttps2iubs zmm2, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6a,0xd3]
          vcvttps2iubs zmm2, zmm3, {sae}

// CHECK: vcvttps2iubs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6a,0xd3]
          vcvttps2iubs zmm2 {k7}, zmm3

// CHECK: vcvttps2iubs zmm2 {k7} {z}, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6a,0xd3]
          vcvttps2iubs zmm2 {k7} {z}, zmm3, {sae}

// CHECK: vcvttps2iubs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6a,0xd3]
          vcvttps2iubs ymm2, ymm3

// CHECK: vcvttps2iubs ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x6a,0xd3]
          vcvttps2iubs ymm2, ymm3, {sae}

// CHECK: vcvttps2iubs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6a,0xd3]
          vcvttps2iubs ymm2 {k7}, ymm3

// CHECK: vcvttps2iubs ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x6a,0xd3]
          vcvttps2iubs ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttps2iubs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2iubs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2iubs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2iubs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2iubs xmm2, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6a,0x10]
          vcvttps2iubs xmm2, dword ptr [eax]{1to4}

// CHECK: vcvttps2iubs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6a,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2iubs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttps2iubs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6a,0x51,0x7f]
          vcvttps2iubs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttps2iubs xmm2 {k7} {z}, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6a,0x52,0x80]
          vcvttps2iubs xmm2 {k7} {z}, dword ptr [edx - 512]{1to4}

// CHECK: vcvttps2iubs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2iubs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2iubs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2iubs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2iubs ymm2, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x6a,0x10]
          vcvttps2iubs ymm2, dword ptr [eax]{1to8}

// CHECK: vcvttps2iubs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6a,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2iubs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttps2iubs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x6a,0x51,0x7f]
          vcvttps2iubs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttps2iubs ymm2 {k7} {z}, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x6a,0x52,0x80]
          vcvttps2iubs ymm2 {k7} {z}, dword ptr [edx - 512]{1to8}

// CHECK: vcvttps2iubs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2iubs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2iubs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2iubs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2iubs zmm2, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x6a,0x10]
          vcvttps2iubs zmm2, dword ptr [eax]{1to16}

// CHECK: vcvttps2iubs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6a,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttps2iubs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvttps2iubs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x6a,0x51,0x7f]
          vcvttps2iubs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvttps2iubs zmm2 {k7} {z}, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x6a,0x52,0x80]
          vcvttps2iubs zmm2 {k7} {z}, dword ptr [edx - 512]{1to16}

