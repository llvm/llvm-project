// RUN: llvm-mc -triple i386 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vcvt2ps2phx ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x65,0x28,0x67,0xd4]
          vcvt2ps2phx ymm2, ymm3, ymm4

// CHECK: vcvt2ps2phx ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x67,0xd4]
          vcvt2ps2phx ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vcvt2ps2phx ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x65,0x2f,0x67,0xd4]
          vcvt2ps2phx ymm2 {k7}, ymm3, ymm4

// CHECK: vcvt2ps2phx ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x67,0xd4]
          vcvt2ps2phx ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vcvt2ps2phx zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0x67,0xd4]
          vcvt2ps2phx zmm2, zmm3, zmm4

// CHECK: vcvt2ps2phx zmm2, zmm3, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x65,0x18,0x67,0xd4]
          vcvt2ps2phx zmm2, zmm3, zmm4, {rn-sae}

// CHECK: vcvt2ps2phx zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x65,0x4f,0x67,0xd4]
          vcvt2ps2phx zmm2 {k7}, zmm3, zmm4

// CHECK: vcvt2ps2phx zmm2 {k7} {z}, zmm3, zmm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x65,0xff,0x67,0xd4]
          vcvt2ps2phx zmm2 {k7} {z}, zmm3, zmm4, {rz-sae}

// CHECK: vcvt2ps2phx xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x65,0x08,0x67,0xd4]
          vcvt2ps2phx xmm2, xmm3, xmm4

// CHECK: vcvt2ps2phx xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x65,0x0f,0x67,0xd4]
          vcvt2ps2phx xmm2 {k7}, xmm3, xmm4

// CHECK: vcvt2ps2phx xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x65,0x8f,0x67,0xd4]
          vcvt2ps2phx xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vcvt2ps2phx zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0x67,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ps2phx zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ps2phx zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x65,0x4f,0x67,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ps2phx zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ps2phx zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x65,0x58,0x67,0x10]
          vcvt2ps2phx zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vcvt2ps2phx zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0x67,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ps2phx zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vcvt2ps2phx zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x65,0xcf,0x67,0x51,0x7f]
          vcvt2ps2phx zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vcvt2ps2phx zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x65,0xdf,0x67,0x52,0x80]
          vcvt2ps2phx zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

// CHECK: vcvt2ps2phx ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x65,0x28,0x67,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ps2phx ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ps2phx ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x65,0x2f,0x67,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ps2phx ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ps2phx ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x65,0x38,0x67,0x10]
          vcvt2ps2phx ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vcvt2ps2phx ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf2,0x65,0x28,0x67,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ps2phx ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vcvt2ps2phx ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x65,0xaf,0x67,0x51,0x7f]
          vcvt2ps2phx ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vcvt2ps2phx ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x65,0xbf,0x67,0x52,0x80]
          vcvt2ps2phx ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vcvt2ps2phx xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x65,0x08,0x67,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ps2phx xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ps2phx xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x65,0x0f,0x67,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ps2phx xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ps2phx xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x65,0x18,0x67,0x10]
          vcvt2ps2phx xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vcvt2ps2phx xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf2,0x65,0x08,0x67,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ps2phx xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vcvt2ps2phx xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x65,0x8f,0x67,0x51,0x7f]
          vcvt2ps2phx xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vcvt2ps2phx xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x65,0x9f,0x67,0x52,0x80]
          vcvt2ps2phx xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vcvtbiasph2bf8 ymm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x74,0xd4]
          vcvtbiasph2bf8 ymm2, zmm3, zmm4

// CHECK: vcvtbiasph2bf8 ymm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0x74,0xd4]
          vcvtbiasph2bf8 ymm2 {k7}, zmm3, zmm4

// CHECK: vcvtbiasph2bf8 ymm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0x74,0xd4]
          vcvtbiasph2bf8 ymm2 {k7} {z}, zmm3, zmm4

// CHECK: vcvtbiasph2bf8 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x08,0x74,0xd4]
          vcvtbiasph2bf8 xmm2, xmm3, xmm4

// CHECK: vcvtbiasph2bf8 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0x74,0xd4]
          vcvtbiasph2bf8 xmm2 {k7}, xmm3, xmm4

// CHECK: vcvtbiasph2bf8 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0x74,0xd4]
          vcvtbiasph2bf8 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vcvtbiasph2bf8 xmm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0x28,0x74,0xd4]
          vcvtbiasph2bf8 xmm2, ymm3, ymm4

// CHECK: vcvtbiasph2bf8 xmm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0x74,0xd4]
          vcvtbiasph2bf8 xmm2 {k7}, ymm3, ymm4

// CHECK: vcvtbiasph2bf8 xmm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0x74,0xd4]
          vcvtbiasph2bf8 xmm2 {k7} {z}, ymm3, ymm4

// CHECK: vcvtbiasph2bf8 xmm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x64,0x28,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8 xmm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbiasph2bf8 xmm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8 xmm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbiasph2bf8 xmm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x64,0x38,0x74,0x10]
          vcvtbiasph2bf8 xmm2, ymm3, word ptr [eax]{1to16}

// CHECK: vcvtbiasph2bf8 xmm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf2,0x64,0x28,0x74,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2bf8 xmm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtbiasph2bf8 xmm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0x74,0x51,0x7f]
          vcvtbiasph2bf8 xmm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vcvtbiasph2bf8 xmm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf2,0x64,0xbf,0x74,0x52,0x80]
          vcvtbiasph2bf8 xmm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vcvtbiasph2bf8 ymm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8 ymm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbiasph2bf8 ymm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8 ymm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbiasph2bf8 ymm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf2,0x64,0x58,0x74,0x10]
          vcvtbiasph2bf8 ymm2, zmm3, word ptr [eax]{1to32}

// CHECK: vcvtbiasph2bf8 ymm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x74,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2bf8 ymm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtbiasph2bf8 ymm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0x74,0x51,0x7f]
          vcvtbiasph2bf8 ymm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vcvtbiasph2bf8 ymm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf2,0x64,0xdf,0x74,0x52,0x80]
          vcvtbiasph2bf8 ymm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vcvtbiasph2bf8 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x64,0x08,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbiasph2bf8 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbiasph2bf8 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x64,0x18,0x74,0x10]
          vcvtbiasph2bf8 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vcvtbiasph2bf8 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf2,0x64,0x08,0x74,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2bf8 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vcvtbiasph2bf8 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0x74,0x51,0x7f]
          vcvtbiasph2bf8 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vcvtbiasph2bf8 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf2,0x64,0x9f,0x74,0x52,0x80]
          vcvtbiasph2bf8 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vcvtbiasph2bf8s ymm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x74,0xd4]
          vcvtbiasph2bf8s ymm2, zmm3, zmm4

// CHECK: vcvtbiasph2bf8s ymm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x4f,0x74,0xd4]
          vcvtbiasph2bf8s ymm2 {k7}, zmm3, zmm4

// CHECK: vcvtbiasph2bf8s ymm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x64,0xcf,0x74,0xd4]
          vcvtbiasph2bf8s ymm2 {k7} {z}, zmm3, zmm4

// CHECK: vcvtbiasph2bf8s xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x74,0xd4]
          vcvtbiasph2bf8s xmm2, xmm3, xmm4

// CHECK: vcvtbiasph2bf8s xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x0f,0x74,0xd4]
          vcvtbiasph2bf8s xmm2 {k7}, xmm3, xmm4

// CHECK: vcvtbiasph2bf8s xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x8f,0x74,0xd4]
          vcvtbiasph2bf8s xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vcvtbiasph2bf8s xmm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x74,0xd4]
          vcvtbiasph2bf8s xmm2, ymm3, ymm4

// CHECK: vcvtbiasph2bf8s xmm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x64,0x2f,0x74,0xd4]
          vcvtbiasph2bf8s xmm2 {k7}, ymm3, ymm4

// CHECK: vcvtbiasph2bf8s xmm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x64,0xaf,0x74,0xd4]
          vcvtbiasph2bf8s xmm2 {k7} {z}, ymm3, ymm4

// CHECK: vcvtbiasph2bf8s xmm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8s xmm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbiasph2bf8s xmm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x64,0x2f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8s xmm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbiasph2bf8s xmm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x64,0x38,0x74,0x10]
          vcvtbiasph2bf8s xmm2, ymm3, word ptr [eax]{1to16}

// CHECK: vcvtbiasph2bf8s xmm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x74,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2bf8s xmm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtbiasph2bf8s xmm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x64,0xaf,0x74,0x51,0x7f]
          vcvtbiasph2bf8s xmm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vcvtbiasph2bf8s xmm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x64,0xbf,0x74,0x52,0x80]
          vcvtbiasph2bf8s xmm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vcvtbiasph2bf8s ymm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8s ymm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbiasph2bf8s ymm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x64,0x4f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8s ymm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbiasph2bf8s ymm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x64,0x58,0x74,0x10]
          vcvtbiasph2bf8s ymm2, zmm3, word ptr [eax]{1to32}

// CHECK: vcvtbiasph2bf8s ymm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x74,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2bf8s ymm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtbiasph2bf8s ymm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x64,0xcf,0x74,0x51,0x7f]
          vcvtbiasph2bf8s ymm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vcvtbiasph2bf8s ymm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x64,0xdf,0x74,0x52,0x80]
          vcvtbiasph2bf8s ymm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vcvtbiasph2bf8s xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8s xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbiasph2bf8s xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x64,0x0f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8s xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbiasph2bf8s xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x64,0x18,0x74,0x10]
          vcvtbiasph2bf8s xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vcvtbiasph2bf8s xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x74,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2bf8s xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vcvtbiasph2bf8s xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x64,0x8f,0x74,0x51,0x7f]
          vcvtbiasph2bf8s xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vcvtbiasph2bf8s xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x64,0x9f,0x74,0x52,0x80]
          vcvtbiasph2bf8s xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vcvtbiasph2hf8 ymm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x18,0xd4]
          vcvtbiasph2hf8 ymm2, zmm3, zmm4

// CHECK: vcvtbiasph2hf8 ymm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x4f,0x18,0xd4]
          vcvtbiasph2hf8 ymm2 {k7}, zmm3, zmm4

// CHECK: vcvtbiasph2hf8 ymm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x64,0xcf,0x18,0xd4]
          vcvtbiasph2hf8 ymm2 {k7} {z}, zmm3, zmm4

// CHECK: vcvtbiasph2hf8 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x18,0xd4]
          vcvtbiasph2hf8 xmm2, xmm3, xmm4

// CHECK: vcvtbiasph2hf8 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x0f,0x18,0xd4]
          vcvtbiasph2hf8 xmm2 {k7}, xmm3, xmm4

// CHECK: vcvtbiasph2hf8 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x8f,0x18,0xd4]
          vcvtbiasph2hf8 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vcvtbiasph2hf8 xmm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x18,0xd4]
          vcvtbiasph2hf8 xmm2, ymm3, ymm4

// CHECK: vcvtbiasph2hf8 xmm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x64,0x2f,0x18,0xd4]
          vcvtbiasph2hf8 xmm2 {k7}, ymm3, ymm4

// CHECK: vcvtbiasph2hf8 xmm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x64,0xaf,0x18,0xd4]
          vcvtbiasph2hf8 xmm2 {k7} {z}, ymm3, ymm4

// CHECK: vcvtbiasph2hf8 xmm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8 xmm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbiasph2hf8 xmm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x64,0x2f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8 xmm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbiasph2hf8 xmm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x64,0x38,0x18,0x10]
          vcvtbiasph2hf8 xmm2, ymm3, word ptr [eax]{1to16}

// CHECK: vcvtbiasph2hf8 xmm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x18,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2hf8 xmm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtbiasph2hf8 xmm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x64,0xaf,0x18,0x51,0x7f]
          vcvtbiasph2hf8 xmm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vcvtbiasph2hf8 xmm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x64,0xbf,0x18,0x52,0x80]
          vcvtbiasph2hf8 xmm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vcvtbiasph2hf8 ymm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8 ymm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbiasph2hf8 ymm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x64,0x4f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8 ymm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbiasph2hf8 ymm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x64,0x58,0x18,0x10]
          vcvtbiasph2hf8 ymm2, zmm3, word ptr [eax]{1to32}

// CHECK: vcvtbiasph2hf8 ymm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x18,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2hf8 ymm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtbiasph2hf8 ymm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x64,0xcf,0x18,0x51,0x7f]
          vcvtbiasph2hf8 ymm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vcvtbiasph2hf8 ymm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x64,0xdf,0x18,0x52,0x80]
          vcvtbiasph2hf8 ymm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vcvtbiasph2hf8 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbiasph2hf8 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x64,0x0f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbiasph2hf8 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x64,0x18,0x18,0x10]
          vcvtbiasph2hf8 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vcvtbiasph2hf8 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x18,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2hf8 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vcvtbiasph2hf8 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x64,0x8f,0x18,0x51,0x7f]
          vcvtbiasph2hf8 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vcvtbiasph2hf8 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x64,0x9f,0x18,0x52,0x80]
          vcvtbiasph2hf8 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vcvtbiasph2hf8s ymm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x1b,0xd4]
          vcvtbiasph2hf8s ymm2, zmm3, zmm4

// CHECK: vcvtbiasph2hf8s ymm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x4f,0x1b,0xd4]
          vcvtbiasph2hf8s ymm2 {k7}, zmm3, zmm4

// CHECK: vcvtbiasph2hf8s ymm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x64,0xcf,0x1b,0xd4]
          vcvtbiasph2hf8s ymm2 {k7} {z}, zmm3, zmm4

// CHECK: vcvtbiasph2hf8s xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x1b,0xd4]
          vcvtbiasph2hf8s xmm2, xmm3, xmm4

// CHECK: vcvtbiasph2hf8s xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x0f,0x1b,0xd4]
          vcvtbiasph2hf8s xmm2 {k7}, xmm3, xmm4

// CHECK: vcvtbiasph2hf8s xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x64,0x8f,0x1b,0xd4]
          vcvtbiasph2hf8s xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vcvtbiasph2hf8s xmm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x1b,0xd4]
          vcvtbiasph2hf8s xmm2, ymm3, ymm4

// CHECK: vcvtbiasph2hf8s xmm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x64,0x2f,0x1b,0xd4]
          vcvtbiasph2hf8s xmm2 {k7}, ymm3, ymm4

// CHECK: vcvtbiasph2hf8s xmm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x64,0xaf,0x1b,0xd4]
          vcvtbiasph2hf8s xmm2 {k7} {z}, ymm3, ymm4

// CHECK: vcvtbiasph2hf8s xmm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8s xmm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbiasph2hf8s xmm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x64,0x2f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8s xmm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbiasph2hf8s xmm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x64,0x38,0x1b,0x10]
          vcvtbiasph2hf8s xmm2, ymm3, word ptr [eax]{1to16}

// CHECK: vcvtbiasph2hf8s xmm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x1b,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2hf8s xmm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtbiasph2hf8s xmm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x64,0xaf,0x1b,0x51,0x7f]
          vcvtbiasph2hf8s xmm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vcvtbiasph2hf8s xmm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x64,0xbf,0x1b,0x52,0x80]
          vcvtbiasph2hf8s xmm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vcvtbiasph2hf8s ymm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8s ymm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbiasph2hf8s ymm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x64,0x4f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8s ymm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbiasph2hf8s ymm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x64,0x58,0x1b,0x10]
          vcvtbiasph2hf8s ymm2, zmm3, word ptr [eax]{1to32}

// CHECK: vcvtbiasph2hf8s ymm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x1b,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2hf8s ymm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtbiasph2hf8s ymm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x64,0xcf,0x1b,0x51,0x7f]
          vcvtbiasph2hf8s ymm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vcvtbiasph2hf8s ymm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x64,0xdf,0x1b,0x52,0x80]
          vcvtbiasph2hf8s ymm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vcvtbiasph2hf8s xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8s xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtbiasph2hf8s xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x64,0x0f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8s xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtbiasph2hf8s xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x64,0x18,0x1b,0x10]
          vcvtbiasph2hf8s xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vcvtbiasph2hf8s xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x1b,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2hf8s xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vcvtbiasph2hf8s xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x64,0x8f,0x1b,0x51,0x7f]
          vcvtbiasph2hf8s xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vcvtbiasph2hf8s xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x64,0x9f,0x1b,0x52,0x80]
          vcvtbiasph2hf8s xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vcvthf82ph xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x1e,0xd3]
          vcvthf82ph xmm2, xmm3

// CHECK: vcvthf82ph xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x1e,0xd3]
          vcvthf82ph xmm2 {k7}, xmm3

// CHECK: vcvthf82ph xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x1e,0xd3]
          vcvthf82ph xmm2 {k7} {z}, xmm3

// CHECK: vcvthf82ph ymm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x1e,0xd3]
          vcvthf82ph ymm2, xmm3

// CHECK: vcvthf82ph ymm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x1e,0xd3]
          vcvthf82ph ymm2 {k7}, xmm3

// CHECK: vcvthf82ph ymm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x1e,0xd3]
          vcvthf82ph ymm2 {k7} {z}, xmm3

// CHECK: vcvthf82ph zmm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x1e,0xd3]
          vcvthf82ph zmm2, ymm3

// CHECK: vcvthf82ph zmm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x1e,0xd3]
          vcvthf82ph zmm2 {k7}, ymm3

// CHECK: vcvthf82ph zmm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x1e,0xd3]
          vcvthf82ph zmm2 {k7} {z}, ymm3

// CHECK: vcvthf82ph xmm2, qword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x1e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvthf82ph xmm2, qword ptr [esp + 8*esi + 268435456]

// CHECK: vcvthf82ph xmm2 {k7}, qword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x1e,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvthf82ph xmm2 {k7}, qword ptr [edi + 4*eax + 291]

// CHECK: vcvthf82ph xmm2, qword ptr [eax]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x1e,0x10]
          vcvthf82ph xmm2, qword ptr [eax]

// CHECK: vcvthf82ph xmm2, qword ptr [2*ebp - 256]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x1e,0x14,0x6d,0x00,0xff,0xff,0xff]
          vcvthf82ph xmm2, qword ptr [2*ebp - 256]

// CHECK: vcvthf82ph xmm2 {k7} {z}, qword ptr [ecx + 1016]
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x1e,0x51,0x7f]
          vcvthf82ph xmm2 {k7} {z}, qword ptr [ecx + 1016]

// CHECK: vcvthf82ph xmm2 {k7} {z}, qword ptr [edx - 1024]
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x1e,0x52,0x80]
          vcvthf82ph xmm2 {k7} {z}, qword ptr [edx - 1024]

// CHECK: vcvthf82ph ymm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x1e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvthf82ph ymm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvthf82ph ymm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x1e,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvthf82ph ymm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvthf82ph ymm2, xmmword ptr [eax]
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x1e,0x10]
          vcvthf82ph ymm2, xmmword ptr [eax]

// CHECK: vcvthf82ph ymm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x1e,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvthf82ph ymm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvthf82ph ymm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x1e,0x51,0x7f]
          vcvthf82ph ymm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvthf82ph ymm2 {k7} {z}, xmmword ptr [edx - 2048]
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x1e,0x52,0x80]
          vcvthf82ph ymm2 {k7} {z}, xmmword ptr [edx - 2048]

// CHECK: vcvthf82ph zmm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x1e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvthf82ph zmm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvthf82ph zmm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x1e,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvthf82ph zmm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvthf82ph zmm2, ymmword ptr [eax]
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x1e,0x10]
          vcvthf82ph zmm2, ymmword ptr [eax]

// CHECK: vcvthf82ph zmm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x1e,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvthf82ph zmm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvthf82ph zmm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x1e,0x51,0x7f]
          vcvthf82ph zmm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvthf82ph zmm2 {k7} {z}, ymmword ptr [edx - 4096]
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x1e,0x52,0x80]
          vcvthf82ph zmm2 {k7} {z}, ymmword ptr [edx - 4096]

// CHECK: vcvt2ph2bf8 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x74,0xd4]
          vcvt2ph2bf8 ymm2, ymm3, ymm4

// CHECK: vcvt2ph2bf8 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x67,0x2f,0x74,0xd4]
          vcvt2ph2bf8 ymm2 {k7}, ymm3, ymm4

// CHECK: vcvt2ph2bf8 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x67,0xaf,0x74,0xd4]
          vcvt2ph2bf8 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vcvt2ph2bf8 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x74,0xd4]
          vcvt2ph2bf8 zmm2, zmm3, zmm4

// CHECK: vcvt2ph2bf8 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x4f,0x74,0xd4]
          vcvt2ph2bf8 zmm2 {k7}, zmm3, zmm4

// CHECK: vcvt2ph2bf8 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x67,0xcf,0x74,0xd4]
          vcvt2ph2bf8 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vcvt2ph2bf8 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x74,0xd4]
          vcvt2ph2bf8 xmm2, xmm3, xmm4

// CHECK: vcvt2ph2bf8 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x0f,0x74,0xd4]
          vcvt2ph2bf8 xmm2 {k7}, xmm3, xmm4

// CHECK: vcvt2ph2bf8 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x8f,0x74,0xd4]
          vcvt2ph2bf8 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vcvt2ph2bf8 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ph2bf8 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x4f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ph2bf8 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf2,0x67,0x58,0x74,0x10]
          vcvt2ph2bf8 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vcvt2ph2bf8 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x74,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ph2bf8 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vcvt2ph2bf8 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x67,0xcf,0x74,0x51,0x7f]
          vcvt2ph2bf8 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vcvt2ph2bf8 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf2,0x67,0xdf,0x74,0x52,0x80]
          vcvt2ph2bf8 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vcvt2ph2bf8 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ph2bf8 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x2f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ph2bf8 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x67,0x38,0x74,0x10]
          vcvt2ph2bf8 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vcvt2ph2bf8 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x74,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ph2bf8 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vcvt2ph2bf8 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x67,0xaf,0x74,0x51,0x7f]
          vcvt2ph2bf8 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vcvt2ph2bf8 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf2,0x67,0xbf,0x74,0x52,0x80]
          vcvt2ph2bf8 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vcvt2ph2bf8 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ph2bf8 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x0f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ph2bf8 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x67,0x18,0x74,0x10]
          vcvt2ph2bf8 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vcvt2ph2bf8 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x74,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ph2bf8 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vcvt2ph2bf8 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x67,0x8f,0x74,0x51,0x7f]
          vcvt2ph2bf8 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vcvt2ph2bf8 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf2,0x67,0x9f,0x74,0x52,0x80]
          vcvt2ph2bf8 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vcvt2ph2bf8s ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x74,0xd4]
          vcvt2ph2bf8s ymm2, ymm3, ymm4

// CHECK: vcvt2ph2bf8s ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x67,0x2f,0x74,0xd4]
          vcvt2ph2bf8s ymm2 {k7}, ymm3, ymm4

// CHECK: vcvt2ph2bf8s ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x67,0xaf,0x74,0xd4]
          vcvt2ph2bf8s ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vcvt2ph2bf8s zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x74,0xd4]
          vcvt2ph2bf8s zmm2, zmm3, zmm4

// CHECK: vcvt2ph2bf8s zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x4f,0x74,0xd4]
          vcvt2ph2bf8s zmm2 {k7}, zmm3, zmm4

// CHECK: vcvt2ph2bf8s zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x67,0xcf,0x74,0xd4]
          vcvt2ph2bf8s zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vcvt2ph2bf8s xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x74,0xd4]
          vcvt2ph2bf8s xmm2, xmm3, xmm4

// CHECK: vcvt2ph2bf8s xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x0f,0x74,0xd4]
          vcvt2ph2bf8s xmm2 {k7}, xmm3, xmm4

// CHECK: vcvt2ph2bf8s xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x8f,0x74,0xd4]
          vcvt2ph2bf8s xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vcvt2ph2bf8s zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8s zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ph2bf8s zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x67,0x4f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8s zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ph2bf8s zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x67,0x58,0x74,0x10]
          vcvt2ph2bf8s zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vcvt2ph2bf8s zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x74,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ph2bf8s zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vcvt2ph2bf8s zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x67,0xcf,0x74,0x51,0x7f]
          vcvt2ph2bf8s zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vcvt2ph2bf8s zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x67,0xdf,0x74,0x52,0x80]
          vcvt2ph2bf8s zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vcvt2ph2bf8s ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8s ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ph2bf8s ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x67,0x2f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8s ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ph2bf8s ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x67,0x38,0x74,0x10]
          vcvt2ph2bf8s ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vcvt2ph2bf8s ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x74,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ph2bf8s ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vcvt2ph2bf8s ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x67,0xaf,0x74,0x51,0x7f]
          vcvt2ph2bf8s ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vcvt2ph2bf8s ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x67,0xbf,0x74,0x52,0x80]
          vcvt2ph2bf8s ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vcvt2ph2bf8s xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8s xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ph2bf8s xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x67,0x0f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8s xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ph2bf8s xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x67,0x18,0x74,0x10]
          vcvt2ph2bf8s xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vcvt2ph2bf8s xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x74,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ph2bf8s xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vcvt2ph2bf8s xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x67,0x8f,0x74,0x51,0x7f]
          vcvt2ph2bf8s xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vcvt2ph2bf8s xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x67,0x9f,0x74,0x52,0x80]
          vcvt2ph2bf8s xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vcvt2ph2hf8 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x18,0xd4]
          vcvt2ph2hf8 ymm2, ymm3, ymm4

// CHECK: vcvt2ph2hf8 ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x67,0x2f,0x18,0xd4]
          vcvt2ph2hf8 ymm2 {k7}, ymm3, ymm4

// CHECK: vcvt2ph2hf8 ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x67,0xaf,0x18,0xd4]
          vcvt2ph2hf8 ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vcvt2ph2hf8 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x18,0xd4]
          vcvt2ph2hf8 zmm2, zmm3, zmm4

// CHECK: vcvt2ph2hf8 zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x4f,0x18,0xd4]
          vcvt2ph2hf8 zmm2 {k7}, zmm3, zmm4

// CHECK: vcvt2ph2hf8 zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x67,0xcf,0x18,0xd4]
          vcvt2ph2hf8 zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vcvt2ph2hf8 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x18,0xd4]
          vcvt2ph2hf8 xmm2, xmm3, xmm4

// CHECK: vcvt2ph2hf8 xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x0f,0x18,0xd4]
          vcvt2ph2hf8 xmm2 {k7}, xmm3, xmm4

// CHECK: vcvt2ph2hf8 xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x8f,0x18,0xd4]
          vcvt2ph2hf8 xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vcvt2ph2hf8 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ph2hf8 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x67,0x4f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8 zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ph2hf8 zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x67,0x58,0x18,0x10]
          vcvt2ph2hf8 zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vcvt2ph2hf8 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x18,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ph2hf8 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vcvt2ph2hf8 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x67,0xcf,0x18,0x51,0x7f]
          vcvt2ph2hf8 zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vcvt2ph2hf8 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x67,0xdf,0x18,0x52,0x80]
          vcvt2ph2hf8 zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vcvt2ph2hf8 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ph2hf8 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x67,0x2f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8 ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ph2hf8 ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x67,0x38,0x18,0x10]
          vcvt2ph2hf8 ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vcvt2ph2hf8 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x18,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ph2hf8 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vcvt2ph2hf8 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x67,0xaf,0x18,0x51,0x7f]
          vcvt2ph2hf8 ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vcvt2ph2hf8 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x67,0xbf,0x18,0x52,0x80]
          vcvt2ph2hf8 ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vcvt2ph2hf8 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ph2hf8 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x67,0x0f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8 xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ph2hf8 xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x67,0x18,0x18,0x10]
          vcvt2ph2hf8 xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vcvt2ph2hf8 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x18,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ph2hf8 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vcvt2ph2hf8 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x67,0x8f,0x18,0x51,0x7f]
          vcvt2ph2hf8 xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vcvt2ph2hf8 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x67,0x9f,0x18,0x52,0x80]
          vcvt2ph2hf8 xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vcvt2ph2hf8s ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x1b,0xd4]
          vcvt2ph2hf8s ymm2, ymm3, ymm4

// CHECK: vcvt2ph2hf8s ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x67,0x2f,0x1b,0xd4]
          vcvt2ph2hf8s ymm2 {k7}, ymm3, ymm4

// CHECK: vcvt2ph2hf8s ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf5,0x67,0xaf,0x1b,0xd4]
          vcvt2ph2hf8s ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vcvt2ph2hf8s zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x1b,0xd4]
          vcvt2ph2hf8s zmm2, zmm3, zmm4

// CHECK: vcvt2ph2hf8s zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x4f,0x1b,0xd4]
          vcvt2ph2hf8s zmm2 {k7}, zmm3, zmm4

// CHECK: vcvt2ph2hf8s zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf5,0x67,0xcf,0x1b,0xd4]
          vcvt2ph2hf8s zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vcvt2ph2hf8s xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x1b,0xd4]
          vcvt2ph2hf8s xmm2, xmm3, xmm4

// CHECK: vcvt2ph2hf8s xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x0f,0x1b,0xd4]
          vcvt2ph2hf8s xmm2 {k7}, xmm3, xmm4

// CHECK: vcvt2ph2hf8s xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf5,0x67,0x8f,0x1b,0xd4]
          vcvt2ph2hf8s xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vcvt2ph2hf8s zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8s zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ph2hf8s zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x67,0x4f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8s zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ph2hf8s zmm2, zmm3, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x67,0x58,0x1b,0x10]
          vcvt2ph2hf8s zmm2, zmm3, word ptr [eax]{1to32}

// CHECK: vcvt2ph2hf8s zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x1b,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ph2hf8s zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vcvt2ph2hf8s zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x67,0xcf,0x1b,0x51,0x7f]
          vcvt2ph2hf8s zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vcvt2ph2hf8s zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x67,0xdf,0x1b,0x52,0x80]
          vcvt2ph2hf8s zmm2 {k7} {z}, zmm3, word ptr [edx - 256]{1to32}

// CHECK: vcvt2ph2hf8s ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8s ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ph2hf8s ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x67,0x2f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8s ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ph2hf8s ymm2, ymm3, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x67,0x38,0x1b,0x10]
          vcvt2ph2hf8s ymm2, ymm3, word ptr [eax]{1to16}

// CHECK: vcvt2ph2hf8s ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x1b,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ph2hf8s ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vcvt2ph2hf8s ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x67,0xaf,0x1b,0x51,0x7f]
          vcvt2ph2hf8s ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vcvt2ph2hf8s ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x67,0xbf,0x1b,0x52,0x80]
          vcvt2ph2hf8s ymm2 {k7} {z}, ymm3, word ptr [edx - 256]{1to16}

// CHECK: vcvt2ph2hf8s xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8s xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvt2ph2hf8s xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x67,0x0f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8s xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvt2ph2hf8s xmm2, xmm3, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x67,0x18,0x1b,0x10]
          vcvt2ph2hf8s xmm2, xmm3, word ptr [eax]{1to8}

// CHECK: vcvt2ph2hf8s xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x1b,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ph2hf8s xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vcvt2ph2hf8s xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x67,0x8f,0x1b,0x51,0x7f]
          vcvt2ph2hf8s xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vcvt2ph2hf8s xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x67,0x9f,0x1b,0x52,0x80]
          vcvt2ph2hf8s xmm2 {k7} {z}, xmm3, word ptr [edx - 256]{1to8}

// CHECK: vcvtph2bf8 xmm2, xmm3
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x74,0xd3]
          vcvtph2bf8 xmm2, xmm3

// CHECK: vcvtph2bf8 xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf2,0x7e,0x0f,0x74,0xd3]
          vcvtph2bf8 xmm2 {k7}, xmm3

// CHECK: vcvtph2bf8 xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf2,0x7e,0x8f,0x74,0xd3]
          vcvtph2bf8 xmm2 {k7} {z}, xmm3

// CHECK: vcvtph2bf8 ymm2, zmm3
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x74,0xd3]
          vcvtph2bf8 ymm2, zmm3

// CHECK: vcvtph2bf8 ymm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf2,0x7e,0x4f,0x74,0xd3]
          vcvtph2bf8 ymm2 {k7}, zmm3

// CHECK: vcvtph2bf8 ymm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf2,0x7e,0xcf,0x74,0xd3]
          vcvtph2bf8 ymm2 {k7} {z}, zmm3

// CHECK: vcvtph2bf8 xmm2, ymm3
// CHECK: encoding: [0x62,0xf2,0x7e,0x28,0x74,0xd3]
          vcvtph2bf8 xmm2, ymm3

// CHECK: vcvtph2bf8 xmm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf2,0x7e,0x2f,0x74,0xd3]
          vcvtph2bf8 xmm2 {k7}, ymm3

// CHECK: vcvtph2bf8 xmm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf2,0x7e,0xaf,0x74,0xd3]
          vcvtph2bf8 xmm2 {k7} {z}, ymm3

// CHECK: vcvtph2bf8 xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2bf8 xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2bf8 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x7e,0x0f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2bf8 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2bf8 xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x7e,0x18,0x74,0x10]
          vcvtph2bf8 xmm2, word ptr [eax]{1to8}

// CHECK: vcvtph2bf8 xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x74,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2bf8 xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvtph2bf8 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x7e,0x8f,0x74,0x51,0x7f]
          vcvtph2bf8 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvtph2bf8 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf2,0x7e,0x9f,0x74,0x52,0x80]
          vcvtph2bf8 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvtph2bf8 xmm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x7e,0x38,0x74,0x10]
          vcvtph2bf8 xmm2, word ptr [eax]{1to16}

// CHECK: vcvtph2bf8 xmm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf2,0x7e,0x28,0x74,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2bf8 xmm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtph2bf8 xmm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x7e,0xaf,0x74,0x51,0x7f]
          vcvtph2bf8 xmm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvtph2bf8 xmm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf2,0x7e,0xbf,0x74,0x52,0x80]
          vcvtph2bf8 xmm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvtph2bf8 ymm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2bf8 ymm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2bf8 ymm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x7e,0x4f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2bf8 ymm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2bf8 ymm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf2,0x7e,0x58,0x74,0x10]
          vcvtph2bf8 ymm2, word ptr [eax]{1to32}

// CHECK: vcvtph2bf8 ymm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x74,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2bf8 ymm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtph2bf8 ymm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x7e,0xcf,0x74,0x51,0x7f]
          vcvtph2bf8 ymm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvtph2bf8 ymm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf2,0x7e,0xdf,0x74,0x52,0x80]
          vcvtph2bf8 ymm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvtph2bf8s xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x74,0xd3]
          vcvtph2bf8s xmm2, xmm3

// CHECK: vcvtph2bf8s xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x74,0xd3]
          vcvtph2bf8s xmm2 {k7}, xmm3

// CHECK: vcvtph2bf8s xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x8f,0x74,0xd3]
          vcvtph2bf8s xmm2 {k7} {z}, xmm3

// CHECK: vcvtph2bf8s ymm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x74,0xd3]
          vcvtph2bf8s ymm2, zmm3

// CHECK: vcvtph2bf8s ymm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x74,0xd3]
          vcvtph2bf8s ymm2 {k7}, zmm3

// CHECK: vcvtph2bf8s ymm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0xcf,0x74,0xd3]
          vcvtph2bf8s ymm2 {k7} {z}, zmm3

// CHECK: vcvtph2bf8s xmm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x74,0xd3]
          vcvtph2bf8s xmm2, ymm3

// CHECK: vcvtph2bf8s xmm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x2f,0x74,0xd3]
          vcvtph2bf8s xmm2 {k7}, ymm3

// CHECK: vcvtph2bf8s xmm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7e,0xaf,0x74,0xd3]
          vcvtph2bf8s xmm2 {k7} {z}, ymm3

// CHECK: vcvtph2bf8s xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2bf8s xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2bf8s xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2bf8s xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2bf8s xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x74,0x10]
          vcvtph2bf8s xmm2, word ptr [eax]{1to8}

// CHECK: vcvtph2bf8s xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x74,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2bf8s xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvtph2bf8s xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7e,0x8f,0x74,0x51,0x7f]
          vcvtph2bf8s xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvtph2bf8s xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7e,0x9f,0x74,0x52,0x80]
          vcvtph2bf8s xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvtph2bf8s xmm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7e,0x38,0x74,0x10]
          vcvtph2bf8s xmm2, word ptr [eax]{1to16}

// CHECK: vcvtph2bf8s xmm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x74,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2bf8s xmm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtph2bf8s xmm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7e,0xaf,0x74,0x51,0x7f]
          vcvtph2bf8s xmm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvtph2bf8s xmm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7e,0xbf,0x74,0x52,0x80]
          vcvtph2bf8s xmm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvtph2bf8s ymm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2bf8s ymm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2bf8s ymm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2bf8s ymm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2bf8s ymm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7e,0x58,0x74,0x10]
          vcvtph2bf8s ymm2, word ptr [eax]{1to32}

// CHECK: vcvtph2bf8s ymm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x74,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2bf8s ymm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtph2bf8s ymm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7e,0xcf,0x74,0x51,0x7f]
          vcvtph2bf8s ymm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvtph2bf8s ymm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7e,0xdf,0x74,0x52,0x80]
          vcvtph2bf8s ymm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvtph2hf8 xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x18,0xd3]
          vcvtph2hf8 xmm2, xmm3

// CHECK: vcvtph2hf8 xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x18,0xd3]
          vcvtph2hf8 xmm2 {k7}, xmm3

// CHECK: vcvtph2hf8 xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x8f,0x18,0xd3]
          vcvtph2hf8 xmm2 {k7} {z}, xmm3

// CHECK: vcvtph2hf8 ymm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x18,0xd3]
          vcvtph2hf8 ymm2, zmm3

// CHECK: vcvtph2hf8 ymm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x18,0xd3]
          vcvtph2hf8 ymm2 {k7}, zmm3

// CHECK: vcvtph2hf8 ymm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0xcf,0x18,0xd3]
          vcvtph2hf8 ymm2 {k7} {z}, zmm3

// CHECK: vcvtph2hf8 xmm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x18,0xd3]
          vcvtph2hf8 xmm2, ymm3

// CHECK: vcvtph2hf8 xmm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x2f,0x18,0xd3]
          vcvtph2hf8 xmm2 {k7}, ymm3

// CHECK: vcvtph2hf8 xmm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7e,0xaf,0x18,0xd3]
          vcvtph2hf8 xmm2 {k7} {z}, ymm3

// CHECK: vcvtph2hf8 xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2hf8 xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2hf8 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2hf8 xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2hf8 xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x18,0x10]
          vcvtph2hf8 xmm2, word ptr [eax]{1to8}

// CHECK: vcvtph2hf8 xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x18,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2hf8 xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvtph2hf8 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7e,0x8f,0x18,0x51,0x7f]
          vcvtph2hf8 xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvtph2hf8 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7e,0x9f,0x18,0x52,0x80]
          vcvtph2hf8 xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvtph2hf8 xmm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7e,0x38,0x18,0x10]
          vcvtph2hf8 xmm2, word ptr [eax]{1to16}

// CHECK: vcvtph2hf8 xmm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x18,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2hf8 xmm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtph2hf8 xmm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7e,0xaf,0x18,0x51,0x7f]
          vcvtph2hf8 xmm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvtph2hf8 xmm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7e,0xbf,0x18,0x52,0x80]
          vcvtph2hf8 xmm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvtph2hf8 ymm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2hf8 ymm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2hf8 ymm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2hf8 ymm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2hf8 ymm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7e,0x58,0x18,0x10]
          vcvtph2hf8 ymm2, word ptr [eax]{1to32}

// CHECK: vcvtph2hf8 ymm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x18,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2hf8 ymm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtph2hf8 ymm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7e,0xcf,0x18,0x51,0x7f]
          vcvtph2hf8 ymm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvtph2hf8 ymm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7e,0xdf,0x18,0x52,0x80]
          vcvtph2hf8 ymm2 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvtph2hf8s xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x1b,0xd3]
          vcvtph2hf8s xmm2, xmm3

// CHECK: vcvtph2hf8s xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x1b,0xd3]
          vcvtph2hf8s xmm2 {k7}, xmm3

// CHECK: vcvtph2hf8s xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x8f,0x1b,0xd3]
          vcvtph2hf8s xmm2 {k7} {z}, xmm3

// CHECK: vcvtph2hf8s ymm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x1b,0xd3]
          vcvtph2hf8s ymm2, zmm3

// CHECK: vcvtph2hf8s ymm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x1b,0xd3]
          vcvtph2hf8s ymm2 {k7}, zmm3

// CHECK: vcvtph2hf8s ymm2 {k7} {z}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7e,0xcf,0x1b,0xd3]
          vcvtph2hf8s ymm2 {k7} {z}, zmm3

// CHECK: vcvtph2hf8s xmm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x1b,0xd3]
          vcvtph2hf8s xmm2, ymm3

// CHECK: vcvtph2hf8s xmm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7e,0x2f,0x1b,0xd3]
          vcvtph2hf8s xmm2 {k7}, ymm3

// CHECK: vcvtph2hf8s xmm2 {k7} {z}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7e,0xaf,0x1b,0xd3]
          vcvtph2hf8s xmm2 {k7} {z}, ymm3

// CHECK: vcvtph2hf8s xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2hf8s xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2hf8s xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2hf8s xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2hf8s xmm2, word ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x1b,0x10]
          vcvtph2hf8s xmm2, word ptr [eax]{1to8}

// CHECK: vcvtph2hf8s xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x1b,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2hf8s xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvtph2hf8s xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7e,0x8f,0x1b,0x51,0x7f]
          vcvtph2hf8s xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvtph2hf8s xmm2 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7e,0x9f,0x1b,0x52,0x80]
          vcvtph2hf8s xmm2 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvtph2hf8s xmm2, word ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7e,0x38,0x1b,0x10]
          vcvtph2hf8s xmm2, word ptr [eax]{1to16}

// CHECK: vcvtph2hf8s xmm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x1b,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2hf8s xmm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvtph2hf8s xmm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7e,0xaf,0x1b,0x51,0x7f]
          vcvtph2hf8s xmm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvtph2hf8s xmm2 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7e,0xbf,0x1b,0x52,0x80]
          vcvtph2hf8s xmm2 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvtph2hf8s ymm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2hf8s ymm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2hf8s ymm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2hf8s ymm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvtph2hf8s ymm2, word ptr [eax]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7e,0x58,0x1b,0x10]
          vcvtph2hf8s ymm2, word ptr [eax]{1to32}

// CHECK: vcvtph2hf8s ymm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x1b,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2hf8s ymm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvtph2hf8s ymm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7e,0xcf,0x1b,0x51,0x7f]
          vcvtph2hf8s ymm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvtph2hf8s ymm2 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7e,0xdf,0x1b,0x52,0x80]
          vcvtph2hf8s ymm2 {k7} {z}, word ptr [edx - 256]{1to32}

