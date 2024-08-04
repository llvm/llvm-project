// RUN: llvm-mc -triple i386 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// VMPSADBW

// CHECK: vmpsadbw xmm2, xmm3, xmm4, 123
// CHECK: encoding: [0xc4,0xe3,0x61,0x42,0xd4,0x7b]
          vmpsadbw xmm2, xmm3, xmm4, 123

// CHECK: vmpsadbw xmm2 {k7}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0x0f,0x42,0xd4,0x7b]
          vmpsadbw xmm2 {k7}, xmm3, xmm4, 123

// CHECK: vmpsadbw xmm2 {k7} {z}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0x8f,0x42,0xd4,0x7b]
          vmpsadbw xmm2 {k7} {z}, xmm3, xmm4, 123

// CHECK: vmpsadbw ymm2, ymm3, ymm4, 123
// CHECK: encoding: [0xc4,0xe3,0x65,0x42,0xd4,0x7b]
          vmpsadbw ymm2, ymm3, ymm4, 123

// CHECK: vmpsadbw ymm2 {k7}, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0x2f,0x42,0xd4,0x7b]
          vmpsadbw ymm2 {k7}, ymm3, ymm4, 123

// CHECK: vmpsadbw ymm2 {k7} {z}, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0xaf,0x42,0xd4,0x7b]
          vmpsadbw ymm2 {k7} {z}, ymm3, ymm4, 123

// CHECK: vmpsadbw zmm2, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0x48,0x42,0xd4,0x7b]
          vmpsadbw zmm2, zmm3, zmm4, 123

// CHECK: vmpsadbw zmm2 {k7}, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0x4f,0x42,0xd4,0x7b]
          vmpsadbw zmm2 {k7}, zmm3, zmm4, 123

// CHECK: vmpsadbw zmm2 {k7} {z}, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0xcf,0x42,0xd4,0x7b]
          vmpsadbw zmm2 {k7} {z}, zmm3, zmm4, 123

// CHECK: vmpsadbw xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0xc4,0xe3,0x61,0x42,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vmpsadbw xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vmpsadbw xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x0f,0x42,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vmpsadbw xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vmpsadbw xmm2, xmm3, xmmword ptr [eax], 123
// CHECK: encoding: [0xc4,0xe3,0x61,0x42,0x10,0x7b]
          vmpsadbw xmm2, xmm3, xmmword ptr [eax], 123

// CHECK: vmpsadbw xmm2, xmm3, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0xc4,0xe3,0x61,0x42,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vmpsadbw xmm2, xmm3, xmmword ptr [2*ebp - 512], 123

// CHECK: vmpsadbw xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x8f,0x42,0x51,0x7f,0x7b]
          vmpsadbw xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032], 123

// CHECK: vmpsadbw xmm2 {k7} {z}, xmm3, xmmword ptr [edx - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x8f,0x42,0x52,0x80,0x7b]
          vmpsadbw xmm2 {k7} {z}, xmm3, xmmword ptr [edx - 2048], 123

// CHECK: vmpsadbw ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0xc4,0xe3,0x65,0x42,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vmpsadbw ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vmpsadbw ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x2f,0x42,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vmpsadbw ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vmpsadbw ymm2, ymm3, ymmword ptr [eax], 123
// CHECK: encoding: [0xc4,0xe3,0x65,0x42,0x10,0x7b]
          vmpsadbw ymm2, ymm3, ymmword ptr [eax], 123

// CHECK: vmpsadbw ymm2, ymm3, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0xc4,0xe3,0x65,0x42,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vmpsadbw ymm2, ymm3, ymmword ptr [2*ebp - 1024], 123

// CHECK: vmpsadbw ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x66,0xaf,0x42,0x51,0x7f,0x7b]
          vmpsadbw ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064], 123

// CHECK: vmpsadbw ymm2 {k7} {z}, ymm3, ymmword ptr [edx - 4096], 123
// CHECK: encoding: [0x62,0xf3,0x66,0xaf,0x42,0x52,0x80,0x7b]
          vmpsadbw ymm2 {k7} {z}, ymm3, ymmword ptr [edx - 4096], 123

// CHECK: vmpsadbw zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x48,0x42,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vmpsadbw zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vmpsadbw zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x4f,0x42,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vmpsadbw zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vmpsadbw zmm2, zmm3, zmmword ptr [eax], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x48,0x42,0x10,0x7b]
          vmpsadbw zmm2, zmm3, zmmword ptr [eax], 123

// CHECK: vmpsadbw zmm2, zmm3, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x48,0x42,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vmpsadbw zmm2, zmm3, zmmword ptr [2*ebp - 2048], 123

// CHECK: vmpsadbw zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x66,0xcf,0x42,0x51,0x7f,0x7b]
          vmpsadbw zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128], 123

// CHECK: vmpsadbw zmm2 {k7} {z}, zmm3, zmmword ptr [edx - 8192], 123
// CHECK: encoding: [0x62,0xf3,0x66,0xcf,0x42,0x52,0x80,0x7b]
          vmpsadbw zmm2 {k7} {z}, zmm3, zmmword ptr [edx - 8192], 123

// YMM Rounding

// CHECK: vaddpd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0x58,0xd4]
          vaddpd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vaddpd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x3f,0x58,0xd4]
          vaddpd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vaddpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0xff,0x58,0xd4]
          vaddpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vaddph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x18,0x58,0xd4]
          vaddph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vaddph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x3f,0x58,0xd4]
          vaddph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vaddph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0xff,0x58,0xd4]
          vaddph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vaddps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0x58,0xd4]
          vaddps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vaddps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x3f,0x58,0xd4]
          vaddps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vaddps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0xff,0x58,0xd4]
          vaddps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
