// RUN: llvm-mc -triple i686-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK:      vbcstnebf162ps xmm2, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x7a,0xb1,0x94,0xf4,0x00,0x00,0x00,0x10]
               vbcstnebf162ps xmm2, word ptr [esp + 8*esi + 268435456]

// CHECK:      vbcstnebf162ps xmm2, word ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x7a,0xb1,0x94,0x87,0x23,0x01,0x00,0x00]
               vbcstnebf162ps xmm2, word ptr [edi + 4*eax + 291]

// CHECK:      vbcstnebf162ps xmm2, word ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x7a,0xb1,0x10]
               vbcstnebf162ps xmm2, word ptr [eax]

// CHECK:      vbcstnebf162ps xmm2, word ptr [2*ebp - 64]
// CHECK: encoding: [0xc4,0xe2,0x7a,0xb1,0x14,0x6d,0xc0,0xff,0xff,0xff]
               vbcstnebf162ps xmm2, word ptr [2*ebp - 64]

// CHECK:      vbcstnebf162ps xmm2, word ptr [ecx + 254]
// CHECK: encoding: [0xc4,0xe2,0x7a,0xb1,0x91,0xfe,0x00,0x00,0x00]
               vbcstnebf162ps xmm2, word ptr [ecx + 254]

// CHECK:      vbcstnebf162ps xmm2, word ptr [edx - 256]
// CHECK: encoding: [0xc4,0xe2,0x7a,0xb1,0x92,0x00,0xff,0xff,0xff]
               vbcstnebf162ps xmm2, word ptr [edx - 256]

// CHECK:      vbcstnebf162ps ymm2, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x7e,0xb1,0x94,0xf4,0x00,0x00,0x00,0x10]
               vbcstnebf162ps ymm2, word ptr [esp + 8*esi + 268435456]

// CHECK:      vbcstnebf162ps ymm2, word ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x7e,0xb1,0x94,0x87,0x23,0x01,0x00,0x00]
               vbcstnebf162ps ymm2, word ptr [edi + 4*eax + 291]

// CHECK:      vbcstnebf162ps ymm2, word ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x7e,0xb1,0x10]
               vbcstnebf162ps ymm2, word ptr [eax]

// CHECK:      vbcstnebf162ps ymm2, word ptr [2*ebp - 64]
// CHECK: encoding: [0xc4,0xe2,0x7e,0xb1,0x14,0x6d,0xc0,0xff,0xff,0xff]
               vbcstnebf162ps ymm2, word ptr [2*ebp - 64]

// CHECK:      vbcstnebf162ps ymm2, word ptr [ecx + 254]
// CHECK: encoding: [0xc4,0xe2,0x7e,0xb1,0x91,0xfe,0x00,0x00,0x00]
               vbcstnebf162ps ymm2, word ptr [ecx + 254]

// CHECK:      vbcstnebf162ps ymm2, word ptr [edx - 256]
// CHECK: encoding: [0xc4,0xe2,0x7e,0xb1,0x92,0x00,0xff,0xff,0xff]
               vbcstnebf162ps ymm2, word ptr [edx - 256]

// CHECK:      vbcstnesh2ps xmm2, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x79,0xb1,0x94,0xf4,0x00,0x00,0x00,0x10]
               vbcstnesh2ps xmm2, word ptr [esp + 8*esi + 268435456]

// CHECK:      vbcstnesh2ps xmm2, word ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x79,0xb1,0x94,0x87,0x23,0x01,0x00,0x00]
               vbcstnesh2ps xmm2, word ptr [edi + 4*eax + 291]

// CHECK:      vbcstnesh2ps xmm2, word ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x79,0xb1,0x10]
               vbcstnesh2ps xmm2, word ptr [eax]

// CHECK:      vbcstnesh2ps xmm2, word ptr [2*ebp - 64]
// CHECK: encoding: [0xc4,0xe2,0x79,0xb1,0x14,0x6d,0xc0,0xff,0xff,0xff]
               vbcstnesh2ps xmm2, word ptr [2*ebp - 64]

// CHECK:      vbcstnesh2ps xmm2, word ptr [ecx + 254]
// CHECK: encoding: [0xc4,0xe2,0x79,0xb1,0x91,0xfe,0x00,0x00,0x00]
               vbcstnesh2ps xmm2, word ptr [ecx + 254]

// CHECK:      vbcstnesh2ps xmm2, word ptr [edx - 256]
// CHECK: encoding: [0xc4,0xe2,0x79,0xb1,0x92,0x00,0xff,0xff,0xff]
               vbcstnesh2ps xmm2, word ptr [edx - 256]

// CHECK:      vbcstnesh2ps ymm2, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x7d,0xb1,0x94,0xf4,0x00,0x00,0x00,0x10]
               vbcstnesh2ps ymm2, word ptr [esp + 8*esi + 268435456]

// CHECK:      vbcstnesh2ps ymm2, word ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x7d,0xb1,0x94,0x87,0x23,0x01,0x00,0x00]
               vbcstnesh2ps ymm2, word ptr [edi + 4*eax + 291]

// CHECK:      vbcstnesh2ps ymm2, word ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x7d,0xb1,0x10]
               vbcstnesh2ps ymm2, word ptr [eax]

// CHECK:      vbcstnesh2ps ymm2, word ptr [2*ebp - 64]
// CHECK: encoding: [0xc4,0xe2,0x7d,0xb1,0x14,0x6d,0xc0,0xff,0xff,0xff]
               vbcstnesh2ps ymm2, word ptr [2*ebp - 64]

// CHECK:      vbcstnesh2ps ymm2, word ptr [ecx + 254]
// CHECK: encoding: [0xc4,0xe2,0x7d,0xb1,0x91,0xfe,0x00,0x00,0x00]
               vbcstnesh2ps ymm2, word ptr [ecx + 254]

// CHECK:      vbcstnesh2ps ymm2, word ptr [edx - 256]
// CHECK: encoding: [0xc4,0xe2,0x7d,0xb1,0x92,0x00,0xff,0xff,0xff]
               vbcstnesh2ps ymm2, word ptr [edx - 256]

// CHECK:      vcvtneebf162ps xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x7a,0xb0,0x94,0xf4,0x00,0x00,0x00,0x10]
               vcvtneebf162ps xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK:      vcvtneebf162ps xmm2, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x7a,0xb0,0x94,0x87,0x23,0x01,0x00,0x00]
               vcvtneebf162ps xmm2, xmmword ptr [edi + 4*eax + 291]

// CHECK:      vcvtneebf162ps xmm2, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x7a,0xb0,0x10]
               vcvtneebf162ps xmm2, xmmword ptr [eax]

// CHECK:      vcvtneebf162ps xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x7a,0xb0,0x14,0x6d,0x00,0xfe,0xff,0xff]
               vcvtneebf162ps xmm2, xmmword ptr [2*ebp - 512]

// CHECK:      vcvtneebf162ps xmm2, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0xc4,0xe2,0x7a,0xb0,0x91,0xf0,0x07,0x00,0x00]
               vcvtneebf162ps xmm2, xmmword ptr [ecx + 2032]

// CHECK:      vcvtneebf162ps xmm2, xmmword ptr [edx - 2048]
// CHECK: encoding: [0xc4,0xe2,0x7a,0xb0,0x92,0x00,0xf8,0xff,0xff]
               vcvtneebf162ps xmm2, xmmword ptr [edx - 2048]

// CHECK:      vcvtneebf162ps ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x7e,0xb0,0x94,0xf4,0x00,0x00,0x00,0x10]
               vcvtneebf162ps ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK:      vcvtneebf162ps ymm2, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x7e,0xb0,0x94,0x87,0x23,0x01,0x00,0x00]
               vcvtneebf162ps ymm2, ymmword ptr [edi + 4*eax + 291]

// CHECK:      vcvtneebf162ps ymm2, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x7e,0xb0,0x10]
               vcvtneebf162ps ymm2, ymmword ptr [eax]

// CHECK:      vcvtneebf162ps ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x7e,0xb0,0x14,0x6d,0x00,0xfc,0xff,0xff]
               vcvtneebf162ps ymm2, ymmword ptr [2*ebp - 1024]

// CHECK:      vcvtneebf162ps ymm2, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0xc4,0xe2,0x7e,0xb0,0x91,0xe0,0x0f,0x00,0x00]
               vcvtneebf162ps ymm2, ymmword ptr [ecx + 4064]

// CHECK:      vcvtneebf162ps ymm2, ymmword ptr [edx - 4096]
// CHECK: encoding: [0xc4,0xe2,0x7e,0xb0,0x92,0x00,0xf0,0xff,0xff]
               vcvtneebf162ps ymm2, ymmword ptr [edx - 4096]

// CHECK:      vcvtneeph2ps xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x79,0xb0,0x94,0xf4,0x00,0x00,0x00,0x10]
               vcvtneeph2ps xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK:      vcvtneeph2ps xmm2, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x79,0xb0,0x94,0x87,0x23,0x01,0x00,0x00]
               vcvtneeph2ps xmm2, xmmword ptr [edi + 4*eax + 291]

// CHECK:      vcvtneeph2ps xmm2, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x79,0xb0,0x10]
               vcvtneeph2ps xmm2, xmmword ptr [eax]

// CHECK:      vcvtneeph2ps xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x79,0xb0,0x14,0x6d,0x00,0xfe,0xff,0xff]
               vcvtneeph2ps xmm2, xmmword ptr [2*ebp - 512]

// CHECK:      vcvtneeph2ps xmm2, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0xc4,0xe2,0x79,0xb0,0x91,0xf0,0x07,0x00,0x00]
               vcvtneeph2ps xmm2, xmmword ptr [ecx + 2032]

// CHECK:      vcvtneeph2ps xmm2, xmmword ptr [edx - 2048]
// CHECK: encoding: [0xc4,0xe2,0x79,0xb0,0x92,0x00,0xf8,0xff,0xff]
               vcvtneeph2ps xmm2, xmmword ptr [edx - 2048]

// CHECK:      vcvtneeph2ps ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x7d,0xb0,0x94,0xf4,0x00,0x00,0x00,0x10]
               vcvtneeph2ps ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK:      vcvtneeph2ps ymm2, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x7d,0xb0,0x94,0x87,0x23,0x01,0x00,0x00]
               vcvtneeph2ps ymm2, ymmword ptr [edi + 4*eax + 291]

// CHECK:      vcvtneeph2ps ymm2, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x7d,0xb0,0x10]
               vcvtneeph2ps ymm2, ymmword ptr [eax]

// CHECK:      vcvtneeph2ps ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x7d,0xb0,0x14,0x6d,0x00,0xfc,0xff,0xff]
               vcvtneeph2ps ymm2, ymmword ptr [2*ebp - 1024]

// CHECK:      vcvtneeph2ps ymm2, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0xc4,0xe2,0x7d,0xb0,0x91,0xe0,0x0f,0x00,0x00]
               vcvtneeph2ps ymm2, ymmword ptr [ecx + 4064]

// CHECK:      vcvtneeph2ps ymm2, ymmword ptr [edx - 4096]
// CHECK: encoding: [0xc4,0xe2,0x7d,0xb0,0x92,0x00,0xf0,0xff,0xff]
               vcvtneeph2ps ymm2, ymmword ptr [edx - 4096]

// CHECK:      vcvtneobf162ps xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x7b,0xb0,0x94,0xf4,0x00,0x00,0x00,0x10]
               vcvtneobf162ps xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK:      vcvtneobf162ps xmm2, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x7b,0xb0,0x94,0x87,0x23,0x01,0x00,0x00]
               vcvtneobf162ps xmm2, xmmword ptr [edi + 4*eax + 291]

// CHECK:      vcvtneobf162ps xmm2, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x7b,0xb0,0x10]
               vcvtneobf162ps xmm2, xmmword ptr [eax]

// CHECK:      vcvtneobf162ps xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x7b,0xb0,0x14,0x6d,0x00,0xfe,0xff,0xff]
               vcvtneobf162ps xmm2, xmmword ptr [2*ebp - 512]

// CHECK:      vcvtneobf162ps xmm2, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0xc4,0xe2,0x7b,0xb0,0x91,0xf0,0x07,0x00,0x00]
               vcvtneobf162ps xmm2, xmmword ptr [ecx + 2032]

// CHECK:      vcvtneobf162ps xmm2, xmmword ptr [edx - 2048]
// CHECK: encoding: [0xc4,0xe2,0x7b,0xb0,0x92,0x00,0xf8,0xff,0xff]
               vcvtneobf162ps xmm2, xmmword ptr [edx - 2048]

// CHECK:      vcvtneobf162ps ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x7f,0xb0,0x94,0xf4,0x00,0x00,0x00,0x10]
               vcvtneobf162ps ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK:      vcvtneobf162ps ymm2, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x7f,0xb0,0x94,0x87,0x23,0x01,0x00,0x00]
               vcvtneobf162ps ymm2, ymmword ptr [edi + 4*eax + 291]

// CHECK:      vcvtneobf162ps ymm2, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x7f,0xb0,0x10]
               vcvtneobf162ps ymm2, ymmword ptr [eax]

// CHECK:      vcvtneobf162ps ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x7f,0xb0,0x14,0x6d,0x00,0xfc,0xff,0xff]
               vcvtneobf162ps ymm2, ymmword ptr [2*ebp - 1024]

// CHECK:      vcvtneobf162ps ymm2, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0xc4,0xe2,0x7f,0xb0,0x91,0xe0,0x0f,0x00,0x00]
               vcvtneobf162ps ymm2, ymmword ptr [ecx + 4064]

// CHECK:      vcvtneobf162ps ymm2, ymmword ptr [edx - 4096]
// CHECK: encoding: [0xc4,0xe2,0x7f,0xb0,0x92,0x00,0xf0,0xff,0xff]
               vcvtneobf162ps ymm2, ymmword ptr [edx - 4096]

// CHECK:      vcvtneoph2ps xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x78,0xb0,0x94,0xf4,0x00,0x00,0x00,0x10]
               vcvtneoph2ps xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK:      vcvtneoph2ps xmm2, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x78,0xb0,0x94,0x87,0x23,0x01,0x00,0x00]
               vcvtneoph2ps xmm2, xmmword ptr [edi + 4*eax + 291]

// CHECK:      vcvtneoph2ps xmm2, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x78,0xb0,0x10]
               vcvtneoph2ps xmm2, xmmword ptr [eax]

// CHECK:      vcvtneoph2ps xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x78,0xb0,0x14,0x6d,0x00,0xfe,0xff,0xff]
               vcvtneoph2ps xmm2, xmmword ptr [2*ebp - 512]

// CHECK:      vcvtneoph2ps xmm2, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0xc4,0xe2,0x78,0xb0,0x91,0xf0,0x07,0x00,0x00]
               vcvtneoph2ps xmm2, xmmword ptr [ecx + 2032]

// CHECK:      vcvtneoph2ps xmm2, xmmword ptr [edx - 2048]
// CHECK: encoding: [0xc4,0xe2,0x78,0xb0,0x92,0x00,0xf8,0xff,0xff]
               vcvtneoph2ps xmm2, xmmword ptr [edx - 2048]

// CHECK:      vcvtneoph2ps ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x7c,0xb0,0x94,0xf4,0x00,0x00,0x00,0x10]
               vcvtneoph2ps ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK:      vcvtneoph2ps ymm2, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x7c,0xb0,0x94,0x87,0x23,0x01,0x00,0x00]
               vcvtneoph2ps ymm2, ymmword ptr [edi + 4*eax + 291]

// CHECK:      vcvtneoph2ps ymm2, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x7c,0xb0,0x10]
               vcvtneoph2ps ymm2, ymmword ptr [eax]

// CHECK:      vcvtneoph2ps ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x7c,0xb0,0x14,0x6d,0x00,0xfc,0xff,0xff]
               vcvtneoph2ps ymm2, ymmword ptr [2*ebp - 1024]

// CHECK:      vcvtneoph2ps ymm2, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0xc4,0xe2,0x7c,0xb0,0x91,0xe0,0x0f,0x00,0x00]
               vcvtneoph2ps ymm2, ymmword ptr [ecx + 4064]

// CHECK:      vcvtneoph2ps ymm2, ymmword ptr [edx - 4096]
// CHECK: encoding: [0xc4,0xe2,0x7c,0xb0,0x92,0x00,0xf0,0xff,0xff]
               vcvtneoph2ps ymm2, ymmword ptr [edx - 4096]

// CHECK:      {vex} vcvtneps2bf16 xmm2, xmm3
// CHECK: encoding: [0xc4,0xe2,0x7a,0x72,0xd3]
               {vex} vcvtneps2bf16 xmm2, xmm3

// CHECK:      {vex} vcvtneps2bf16 xmm2, ymm3
// CHECK: encoding: [0xc4,0xe2,0x7e,0x72,0xd3]
               {vex} vcvtneps2bf16 xmm2, ymm3

// CHECK:      {vex} vcvtneps2bf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x7a,0x72,0x94,0xf4,0x00,0x00,0x00,0x10]
               {vex} vcvtneps2bf16 xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK:      {vex} vcvtneps2bf16 xmm2, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x7a,0x72,0x94,0x87,0x23,0x01,0x00,0x00]
               {vex} vcvtneps2bf16 xmm2, xmmword ptr [edi + 4*eax + 291]

// CHECK:      {vex} vcvtneps2bf16 xmm2, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x7a,0x72,0x10]
               {vex} vcvtneps2bf16 xmm2, xmmword ptr [eax]

// CHECK:      {vex} vcvtneps2bf16 xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x7a,0x72,0x14,0x6d,0x00,0xfe,0xff,0xff]
               {vex} vcvtneps2bf16 xmm2, xmmword ptr [2*ebp - 512]

// CHECK:      {vex} vcvtneps2bf16 xmm2, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0xc4,0xe2,0x7a,0x72,0x91,0xf0,0x07,0x00,0x00]
               {vex} vcvtneps2bf16 xmm2, xmmword ptr [ecx + 2032]

// CHECK:      {vex} vcvtneps2bf16 xmm2, xmmword ptr [edx - 2048]
// CHECK: encoding: [0xc4,0xe2,0x7a,0x72,0x92,0x00,0xf8,0xff,0xff]
               {vex} vcvtneps2bf16 xmm2, xmmword ptr [edx - 2048]

// CHECK:      {vex} vcvtneps2bf16 xmm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x7e,0x72,0x14,0x6d,0x00,0xfc,0xff,0xff]
               {vex} vcvtneps2bf16 xmm2, ymmword ptr [2*ebp - 1024]

// CHECK:      {vex} vcvtneps2bf16 xmm2, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0xc4,0xe2,0x7e,0x72,0x91,0xe0,0x0f,0x00,0x00]
               {vex} vcvtneps2bf16 xmm2, ymmword ptr [ecx + 4064]

// CHECK:      {vex} vcvtneps2bf16 xmm2, ymmword ptr [edx - 4096]
// CHECK: encoding: [0xc4,0xe2,0x7e,0x72,0x92,0x00,0xf0,0xff,0xff]
               {vex} vcvtneps2bf16 xmm2, ymmword ptr [edx - 4096]

