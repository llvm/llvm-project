// RUN: llvm-mc -triple i686-unknown-unknown -mattr=+avxvnniint8 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vpdpbssd ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x67,0x50,0xd4]
     vpdpbssd ymm2, ymm3, ymm4

// CHECK: vpdpbssd xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x63,0x50,0xd4]
     vpdpbssd xmm2, xmm3, xmm4

// CHECK: vpdpbssd ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x67,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
     vpdpbssd ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbssd ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x67,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
     vpdpbssd ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbssd ymm2, ymm3, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x67,0x50,0x10]
     vpdpbssd ymm2, ymm3, ymmword ptr [eax]

// CHECK: vpdpbssd ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x67,0x50,0x14,0x6d,0x00,0xfc,0xff,0xff]
     vpdpbssd ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpbssd xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x63,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
     vpdpbssd xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbssd xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x63,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
     vpdpbssd xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbssd xmm2, xmm3, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x63,0x50,0x10]
     vpdpbssd xmm2, xmm3, xmmword ptr [eax]

// CHECK: vpdpbssd xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x63,0x50,0x14,0x6d,0x00,0xfe,0xff,0xff]
     vpdpbssd xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpbssds ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x67,0x51,0xd4]
     vpdpbssds ymm2, ymm3, ymm4

// CHECK: vpdpbssds xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x63,0x51,0xd4]
     vpdpbssds xmm2, xmm3, xmm4

// CHECK: vpdpbssds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x67,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
     vpdpbssds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbssds ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x67,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
     vpdpbssds ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbssds ymm2, ymm3, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x67,0x51,0x10]
     vpdpbssds ymm2, ymm3, ymmword ptr [eax]

// CHECK: vpdpbssds ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x67,0x51,0x14,0x6d,0x00,0xfc,0xff,0xff]
     vpdpbssds ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpbssds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x63,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
     vpdpbssds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbssds xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x63,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
     vpdpbssds xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbssds xmm2, xmm3, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x63,0x51,0x10]
     vpdpbssds xmm2, xmm3, xmmword ptr [eax]

// CHECK: vpdpbssds xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x63,0x51,0x14,0x6d,0x00,0xfe,0xff,0xff]
     vpdpbssds xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpbsud ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x66,0x50,0xd4]
     vpdpbsud ymm2, ymm3, ymm4

// CHECK: vpdpbsud xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x62,0x50,0xd4]
     vpdpbsud xmm2, xmm3, xmm4

// CHECK: vpdpbsud ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x66,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
     vpdpbsud ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbsud ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x66,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
     vpdpbsud ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbsud ymm2, ymm3, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x66,0x50,0x10]
     vpdpbsud ymm2, ymm3, ymmword ptr [eax]

// CHECK: vpdpbsud ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x66,0x50,0x14,0x6d,0x00,0xfc,0xff,0xff]
     vpdpbsud ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpbsud xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x62,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
     vpdpbsud xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbsud xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x62,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
     vpdpbsud xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbsud xmm2, xmm3, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x62,0x50,0x10]
     vpdpbsud xmm2, xmm3, xmmword ptr [eax]

// CHECK: vpdpbsud xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x62,0x50,0x14,0x6d,0x00,0xfe,0xff,0xff]
     vpdpbsud xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpbsuds ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x66,0x51,0xd4]
     vpdpbsuds ymm2, ymm3, ymm4

// CHECK: vpdpbsuds xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x62,0x51,0xd4]
     vpdpbsuds xmm2, xmm3, xmm4

// CHECK: vpdpbsuds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x66,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
     vpdpbsuds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbsuds ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x66,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
     vpdpbsuds ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbsuds ymm2, ymm3, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x66,0x51,0x10]
     vpdpbsuds ymm2, ymm3, ymmword ptr [eax]

// CHECK: vpdpbsuds ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x66,0x51,0x14,0x6d,0x00,0xfc,0xff,0xff]
     vpdpbsuds ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpbsuds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x62,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
     vpdpbsuds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbsuds xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x62,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
     vpdpbsuds xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbsuds xmm2, xmm3, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x62,0x51,0x10]
     vpdpbsuds xmm2, xmm3, xmmword ptr [eax]

// CHECK: vpdpbsuds xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x62,0x51,0x14,0x6d,0x00,0xfe,0xff,0xff]
     vpdpbsuds xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpbuud ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x64,0x50,0xd4]
     vpdpbuud ymm2, ymm3, ymm4

// CHECK: vpdpbuud xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x60,0x50,0xd4]
     vpdpbuud xmm2, xmm3, xmm4

// CHECK: vpdpbuud ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x64,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
     vpdpbuud ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbuud ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x64,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
     vpdpbuud ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbuud ymm2, ymm3, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x64,0x50,0x10]
     vpdpbuud ymm2, ymm3, ymmword ptr [eax]

// CHECK: vpdpbuud ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x64,0x50,0x14,0x6d,0x00,0xfc,0xff,0xff]
     vpdpbuud ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpbuud xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x60,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
     vpdpbuud xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbuud xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x60,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
     vpdpbuud xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbuud xmm2, xmm3, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x60,0x50,0x10]
     vpdpbuud xmm2, xmm3, xmmword ptr [eax]

// CHECK: vpdpbuud xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x60,0x50,0x14,0x6d,0x00,0xfe,0xff,0xff]
     vpdpbuud xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpbuuds ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x64,0x51,0xd4]
     vpdpbuuds ymm2, ymm3, ymm4

// CHECK: vpdpbuuds xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x60,0x51,0xd4]
     vpdpbuuds xmm2, xmm3, xmm4

// CHECK: vpdpbuuds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x64,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
     vpdpbuuds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbuuds ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x64,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
     vpdpbuuds ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbuuds ymm2, ymm3, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x64,0x51,0x10]
     vpdpbuuds ymm2, ymm3, ymmword ptr [eax]

// CHECK: vpdpbuuds ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x64,0x51,0x14,0x6d,0x00,0xfc,0xff,0xff]
     vpdpbuuds ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpbuuds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x60,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
     vpdpbuuds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbuuds xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x60,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
     vpdpbuuds xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbuuds xmm2, xmm3, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x60,0x51,0x10]
     vpdpbuuds xmm2, xmm3, xmmword ptr [eax]

// CHECK: vpdpbuuds xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x60,0x51,0x14,0x6d,0x00,0xfe,0xff,0xff]
     vpdpbuuds xmm2, xmm3, xmmword ptr [2*ebp - 512]

