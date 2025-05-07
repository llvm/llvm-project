// RUN: llvm-mc -triple i386 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// VNNI FP16

// CHECK: vdpphps xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x08,0x52,0xd4]
          vdpphps xmm2, xmm3, xmm4

// CHECK: vdpphps xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0x52,0xd4]
          vdpphps xmm2 {k7}, xmm3, xmm4

// CHECK: vdpphps xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0x52,0xd4]
          vdpphps xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vdpphps ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0x28,0x52,0xd4]
          vdpphps ymm2, ymm3, ymm4

// CHECK: vdpphps ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0x52,0xd4]
          vdpphps ymm2 {k7}, ymm3, ymm4

// CHECK: vdpphps ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0x52,0xd4]
          vdpphps ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vdpphps zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x52,0xd4]
          vdpphps zmm2, zmm3, zmm4

// CHECK: vdpphps zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0x52,0xd4]
          vdpphps zmm2 {k7}, zmm3, zmm4

// CHECK: vdpphps zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0x52,0xd4]
          vdpphps zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vdpphps xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x64,0x08,0x52,0x94,0xf4,0x00,0x00,0x00,0x10]
          vdpphps xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vdpphps xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0x52,0x94,0x87,0x23,0x01,0x00,0x00]
          vdpphps xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vdpphps xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x64,0x18,0x52,0x10]
          vdpphps xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vdpphps xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf2,0x64,0x08,0x52,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vdpphps xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vdpphps xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0x52,0x51,0x7f]
          vdpphps xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vdpphps xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x64,0x9f,0x52,0x52,0x80]
          vdpphps xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vdpphps ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x64,0x28,0x52,0x94,0xf4,0x00,0x00,0x00,0x10]
          vdpphps ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vdpphps ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0x52,0x94,0x87,0x23,0x01,0x00,0x00]
          vdpphps ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vdpphps ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x64,0x38,0x52,0x10]
          vdpphps ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vdpphps ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf2,0x64,0x28,0x52,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vdpphps ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vdpphps ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0x52,0x51,0x7f]
          vdpphps ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vdpphps ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x64,0xbf,0x52,0x52,0x80]
          vdpphps ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vdpphps zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x52,0x94,0xf4,0x00,0x00,0x00,0x10]
          vdpphps zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vdpphps zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0x52,0x94,0x87,0x23,0x01,0x00,0x00]
          vdpphps zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vdpphps zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x64,0x58,0x52,0x10]
          vdpphps zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vdpphps zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x52,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vdpphps zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vdpphps zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0x52,0x51,0x7f]
          vdpphps zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vdpphps zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x64,0xdf,0x52,0x52,0x80]
          vdpphps zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

// VNNI INT8

// CHECK: vpdpbssd xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x63,0x50,0xd4]
          vpdpbssd xmm2, xmm3, xmm4

// CHECK: vpdpbssd xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x0f,0x50,0xd4]
          vpdpbssd xmm2 {k7}, xmm3, xmm4

// CHECK: vpdpbssd xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x8f,0x50,0xd4]
          vpdpbssd xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vpdpbssd ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x67,0x50,0xd4]
          vpdpbssd ymm2, ymm3, ymm4

// CHECK: vpdpbssd ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x67,0x2f,0x50,0xd4]
          vpdpbssd ymm2 {k7}, ymm3, ymm4

// CHECK: vpdpbssd ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x67,0xaf,0x50,0xd4]
          vpdpbssd ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vpdpbssd zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x50,0xd4]
          vpdpbssd zmm2, zmm3, zmm4

// CHECK: vpdpbssd zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x4f,0x50,0xd4]
          vpdpbssd zmm2 {k7}, zmm3, zmm4

// CHECK: vpdpbssd zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x67,0xcf,0x50,0xd4]
          vpdpbssd zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vpdpbssd xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x63,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbssd xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbssd xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x0f,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbssd xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbssd xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x67,0x18,0x50,0x10]
          vpdpbssd xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vpdpbssd xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x63,0x50,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vpdpbssd xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpbssd xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x67,0x8f,0x50,0x51,0x7f]
          vpdpbssd xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vpdpbssd xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x67,0x9f,0x50,0x52,0x80]
          vpdpbssd xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vpdpbssd ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x67,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbssd ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbssd ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x2f,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbssd ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbssd ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x67,0x38,0x50,0x10]
          vpdpbssd ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vpdpbssd ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x67,0x50,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vpdpbssd ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpbssd ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x67,0xaf,0x50,0x51,0x7f]
          vpdpbssd ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vpdpbssd ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x67,0xbf,0x50,0x52,0x80]
          vpdpbssd ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vpdpbssd zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbssd zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbssd zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x4f,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbssd zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbssd zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x67,0x58,0x50,0x10]
          vpdpbssd zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vpdpbssd zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x50,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vpdpbssd zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vpdpbssd zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x67,0xcf,0x50,0x51,0x7f]
          vpdpbssd zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vpdpbssd zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x67,0xdf,0x50,0x52,0x80]
          vpdpbssd zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

// CHECK: vpdpbssds xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x63,0x51,0xd4]
          vpdpbssds xmm2, xmm3, xmm4

// CHECK: vpdpbssds xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x0f,0x51,0xd4]
          vpdpbssds xmm2 {k7}, xmm3, xmm4

// CHECK: vpdpbssds xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x8f,0x51,0xd4]
          vpdpbssds xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vpdpbssds ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x67,0x51,0xd4]
          vpdpbssds ymm2, ymm3, ymm4

// CHECK: vpdpbssds ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x67,0x2f,0x51,0xd4]
          vpdpbssds ymm2 {k7}, ymm3, ymm4

// CHECK: vpdpbssds ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x67,0xaf,0x51,0xd4]
          vpdpbssds ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vpdpbssds zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x51,0xd4]
          vpdpbssds zmm2, zmm3, zmm4

// CHECK: vpdpbssds zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x4f,0x51,0xd4]
          vpdpbssds zmm2 {k7}, zmm3, zmm4

// CHECK: vpdpbssds zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x67,0xcf,0x51,0xd4]
          vpdpbssds zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vpdpbssds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x63,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbssds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbssds xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x0f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbssds xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbssds xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x67,0x18,0x51,0x10]
          vpdpbssds xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vpdpbssds xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x63,0x51,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vpdpbssds xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpbssds xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x67,0x8f,0x51,0x51,0x7f]
          vpdpbssds xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vpdpbssds xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x67,0x9f,0x51,0x52,0x80]
          vpdpbssds xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vpdpbssds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x67,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbssds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbssds ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x2f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbssds ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbssds ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x67,0x38,0x51,0x10]
          vpdpbssds ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vpdpbssds ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x67,0x51,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vpdpbssds ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpbssds ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x67,0xaf,0x51,0x51,0x7f]
          vpdpbssds ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vpdpbssds ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x67,0xbf,0x51,0x52,0x80]
          vpdpbssds ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vpdpbssds zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbssds zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbssds zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x4f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbssds zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbssds zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x67,0x58,0x51,0x10]
          vpdpbssds zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vpdpbssds zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x51,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vpdpbssds zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vpdpbssds zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x67,0xcf,0x51,0x51,0x7f]
          vpdpbssds zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vpdpbssds zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x67,0xdf,0x51,0x52,0x80]
          vpdpbssds zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

// CHECK: vpdpbsud xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x62,0x50,0xd4]
          vpdpbsud xmm2, xmm3, xmm4

// CHECK: vpdpbsud xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x0f,0x50,0xd4]
          vpdpbsud xmm2 {k7}, xmm3, xmm4

// CHECK: vpdpbsud xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x8f,0x50,0xd4]
          vpdpbsud xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vpdpbsud ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x66,0x50,0xd4]
          vpdpbsud ymm2, ymm3, ymm4

// CHECK: vpdpbsud ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x66,0x2f,0x50,0xd4]
          vpdpbsud ymm2 {k7}, ymm3, ymm4

// CHECK: vpdpbsud ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x66,0xaf,0x50,0xd4]
          vpdpbsud ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vpdpbsud zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0x50,0xd4]
          vpdpbsud zmm2, zmm3, zmm4

// CHECK: vpdpbsud zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x4f,0x50,0xd4]
          vpdpbsud zmm2 {k7}, zmm3, zmm4

// CHECK: vpdpbsud zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x66,0xcf,0x50,0xd4]
          vpdpbsud zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vpdpbsud xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x62,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbsud xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbsud xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x0f,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbsud xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbsud xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x66,0x18,0x50,0x10]
          vpdpbsud xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vpdpbsud xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x62,0x50,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vpdpbsud xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpbsud xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x66,0x8f,0x50,0x51,0x7f]
          vpdpbsud xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vpdpbsud xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x66,0x9f,0x50,0x52,0x80]
          vpdpbsud xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vpdpbsud ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x66,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbsud ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbsud ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x2f,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbsud ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbsud ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x66,0x38,0x50,0x10]
          vpdpbsud ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vpdpbsud ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x66,0x50,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vpdpbsud ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpbsud ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x66,0xaf,0x50,0x51,0x7f]
          vpdpbsud ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vpdpbsud ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x66,0xbf,0x50,0x52,0x80]
          vpdpbsud ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vpdpbsud zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbsud zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbsud zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x4f,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbsud zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbsud zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x66,0x58,0x50,0x10]
          vpdpbsud zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vpdpbsud zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0x50,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vpdpbsud zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vpdpbsud zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x66,0xcf,0x50,0x51,0x7f]
          vpdpbsud zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vpdpbsud zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x66,0xdf,0x50,0x52,0x80]
          vpdpbsud zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

// CHECK: vpdpbsuds xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x62,0x51,0xd4]
          vpdpbsuds xmm2, xmm3, xmm4

// CHECK: vpdpbsuds xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x0f,0x51,0xd4]
          vpdpbsuds xmm2 {k7}, xmm3, xmm4

// CHECK: vpdpbsuds xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x8f,0x51,0xd4]
          vpdpbsuds xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vpdpbsuds ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x66,0x51,0xd4]
          vpdpbsuds ymm2, ymm3, ymm4

// CHECK: vpdpbsuds ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x66,0x2f,0x51,0xd4]
          vpdpbsuds ymm2 {k7}, ymm3, ymm4

// CHECK: vpdpbsuds ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x66,0xaf,0x51,0xd4]
          vpdpbsuds ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vpdpbsuds zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0x51,0xd4]
          vpdpbsuds zmm2, zmm3, zmm4

// CHECK: vpdpbsuds zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x4f,0x51,0xd4]
          vpdpbsuds zmm2 {k7}, zmm3, zmm4

// CHECK: vpdpbsuds zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x66,0xcf,0x51,0xd4]
          vpdpbsuds zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vpdpbsuds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x62,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbsuds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbsuds xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x0f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbsuds xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbsuds xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x66,0x18,0x51,0x10]
          vpdpbsuds xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vpdpbsuds xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x62,0x51,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vpdpbsuds xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpbsuds xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x66,0x8f,0x51,0x51,0x7f]
          vpdpbsuds xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vpdpbsuds xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x66,0x9f,0x51,0x52,0x80]
          vpdpbsuds xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vpdpbsuds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x66,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbsuds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbsuds ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x2f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbsuds ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbsuds ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x66,0x38,0x51,0x10]
          vpdpbsuds ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vpdpbsuds ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x66,0x51,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vpdpbsuds ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpbsuds ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x66,0xaf,0x51,0x51,0x7f]
          vpdpbsuds ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vpdpbsuds ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x66,0xbf,0x51,0x52,0x80]
          vpdpbsuds ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vpdpbsuds zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbsuds zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbsuds zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x4f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbsuds zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbsuds zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x66,0x58,0x51,0x10]
          vpdpbsuds zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vpdpbsuds zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0x51,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vpdpbsuds zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vpdpbsuds zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x66,0xcf,0x51,0x51,0x7f]
          vpdpbsuds zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vpdpbsuds zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x66,0xdf,0x51,0x52,0x80]
          vpdpbsuds zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

// CHECK: vpdpbuud xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x60,0x50,0xd4]
          vpdpbuud xmm2, xmm3, xmm4

// CHECK: vpdpbuud xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0x50,0xd4]
          vpdpbuud xmm2 {k7}, xmm3, xmm4

// CHECK: vpdpbuud xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0x50,0xd4]
          vpdpbuud xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vpdpbuud ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x64,0x50,0xd4]
          vpdpbuud ymm2, ymm3, ymm4

// CHECK: vpdpbuud ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0x50,0xd4]
          vpdpbuud ymm2 {k7}, ymm3, ymm4

// CHECK: vpdpbuud ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0x50,0xd4]
          vpdpbuud ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vpdpbuud zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x50,0xd4]
          vpdpbuud zmm2, zmm3, zmm4

// CHECK: vpdpbuud zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0x50,0xd4]
          vpdpbuud zmm2 {k7}, zmm3, zmm4

// CHECK: vpdpbuud zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0x50,0xd4]
          vpdpbuud zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vpdpbuud xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x60,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbuud xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbuud xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbuud xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbuud xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x64,0x18,0x50,0x10]
          vpdpbuud xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vpdpbuud xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x60,0x50,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vpdpbuud xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpbuud xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0x50,0x51,0x7f]
          vpdpbuud xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vpdpbuud xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x64,0x9f,0x50,0x52,0x80]
          vpdpbuud xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vpdpbuud ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x64,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbuud ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbuud ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbuud ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbuud ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x64,0x38,0x50,0x10]
          vpdpbuud ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vpdpbuud ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x64,0x50,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vpdpbuud ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpbuud ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0x50,0x51,0x7f]
          vpdpbuud ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vpdpbuud ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x64,0xbf,0x50,0x52,0x80]
          vpdpbuud ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vpdpbuud zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x50,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbuud zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbuud zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0x50,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbuud zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbuud zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x64,0x58,0x50,0x10]
          vpdpbuud zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vpdpbuud zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x50,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vpdpbuud zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vpdpbuud zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0x50,0x51,0x7f]
          vpdpbuud zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vpdpbuud zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x64,0xdf,0x50,0x52,0x80]
          vpdpbuud zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

// CHECK: vpdpbuuds xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x60,0x51,0xd4]
          vpdpbuuds xmm2, xmm3, xmm4

// CHECK: vpdpbuuds xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0x51,0xd4]
          vpdpbuuds xmm2 {k7}, xmm3, xmm4

// CHECK: vpdpbuuds xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0x51,0xd4]
          vpdpbuuds xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vpdpbuuds ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x64,0x51,0xd4]
          vpdpbuuds ymm2, ymm3, ymm4

// CHECK: vpdpbuuds ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0x51,0xd4]
          vpdpbuuds ymm2 {k7}, ymm3, ymm4

// CHECK: vpdpbuuds ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0x51,0xd4]
          vpdpbuuds ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vpdpbuuds zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x51,0xd4]
          vpdpbuuds zmm2, zmm3, zmm4

// CHECK: vpdpbuuds zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0x51,0xd4]
          vpdpbuuds zmm2 {k7}, zmm3, zmm4

// CHECK: vpdpbuuds zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0x51,0xd4]
          vpdpbuuds zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vpdpbuuds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x60,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbuuds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbuuds xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbuuds xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbuuds xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x64,0x18,0x51,0x10]
          vpdpbuuds xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vpdpbuuds xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x60,0x51,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vpdpbuuds xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpbuuds xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0x51,0x51,0x7f]
          vpdpbuuds xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vpdpbuuds xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x64,0x9f,0x51,0x52,0x80]
          vpdpbuuds xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vpdpbuuds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x64,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbuuds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbuuds ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbuuds ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbuuds ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x64,0x38,0x51,0x10]
          vpdpbuuds ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vpdpbuuds ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x64,0x51,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vpdpbuuds ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpbuuds ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0x51,0x51,0x7f]
          vpdpbuuds ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vpdpbuuds ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x64,0xbf,0x51,0x52,0x80]
          vpdpbuuds ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vpdpbuuds zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpbuuds zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpbuuds zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpbuuds zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpbuuds zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x64,0x58,0x51,0x10]
          vpdpbuuds zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vpdpbuuds zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x51,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vpdpbuuds zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vpdpbuuds zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0x51,0x51,0x7f]
          vpdpbuuds zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vpdpbuuds zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x64,0xdf,0x51,0x52,0x80]
          vpdpbuuds zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

// VNNI INT16

// CHECK: vpdpwsud xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x62,0xd2,0xd4]
          vpdpwsud xmm2, xmm3, xmm4

// CHECK: vpdpwsud xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x0f,0xd2,0xd4]
          vpdpwsud xmm2 {k7}, xmm3, xmm4

// CHECK: vpdpwsud xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x8f,0xd2,0xd4]
          vpdpwsud xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vpdpwsud ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x66,0xd2,0xd4]
          vpdpwsud ymm2, ymm3, ymm4

// CHECK: vpdpwsud ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x66,0x2f,0xd2,0xd4]
          vpdpwsud ymm2 {k7}, ymm3, ymm4

// CHECK: vpdpwsud ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x66,0xaf,0xd2,0xd4]
          vpdpwsud ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vpdpwsud zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xd2,0xd4]
          vpdpwsud zmm2, zmm3, zmm4

// CHECK: vpdpwsud zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x4f,0xd2,0xd4]
          vpdpwsud zmm2 {k7}, zmm3, zmm4

// CHECK: vpdpwsud zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x66,0xcf,0xd2,0xd4]
          vpdpwsud zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vpdpwsud xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x62,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwsud xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwsud xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x0f,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwsud xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwsud xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x66,0x18,0xd2,0x10]
          vpdpwsud xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vpdpwsud xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x62,0xd2,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwsud xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpwsud xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x66,0x8f,0xd2,0x51,0x7f]
          vpdpwsud xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vpdpwsud xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x66,0x9f,0xd2,0x52,0x80]
          vpdpwsud xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vpdpwsud ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x66,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwsud ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwsud ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x2f,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwsud ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwsud ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x66,0x38,0xd2,0x10]
          vpdpwsud ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vpdpwsud ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x66,0xd2,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwsud ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpwsud ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x66,0xaf,0xd2,0x51,0x7f]
          vpdpwsud ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vpdpwsud ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x66,0xbf,0xd2,0x52,0x80]
          vpdpwsud ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vpdpwsud zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwsud zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwsud zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x4f,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwsud zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwsud zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x66,0x58,0xd2,0x10]
          vpdpwsud zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vpdpwsud zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xd2,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vpdpwsud zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vpdpwsud zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x66,0xcf,0xd2,0x51,0x7f]
          vpdpwsud zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vpdpwsud zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x66,0xdf,0xd2,0x52,0x80]
          vpdpwsud zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

// CHECK: vpdpwsuds xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x62,0xd3,0xd4]
          vpdpwsuds xmm2, xmm3, xmm4

// CHECK: vpdpwsuds xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x0f,0xd3,0xd4]
          vpdpwsuds xmm2 {k7}, xmm3, xmm4

// CHECK: vpdpwsuds xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x8f,0xd3,0xd4]
          vpdpwsuds xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vpdpwsuds ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x66,0xd3,0xd4]
          vpdpwsuds ymm2, ymm3, ymm4

// CHECK: vpdpwsuds ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x66,0x2f,0xd3,0xd4]
          vpdpwsuds ymm2 {k7}, ymm3, ymm4

// CHECK: vpdpwsuds ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x66,0xaf,0xd3,0xd4]
          vpdpwsuds ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vpdpwsuds zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xd3,0xd4]
          vpdpwsuds zmm2, zmm3, zmm4

// CHECK: vpdpwsuds zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x4f,0xd3,0xd4]
          vpdpwsuds zmm2 {k7}, zmm3, zmm4

// CHECK: vpdpwsuds zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x66,0xcf,0xd3,0xd4]
          vpdpwsuds zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vpdpwsuds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x62,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwsuds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwsuds xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x0f,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwsuds xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwsuds xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x66,0x18,0xd3,0x10]
          vpdpwsuds xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vpdpwsuds xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x62,0xd3,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwsuds xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpwsuds xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x66,0x8f,0xd3,0x51,0x7f]
          vpdpwsuds xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vpdpwsuds xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x66,0x9f,0xd3,0x52,0x80]
          vpdpwsuds xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vpdpwsuds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x66,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwsuds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwsuds ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x2f,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwsuds ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwsuds ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x66,0x38,0xd3,0x10]
          vpdpwsuds ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vpdpwsuds ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x66,0xd3,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwsuds ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpwsuds ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x66,0xaf,0xd3,0x51,0x7f]
          vpdpwsuds ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vpdpwsuds ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x66,0xbf,0xd3,0x52,0x80]
          vpdpwsuds ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vpdpwsuds zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwsuds zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwsuds zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x4f,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwsuds zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwsuds zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x66,0x58,0xd3,0x10]
          vpdpwsuds zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vpdpwsuds zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xd3,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vpdpwsuds zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vpdpwsuds zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x66,0xcf,0xd3,0x51,0x7f]
          vpdpwsuds zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vpdpwsuds zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x66,0xdf,0xd3,0x52,0x80]
          vpdpwsuds zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

// CHECK: vpdpwusd xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x61,0xd2,0xd4]
          vpdpwusd xmm2, xmm3, xmm4

// CHECK: vpdpwusd xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x65,0x0f,0xd2,0xd4]
          vpdpwusd xmm2 {k7}, xmm3, xmm4

// CHECK: vpdpwusd xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x65,0x8f,0xd2,0xd4]
          vpdpwusd xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vpdpwusd ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x65,0xd2,0xd4]
          vpdpwusd ymm2, ymm3, ymm4

// CHECK: vpdpwusd ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x65,0x2f,0xd2,0xd4]
          vpdpwusd ymm2 {k7}, ymm3, ymm4

// CHECK: vpdpwusd ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x65,0xaf,0xd2,0xd4]
          vpdpwusd ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vpdpwusd zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0xd2,0xd4]
          vpdpwusd zmm2, zmm3, zmm4

// CHECK: vpdpwusd zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x65,0x4f,0xd2,0xd4]
          vpdpwusd zmm2 {k7}, zmm3, zmm4

// CHECK: vpdpwusd zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x65,0xcf,0xd2,0xd4]
          vpdpwusd zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vpdpwusd xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x61,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwusd xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwusd xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x65,0x0f,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwusd xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwusd xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x65,0x18,0xd2,0x10]
          vpdpwusd xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vpdpwusd xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x61,0xd2,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwusd xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpwusd xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x65,0x8f,0xd2,0x51,0x7f]
          vpdpwusd xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vpdpwusd xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x65,0x9f,0xd2,0x52,0x80]
          vpdpwusd xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vpdpwusd ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x65,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwusd ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwusd ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x65,0x2f,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwusd ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwusd ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x65,0x38,0xd2,0x10]
          vpdpwusd ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vpdpwusd ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x65,0xd2,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwusd ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpwusd ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x65,0xaf,0xd2,0x51,0x7f]
          vpdpwusd ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vpdpwusd ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x65,0xbf,0xd2,0x52,0x80]
          vpdpwusd ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vpdpwusd zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwusd zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwusd zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x65,0x4f,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwusd zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwusd zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x65,0x58,0xd2,0x10]
          vpdpwusd zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vpdpwusd zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0xd2,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vpdpwusd zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vpdpwusd zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x65,0xcf,0xd2,0x51,0x7f]
          vpdpwusd zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vpdpwusd zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x65,0xdf,0xd2,0x52,0x80]
          vpdpwusd zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

// CHECK: vpdpwusds xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x61,0xd3,0xd4]
          vpdpwusds xmm2, xmm3, xmm4

// CHECK: vpdpwusds xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x65,0x0f,0xd3,0xd4]
          vpdpwusds xmm2 {k7}, xmm3, xmm4

// CHECK: vpdpwusds xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x65,0x8f,0xd3,0xd4]
          vpdpwusds xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vpdpwusds ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x65,0xd3,0xd4]
          vpdpwusds ymm2, ymm3, ymm4

// CHECK: vpdpwusds ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x65,0x2f,0xd3,0xd4]
          vpdpwusds ymm2 {k7}, ymm3, ymm4

// CHECK: vpdpwusds ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x65,0xaf,0xd3,0xd4]
          vpdpwusds ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vpdpwusds zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0xd3,0xd4]
          vpdpwusds zmm2, zmm3, zmm4

// CHECK: vpdpwusds zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x65,0x4f,0xd3,0xd4]
          vpdpwusds zmm2 {k7}, zmm3, zmm4

// CHECK: vpdpwusds zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x65,0xcf,0xd3,0xd4]
          vpdpwusds zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vpdpwusds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x61,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwusds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwusds xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x65,0x0f,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwusds xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwusds xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x65,0x18,0xd3,0x10]
          vpdpwusds xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vpdpwusds xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x61,0xd3,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwusds xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpwusds xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x65,0x8f,0xd3,0x51,0x7f]
          vpdpwusds xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vpdpwusds xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x65,0x9f,0xd3,0x52,0x80]
          vpdpwusds xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vpdpwusds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x65,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwusds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwusds ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x65,0x2f,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwusds ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwusds ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x65,0x38,0xd3,0x10]
          vpdpwusds ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vpdpwusds ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x65,0xd3,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwusds ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpwusds ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x65,0xaf,0xd3,0x51,0x7f]
          vpdpwusds ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vpdpwusds ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x65,0xbf,0xd3,0x52,0x80]
          vpdpwusds ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vpdpwusds zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwusds zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwusds zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x65,0x4f,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwusds zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwusds zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x65,0x58,0xd3,0x10]
          vpdpwusds zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vpdpwusds zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0xd3,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vpdpwusds zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vpdpwusds zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x65,0xcf,0xd3,0x51,0x7f]
          vpdpwusds zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vpdpwusds zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x65,0xdf,0xd3,0x52,0x80]
          vpdpwusds zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

// CHECK: vpdpwuud xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x60,0xd2,0xd4]
          vpdpwuud xmm2, xmm3, xmm4

// CHECK: vpdpwuud xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0xd2,0xd4]
          vpdpwuud xmm2 {k7}, xmm3, xmm4

// CHECK: vpdpwuud xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0xd2,0xd4]
          vpdpwuud xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vpdpwuud ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x64,0xd2,0xd4]
          vpdpwuud ymm2, ymm3, ymm4

// CHECK: vpdpwuud ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0xd2,0xd4]
          vpdpwuud ymm2 {k7}, ymm3, ymm4

// CHECK: vpdpwuud ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0xd2,0xd4]
          vpdpwuud ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vpdpwuud zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0xd2,0xd4]
          vpdpwuud zmm2, zmm3, zmm4

// CHECK: vpdpwuud zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0xd2,0xd4]
          vpdpwuud zmm2 {k7}, zmm3, zmm4

// CHECK: vpdpwuud zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0xd2,0xd4]
          vpdpwuud zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vpdpwuud xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x60,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwuud xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwuud xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwuud xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwuud xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x64,0x18,0xd2,0x10]
          vpdpwuud xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vpdpwuud xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x60,0xd2,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwuud xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpwuud xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0xd2,0x51,0x7f]
          vpdpwuud xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vpdpwuud xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x64,0x9f,0xd2,0x52,0x80]
          vpdpwuud xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vpdpwuud ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x64,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwuud ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwuud ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwuud ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwuud ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x64,0x38,0xd2,0x10]
          vpdpwuud ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vpdpwuud ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x64,0xd2,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwuud ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpwuud ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0xd2,0x51,0x7f]
          vpdpwuud ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vpdpwuud ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x64,0xbf,0xd2,0x52,0x80]
          vpdpwuud ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vpdpwuud zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwuud zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwuud zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwuud zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwuud zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x64,0x58,0xd2,0x10]
          vpdpwuud zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vpdpwuud zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0xd2,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vpdpwuud zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vpdpwuud zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0xd2,0x51,0x7f]
          vpdpwuud zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vpdpwuud zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x64,0xdf,0xd2,0x52,0x80]
          vpdpwuud zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

// CHECK: vpdpwuuds xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x60,0xd3,0xd4]
          vpdpwuuds xmm2, xmm3, xmm4

// CHECK: vpdpwuuds xmm2 {k7}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0xd3,0xd4]
          vpdpwuuds xmm2 {k7}, xmm3, xmm4

// CHECK: vpdpwuuds xmm2 {k7} {z}, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0xd3,0xd4]
          vpdpwuuds xmm2 {k7} {z}, xmm3, xmm4

// CHECK: vpdpwuuds ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0x64,0xd3,0xd4]
          vpdpwuuds ymm2, ymm3, ymm4

// CHECK: vpdpwuuds ymm2 {k7}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0xd3,0xd4]
          vpdpwuuds ymm2 {k7}, ymm3, ymm4

// CHECK: vpdpwuuds ymm2 {k7} {z}, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0xd3,0xd4]
          vpdpwuuds ymm2 {k7} {z}, ymm3, ymm4

// CHECK: vpdpwuuds zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0xd3,0xd4]
          vpdpwuuds zmm2, zmm3, zmm4

// CHECK: vpdpwuuds zmm2 {k7}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0xd3,0xd4]
          vpdpwuuds zmm2 {k7}, zmm3, zmm4

// CHECK: vpdpwuuds zmm2 {k7} {z}, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0xd3,0xd4]
          vpdpwuuds zmm2 {k7} {z}, zmm3, zmm4

// CHECK: vpdpwuuds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x60,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwuuds xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwuuds xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwuuds xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwuuds xmm2, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x64,0x18,0xd3,0x10]
          vpdpwuuds xmm2, xmm3, dword ptr [eax]{1to4}

// CHECK: vpdpwuuds xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x60,0xd3,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwuuds xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vpdpwuuds xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0xd3,0x51,0x7f]
          vpdpwuuds xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vpdpwuuds xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x64,0x9f,0xd3,0x52,0x80]
          vpdpwuuds xmm2 {k7} {z}, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vpdpwuuds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x64,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwuuds ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwuuds ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwuuds ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwuuds ymm2, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x64,0x38,0xd3,0x10]
          vpdpwuuds ymm2, ymm3, dword ptr [eax]{1to8}

// CHECK: vpdpwuuds ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x64,0xd3,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwuuds ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vpdpwuuds ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0xd3,0x51,0x7f]
          vpdpwuuds ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vpdpwuuds ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x64,0xbf,0xd3,0x52,0x80]
          vpdpwuuds ymm2 {k7} {z}, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vpdpwuuds zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
          vpdpwuuds zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vpdpwuuds zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
          vpdpwuuds zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vpdpwuuds zmm2, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x64,0x58,0xd3,0x10]
          vpdpwuuds zmm2, zmm3, dword ptr [eax]{1to16}

// CHECK: vpdpwuuds zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0xd3,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vpdpwuuds zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vpdpwuuds zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0xd3,0x51,0x7f]
          vpdpwuuds zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vpdpwuuds zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x64,0xdf,0xd3,0x52,0x80]
          vpdpwuuds zmm2 {k7} {z}, zmm3, dword ptr [edx - 512]{1to16}

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
