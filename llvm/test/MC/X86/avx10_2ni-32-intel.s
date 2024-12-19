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

// CHECK: vcmppd k5, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0xc2,0xec,0x7b]
          vcmppd k5, ymm3, ymm4, {sae}, 123

// CHECK: vcmppd k5 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf1,0xe1,0x1f,0xc2,0xec,0x7b]
          vcmppd k5 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vcmpph k5, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x60,0x18,0xc2,0xec,0x7b]
          vcmpph k5, ymm3, ymm4, {sae}, 123

// CHECK: vcmpph k5 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x60,0x1f,0xc2,0xec,0x7b]
          vcmpph k5 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vcmpps k5, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0xc2,0xec,0x7b]
          vcmpps k5, ymm3, ymm4, {sae}, 123

// CHECK: vcmpps k5 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf1,0x60,0x1f,0xc2,0xec,0x7b]
          vcmpps k5 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vcvtdq2ph xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x5b,0xd3]
          vcvtdq2ph xmm2, ymm3, {rn-sae}

// CHECK: vcvtdq2ph xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x3f,0x5b,0xd3]
          vcvtdq2ph xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtdq2ph xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0xff,0x5b,0xd3]
          vcvtdq2ph xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtdq2ps ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x18,0x5b,0xd3]
          vcvtdq2ps ymm2, ymm3, {rn-sae}

// CHECK: vcvtdq2ps ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x3f,0x5b,0xd3]
          vcvtdq2ps ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtdq2ps ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0xff,0x5b,0xd3]
          vcvtdq2ps ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtpd2dq xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xfb,0x18,0xe6,0xd3]
          vcvtpd2dq xmm2, ymm3, {rn-sae}

// CHECK: vcvtpd2dq xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xfb,0x3f,0xe6,0xd3]
          vcvtpd2dq xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtpd2dq xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xfb,0xff,0xe6,0xd3]
          vcvtpd2dq xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtpd2ph xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0xf9,0x18,0x5a,0xd3]
          vcvtpd2ph xmm2, ymm3, {rn-sae}

// CHECK: vcvtpd2ph xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0xf9,0x3f,0x5a,0xd3]
          vcvtpd2ph xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtpd2ph xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0xf9,0xff,0x5a,0xd3]
          vcvtpd2ph xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtpd2ps xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0x5a,0xd3]
          vcvtpd2ps xmm2, ymm3, {rn-sae}

// CHECK: vcvtpd2ps xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x3f,0x5a,0xd3]
          vcvtpd2ps xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtpd2ps xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0xff,0x5a,0xd3]
          vcvtpd2ps xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtpd2qq ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0x7b,0xd3]
          vcvtpd2qq ymm2, ymm3, {rn-sae}

// CHECK: vcvtpd2qq ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x3f,0x7b,0xd3]
          vcvtpd2qq ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtpd2qq ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0xff,0x7b,0xd3]
          vcvtpd2qq ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtpd2udq xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x18,0x79,0xd3]
          vcvtpd2udq xmm2, ymm3, {rn-sae}

// CHECK: vcvtpd2udq xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x3f,0x79,0xd3]
          vcvtpd2udq xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtpd2udq xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0xff,0x79,0xd3]
          vcvtpd2udq xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtpd2uqq ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0x79,0xd3]
          vcvtpd2uqq ymm2, ymm3, {rn-sae}

// CHECK: vcvtpd2uqq ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x3f,0x79,0xd3]
          vcvtpd2uqq ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtpd2uqq ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0xff,0x79,0xd3]
          vcvtpd2uqq ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtph2dq ymm2, xmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x5b,0xd3]
          vcvtph2dq ymm2, xmm3, {rn-sae}

// CHECK: vcvtph2dq ymm2 {k7}, xmm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x3f,0x5b,0xd3]
          vcvtph2dq ymm2 {k7}, xmm3, {rd-sae}

// CHECK: vcvtph2dq ymm2 {k7} {z}, xmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x5b,0xd3]
          vcvtph2dq ymm2 {k7} {z}, xmm3, {rz-sae}

// CHECK: vcvtph2pd ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x5a,0xd3]
          vcvtph2pd ymm2, xmm3, {sae}

// CHECK: vcvtph2pd ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x1f,0x5a,0xd3]
          vcvtph2pd ymm2 {k7}, xmm3, {sae}

// CHECK: vcvtph2pd ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x5a,0xd3]
          vcvtph2pd ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvtph2ps ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf2,0x79,0x18,0x13,0xd3]
          vcvtph2ps ymm2, xmm3, {sae}

// CHECK: vcvtph2ps ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf2,0x79,0x1f,0x13,0xd3]
          vcvtph2ps ymm2 {k7}, xmm3, {sae}

// CHECK: vcvtph2ps ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf2,0x79,0x9f,0x13,0xd3]
          vcvtph2ps ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvtph2psx ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf6,0x79,0x18,0x13,0xd3]
          vcvtph2psx ymm2, xmm3, {sae}

// CHECK: vcvtph2psx ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf6,0x79,0x1f,0x13,0xd3]
          vcvtph2psx ymm2 {k7}, xmm3, {sae}

// CHECK: vcvtph2psx ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf6,0x79,0x9f,0x13,0xd3]
          vcvtph2psx ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvtph2qq ymm2, xmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x7b,0xd3]
          vcvtph2qq ymm2, xmm3, {rn-sae}

// CHECK: vcvtph2qq ymm2 {k7}, xmm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x3f,0x7b,0xd3]
          vcvtph2qq ymm2 {k7}, xmm3, {rd-sae}

// CHECK: vcvtph2qq ymm2 {k7} {z}, xmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x7b,0xd3]
          vcvtph2qq ymm2 {k7} {z}, xmm3, {rz-sae}

// CHECK: vcvtph2udq ymm2, xmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x79,0xd3]
          vcvtph2udq ymm2, xmm3, {rn-sae}

// CHECK: vcvtph2udq ymm2 {k7}, xmm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x3f,0x79,0xd3]
          vcvtph2udq ymm2 {k7}, xmm3, {rd-sae}

// CHECK: vcvtph2udq ymm2 {k7} {z}, xmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0xff,0x79,0xd3]
          vcvtph2udq ymm2 {k7} {z}, xmm3, {rz-sae}

// CHECK: vcvtph2uqq ymm2, xmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x79,0xd3]
          vcvtph2uqq ymm2, xmm3, {rn-sae}

// CHECK: vcvtph2uqq ymm2 {k7}, xmm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x3f,0x79,0xd3]
          vcvtph2uqq ymm2 {k7}, xmm3, {rd-sae}

// CHECK: vcvtph2uqq ymm2 {k7} {z}, xmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x79,0xd3]
          vcvtph2uqq ymm2 {k7} {z}, xmm3, {rz-sae}

// CHECK: vcvtph2uw ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x7d,0xd3]
          vcvtph2uw ymm2, ymm3, {rn-sae}

// CHECK: vcvtph2uw ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x3f,0x7d,0xd3]
          vcvtph2uw ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtph2uw ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0xff,0x7d,0xd3]
          vcvtph2uw ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtph2w ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x7d,0xd3]
          vcvtph2w ymm2, ymm3, {rn-sae}

// CHECK: vcvtph2w ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x3f,0x7d,0xd3]
          vcvtph2w ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtph2w ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x7d,0xd3]
          vcvtph2w ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtps2dq ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x18,0x5b,0xd3]
          vcvtps2dq ymm2, ymm3, {rn-sae}

// CHECK: vcvtps2dq ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x3f,0x5b,0xd3]
          vcvtps2dq ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtps2dq ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0xff,0x5b,0xd3]
          vcvtps2dq ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtps2pd ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x18,0x5a,0xd3]
          vcvtps2pd ymm2, xmm3, {sae}

// CHECK: vcvtps2pd ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x1f,0x5a,0xd3]
          vcvtps2pd ymm2 {k7}, xmm3, {sae}

// CHECK: vcvtps2pd ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x9f,0x5a,0xd3]
          vcvtps2pd ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvtps2ph xmm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x18,0x1d,0xda,0x7b]
          vcvtps2ph xmm2, ymm3, {sae}, 123

// CHECK: vcvtps2ph xmm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x1f,0x1d,0xda,0x7b]
          vcvtps2ph xmm2 {k7}, ymm3, {sae}, 123

// CHECK: vcvtps2ph xmm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x9f,0x1d,0xda,0x7b]
          vcvtps2ph xmm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vcvtps2phx xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x1d,0xd3]
          vcvtps2phx xmm2, ymm3, {rn-sae}

// CHECK: vcvtps2phx xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x3f,0x1d,0xd3]
          vcvtps2phx xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtps2phx xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x1d,0xd3]
          vcvtps2phx xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtps2qq ymm2, xmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x18,0x7b,0xd3]
          vcvtps2qq ymm2, xmm3, {rn-sae}

// CHECK: vcvtps2qq ymm2 {k7}, xmm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x3f,0x7b,0xd3]
          vcvtps2qq ymm2 {k7}, xmm3, {rd-sae}

// CHECK: vcvtps2qq ymm2 {k7} {z}, xmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0xff,0x7b,0xd3]
          vcvtps2qq ymm2 {k7} {z}, xmm3, {rz-sae}

// CHECK: vcvtps2udq ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x18,0x79,0xd3]
          vcvtps2udq ymm2, ymm3, {rn-sae}

// CHECK: vcvtps2udq ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x3f,0x79,0xd3]
          vcvtps2udq ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtps2udq ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0xff,0x79,0xd3]
          vcvtps2udq ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtps2uqq ymm2, xmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x18,0x79,0xd3]
          vcvtps2uqq ymm2, xmm3, {rn-sae}

// CHECK: vcvtps2uqq ymm2 {k7}, xmm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x3f,0x79,0xd3]
          vcvtps2uqq ymm2 {k7}, xmm3, {rd-sae}

// CHECK: vcvtps2uqq ymm2 {k7} {z}, xmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0xff,0x79,0xd3]
          vcvtps2uqq ymm2 {k7} {z}, xmm3, {rz-sae}

// CHECK: vcvtqq2pd ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xfa,0x18,0xe6,0xd3]
          vcvtqq2pd ymm2, ymm3, {rn-sae}

// CHECK: vcvtqq2pd ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xfa,0x3f,0xe6,0xd3]
          vcvtqq2pd ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtqq2pd ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xfa,0xff,0xe6,0xd3]
          vcvtqq2pd ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtqq2ph xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0xf8,0x18,0x5b,0xd3]
          vcvtqq2ph xmm2, ymm3, {rn-sae}

// CHECK: vcvtqq2ph xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0xf8,0x3f,0x5b,0xd3]
          vcvtqq2ph xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtqq2ph xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0xf8,0xff,0x5b,0xd3]
          vcvtqq2ph xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtqq2ps xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x18,0x5b,0xd3]
          vcvtqq2ps xmm2, ymm3, {rn-sae}

// CHECK: vcvtqq2ps xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x3f,0x5b,0xd3]
          vcvtqq2ps xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtqq2ps xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0xff,0x5b,0xd3]
          vcvtqq2ps xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvttpd2dq xmm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0xe6,0xd3]
          vcvttpd2dq xmm2, ymm3, {sae}

// CHECK: vcvttpd2dq xmm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x1f,0xe6,0xd3]
          vcvttpd2dq xmm2 {k7}, ymm3, {sae}

// CHECK: vcvttpd2dq xmm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x9f,0xe6,0xd3]
          vcvttpd2dq xmm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttpd2qq ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0x7a,0xd3]
          vcvttpd2qq ymm2, ymm3, {sae}

// CHECK: vcvttpd2qq ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x1f,0x7a,0xd3]
          vcvttpd2qq ymm2 {k7}, ymm3, {sae}

// CHECK: vcvttpd2qq ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x9f,0x7a,0xd3]
          vcvttpd2qq ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttpd2udq xmm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x18,0x78,0xd3]
          vcvttpd2udq xmm2, ymm3, {sae}

// CHECK: vcvttpd2udq xmm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x1f,0x78,0xd3]
          vcvttpd2udq xmm2 {k7}, ymm3, {sae}

// CHECK: vcvttpd2udq xmm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x9f,0x78,0xd3]
          vcvttpd2udq xmm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttpd2uqq ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0x78,0xd3]
          vcvttpd2uqq ymm2, ymm3, {sae}

// CHECK: vcvttpd2uqq ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x1f,0x78,0xd3]
          vcvttpd2uqq ymm2 {k7}, ymm3, {sae}

// CHECK: vcvttpd2uqq ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x9f,0x78,0xd3]
          vcvttpd2uqq ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttph2dq ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7a,0x18,0x5b,0xd3]
          vcvttph2dq ymm2, xmm3, {sae}

// CHECK: vcvttph2dq ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7a,0x1f,0x5b,0xd3]
          vcvttph2dq ymm2 {k7}, xmm3, {sae}

// CHECK: vcvttph2dq ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7a,0x9f,0x5b,0xd3]
          vcvttph2dq ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvttph2qq ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x7a,0xd3]
          vcvttph2qq ymm2, xmm3, {sae}

// CHECK: vcvttph2qq ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x1f,0x7a,0xd3]
          vcvttph2qq ymm2 {k7}, xmm3, {sae}

// CHECK: vcvttph2qq ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x7a,0xd3]
          vcvttph2qq ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvttph2udq ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x78,0xd3]
          vcvttph2udq ymm2, xmm3, {sae}

// CHECK: vcvttph2udq ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x1f,0x78,0xd3]
          vcvttph2udq ymm2 {k7}, xmm3, {sae}

// CHECK: vcvttph2udq ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x78,0xd3]
          vcvttph2udq ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvttph2uqq ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x78,0xd3]
          vcvttph2uqq ymm2, xmm3, {sae}

// CHECK: vcvttph2uqq ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x1f,0x78,0xd3]
          vcvttph2uqq ymm2 {k7}, xmm3, {sae}

// CHECK: vcvttph2uqq ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x78,0xd3]
          vcvttph2uqq ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvttph2uw ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x7c,0xd3]
          vcvttph2uw ymm2, ymm3, {sae}

// CHECK: vcvttph2uw ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x1f,0x7c,0xd3]
          vcvttph2uw ymm2 {k7}, ymm3, {sae}

// CHECK: vcvttph2uw ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x7c,0xd3]
          vcvttph2uw ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttph2w ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x7c,0xd3]
          vcvttph2w ymm2, ymm3, {sae}

// CHECK: vcvttph2w ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x1f,0x7c,0xd3]
          vcvttph2w ymm2 {k7}, ymm3, {sae}

// CHECK: vcvttph2w ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x7c,0xd3]
          vcvttph2w ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttps2dq ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x7a,0x18,0x5b,0xd3]
          vcvttps2dq ymm2, ymm3, {sae}

// CHECK: vcvttps2dq ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x7a,0x1f,0x5b,0xd3]
          vcvttps2dq ymm2 {k7}, ymm3, {sae}

// CHECK: vcvttps2dq ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x7a,0x9f,0x5b,0xd3]
          vcvttps2dq ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttps2qq ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x18,0x7a,0xd3]
          vcvttps2qq ymm2, xmm3, {sae}

// CHECK: vcvttps2qq ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x1f,0x7a,0xd3]
          vcvttps2qq ymm2 {k7}, xmm3, {sae}

// CHECK: vcvttps2qq ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x9f,0x7a,0xd3]
          vcvttps2qq ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvttps2udq ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x18,0x78,0xd3]
          vcvttps2udq ymm2, ymm3, {sae}

// CHECK: vcvttps2udq ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x1f,0x78,0xd3]
          vcvttps2udq ymm2 {k7}, ymm3, {sae}

// CHECK: vcvttps2udq ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x9f,0x78,0xd3]
          vcvttps2udq ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttps2uqq ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x18,0x78,0xd3]
          vcvttps2uqq ymm2, xmm3, {sae}

// CHECK: vcvttps2uqq ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x1f,0x78,0xd3]
          vcvttps2uqq ymm2 {k7}, xmm3, {sae}

// CHECK: vcvttps2uqq ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x9f,0x78,0xd3]
          vcvttps2uqq ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvtudq2ph xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7b,0x18,0x7a,0xd3]
          vcvtudq2ph xmm2, ymm3, {rn-sae}

// CHECK: vcvtudq2ph xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x7b,0x3f,0x7a,0xd3]
          vcvtudq2ph xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtudq2ph xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x7b,0xff,0x7a,0xd3]
          vcvtudq2ph xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtudq2ps ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x7b,0x18,0x7a,0xd3]
          vcvtudq2ps ymm2, ymm3, {rn-sae}

// CHECK: vcvtudq2ps ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x7b,0x3f,0x7a,0xd3]
          vcvtudq2ps ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtudq2ps ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x7b,0xff,0x7a,0xd3]
          vcvtudq2ps ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtuqq2pd ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xfa,0x18,0x7a,0xd3]
          vcvtuqq2pd ymm2, ymm3, {rn-sae}

// CHECK: vcvtuqq2pd ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xfa,0x3f,0x7a,0xd3]
          vcvtuqq2pd ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtuqq2pd ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xfa,0xff,0x7a,0xd3]
          vcvtuqq2pd ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtuqq2ph xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0xfb,0x18,0x7a,0xd3]
          vcvtuqq2ph xmm2, ymm3, {rn-sae}

// CHECK: vcvtuqq2ph xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0xfb,0x3f,0x7a,0xd3]
          vcvtuqq2ph xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtuqq2ph xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0xfb,0xff,0x7a,0xd3]
          vcvtuqq2ph xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtuqq2ps xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xfb,0x18,0x7a,0xd3]
          vcvtuqq2ps xmm2, ymm3, {rn-sae}

// CHECK: vcvtuqq2ps xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xfb,0x3f,0x7a,0xd3]
          vcvtuqq2ps xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtuqq2ps xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xfb,0xff,0x7a,0xd3]
          vcvtuqq2ps xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtuw2ph ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7b,0x18,0x7d,0xd3]
          vcvtuw2ph ymm2, ymm3, {rn-sae}

// CHECK: vcvtuw2ph ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x7b,0x3f,0x7d,0xd3]
          vcvtuw2ph ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtuw2ph ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x7b,0xff,0x7d,0xd3]
          vcvtuw2ph ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtw2ph ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7a,0x18,0x7d,0xd3]
          vcvtw2ph ymm2, ymm3, {rn-sae}

// CHECK: vcvtw2ph ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x7a,0x3f,0x7d,0xd3]
          vcvtw2ph ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtw2ph ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x7a,0xff,0x7d,0xd3]
          vcvtw2ph ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vdivpd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0x5e,0xd4]
          vdivpd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vdivpd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x3f,0x5e,0xd4]
          vdivpd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vdivpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0xff,0x5e,0xd4]
          vdivpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vdivph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x18,0x5e,0xd4]
          vdivph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vdivph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x3f,0x5e,0xd4]
          vdivph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vdivph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0xff,0x5e,0xd4]
          vdivph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vdivps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0x5e,0xd4]
          vdivps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vdivps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x3f,0x5e,0xd4]
          vdivps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vdivps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0xff,0x5e,0xd4]
          vdivps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfcmaddcph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x63,0x18,0x56,0xd4]
          vfcmaddcph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfcmaddcph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x63,0x3f,0x56,0xd4]
          vfcmaddcph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfcmaddcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x63,0xff,0x56,0xd4]
          vfcmaddcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfcmulcph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x63,0x18,0xd6,0xd4]
          vfcmulcph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfcmulcph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x63,0x3f,0xd6,0xd4]
          vfcmulcph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfcmulcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x63,0xff,0xd6,0xd4]
          vfcmulcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfixupimmpd ymm2, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x18,0x54,0xd4,0x7b]
          vfixupimmpd ymm2, ymm3, ymm4, {sae}, 123

// CHECK: vfixupimmpd ymm2 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x1f,0x54,0xd4,0x7b]
          vfixupimmpd ymm2 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vfixupimmpd ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x9f,0x54,0xd4,0x7b]
          vfixupimmpd ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123

// CHECK: vfixupimmps ymm2, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x18,0x54,0xd4,0x7b]
          vfixupimmps ymm2, ymm3, ymm4, {sae}, 123

// CHECK: vfixupimmps ymm2 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x1f,0x54,0xd4,0x7b]
          vfixupimmps ymm2 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vfixupimmps ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x9f,0x54,0xd4,0x7b]
          vfixupimmps ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123

// CHECK: vfmadd132pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x98,0xd4]
          vfmadd132pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x98,0xd4]
          vfmadd132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x98,0xd4]
          vfmadd132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd132ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x98,0xd4]
          vfmadd132ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x98,0xd4]
          vfmadd132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x98,0xd4]
          vfmadd132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd132ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x98,0xd4]
          vfmadd132ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x98,0xd4]
          vfmadd132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x98,0xd4]
          vfmadd132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd213pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xa8,0xd4]
          vfmadd213pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xa8,0xd4]
          vfmadd213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xa8,0xd4]
          vfmadd213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd213ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xa8,0xd4]
          vfmadd213ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xa8,0xd4]
          vfmadd213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xa8,0xd4]
          vfmadd213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd213ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xa8,0xd4]
          vfmadd213ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xa8,0xd4]
          vfmadd213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xa8,0xd4]
          vfmadd213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd231pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xb8,0xd4]
          vfmadd231pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xb8,0xd4]
          vfmadd231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xb8,0xd4]
          vfmadd231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd231ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xb8,0xd4]
          vfmadd231ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xb8,0xd4]
          vfmadd231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xb8,0xd4]
          vfmadd231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd231ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xb8,0xd4]
          vfmadd231ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xb8,0xd4]
          vfmadd231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xb8,0xd4]
          vfmadd231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddcph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x62,0x18,0x56,0xd4]
          vfmaddcph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddcph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x62,0x3f,0x56,0xd4]
          vfmaddcph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x62,0xff,0x56,0xd4]
          vfmaddcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub132pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x96,0xd4]
          vfmaddsub132pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x96,0xd4]
          vfmaddsub132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x96,0xd4]
          vfmaddsub132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub132ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x96,0xd4]
          vfmaddsub132ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x96,0xd4]
          vfmaddsub132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x96,0xd4]
          vfmaddsub132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub132ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x96,0xd4]
          vfmaddsub132ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x96,0xd4]
          vfmaddsub132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x96,0xd4]
          vfmaddsub132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub213pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xa6,0xd4]
          vfmaddsub213pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xa6,0xd4]
          vfmaddsub213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xa6,0xd4]
          vfmaddsub213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub213ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xa6,0xd4]
          vfmaddsub213ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xa6,0xd4]
          vfmaddsub213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xa6,0xd4]
          vfmaddsub213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub213ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xa6,0xd4]
          vfmaddsub213ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xa6,0xd4]
          vfmaddsub213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xa6,0xd4]
          vfmaddsub213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub231pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xb6,0xd4]
          vfmaddsub231pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xb6,0xd4]
          vfmaddsub231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xb6,0xd4]
          vfmaddsub231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub231ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xb6,0xd4]
          vfmaddsub231ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xb6,0xd4]
          vfmaddsub231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xb6,0xd4]
          vfmaddsub231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub231ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xb6,0xd4]
          vfmaddsub231ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xb6,0xd4]
          vfmaddsub231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xb6,0xd4]
          vfmaddsub231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub132pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x9a,0xd4]
          vfmsub132pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x9a,0xd4]
          vfmsub132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x9a,0xd4]
          vfmsub132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub132ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x9a,0xd4]
          vfmsub132ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x9a,0xd4]
          vfmsub132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x9a,0xd4]
          vfmsub132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub132ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x9a,0xd4]
          vfmsub132ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x9a,0xd4]
          vfmsub132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x9a,0xd4]
          vfmsub132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub213pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xaa,0xd4]
          vfmsub213pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xaa,0xd4]
          vfmsub213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xaa,0xd4]
          vfmsub213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub213ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xaa,0xd4]
          vfmsub213ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xaa,0xd4]
          vfmsub213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xaa,0xd4]
          vfmsub213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub213ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xaa,0xd4]
          vfmsub213ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xaa,0xd4]
          vfmsub213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xaa,0xd4]
          vfmsub213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub231pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xba,0xd4]
          vfmsub231pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xba,0xd4]
          vfmsub231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xba,0xd4]
          vfmsub231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub231ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xba,0xd4]
          vfmsub231ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xba,0xd4]
          vfmsub231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xba,0xd4]
          vfmsub231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub231ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xba,0xd4]
          vfmsub231ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xba,0xd4]
          vfmsub231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xba,0xd4]
          vfmsub231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd132pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x97,0xd4]
          vfmsubadd132pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x97,0xd4]
          vfmsubadd132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x97,0xd4]
          vfmsubadd132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd132ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x97,0xd4]
          vfmsubadd132ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x97,0xd4]
          vfmsubadd132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x97,0xd4]
          vfmsubadd132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd132ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x97,0xd4]
          vfmsubadd132ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x97,0xd4]
          vfmsubadd132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x97,0xd4]
          vfmsubadd132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd213pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xa7,0xd4]
          vfmsubadd213pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xa7,0xd4]
          vfmsubadd213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xa7,0xd4]
          vfmsubadd213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd213ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xa7,0xd4]
          vfmsubadd213ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xa7,0xd4]
          vfmsubadd213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xa7,0xd4]
          vfmsubadd213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd213ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xa7,0xd4]
          vfmsubadd213ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xa7,0xd4]
          vfmsubadd213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xa7,0xd4]
          vfmsubadd213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd231pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xb7,0xd4]
          vfmsubadd231pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xb7,0xd4]
          vfmsubadd231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xb7,0xd4]
          vfmsubadd231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd231ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xb7,0xd4]
          vfmsubadd231ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xb7,0xd4]
          vfmsubadd231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xb7,0xd4]
          vfmsubadd231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd231ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xb7,0xd4]
          vfmsubadd231ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xb7,0xd4]
          vfmsubadd231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xb7,0xd4]
          vfmsubadd231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmulcph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x62,0x18,0xd6,0xd4]
          vfmulcph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmulcph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x62,0x3f,0xd6,0xd4]
          vfmulcph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmulcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x62,0xff,0xd6,0xd4]
          vfmulcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd132pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x9c,0xd4]
          vfnmadd132pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x9c,0xd4]
          vfnmadd132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x9c,0xd4]
          vfnmadd132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd132ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x9c,0xd4]
          vfnmadd132ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x9c,0xd4]
          vfnmadd132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x9c,0xd4]
          vfnmadd132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd132ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x9c,0xd4]
          vfnmadd132ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x9c,0xd4]
          vfnmadd132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x9c,0xd4]
          vfnmadd132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd213pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xac,0xd4]
          vfnmadd213pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xac,0xd4]
          vfnmadd213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xac,0xd4]
          vfnmadd213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd213ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xac,0xd4]
          vfnmadd213ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xac,0xd4]
          vfnmadd213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xac,0xd4]
          vfnmadd213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd213ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xac,0xd4]
          vfnmadd213ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xac,0xd4]
          vfnmadd213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xac,0xd4]
          vfnmadd213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd231pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xbc,0xd4]
          vfnmadd231pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xbc,0xd4]
          vfnmadd231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xbc,0xd4]
          vfnmadd231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd231ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xbc,0xd4]
          vfnmadd231ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xbc,0xd4]
          vfnmadd231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xbc,0xd4]
          vfnmadd231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd231ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xbc,0xd4]
          vfnmadd231ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xbc,0xd4]
          vfnmadd231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xbc,0xd4]
          vfnmadd231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub132pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x9e,0xd4]
          vfnmsub132pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x9e,0xd4]
          vfnmsub132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x9e,0xd4]
          vfnmsub132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub132ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x9e,0xd4]
          vfnmsub132ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x9e,0xd4]
          vfnmsub132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x9e,0xd4]
          vfnmsub132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub132ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x9e,0xd4]
          vfnmsub132ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x9e,0xd4]
          vfnmsub132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x9e,0xd4]
          vfnmsub132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub213pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xae,0xd4]
          vfnmsub213pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xae,0xd4]
          vfnmsub213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xae,0xd4]
          vfnmsub213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub213ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xae,0xd4]
          vfnmsub213ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xae,0xd4]
          vfnmsub213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xae,0xd4]
          vfnmsub213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub213ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xae,0xd4]
          vfnmsub213ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xae,0xd4]
          vfnmsub213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xae,0xd4]
          vfnmsub213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub231pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xbe,0xd4]
          vfnmsub231pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xbe,0xd4]
          vfnmsub231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xbe,0xd4]
          vfnmsub231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub231ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xbe,0xd4]
          vfnmsub231ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xbe,0xd4]
          vfnmsub231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xbe,0xd4]
          vfnmsub231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub231ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xbe,0xd4]
          vfnmsub231ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xbe,0xd4]
          vfnmsub231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xbe,0xd4]
          vfnmsub231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vgetexppd ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf2,0xf9,0x18,0x42,0xd3]
          vgetexppd ymm2, ymm3, {sae}

// CHECK: vgetexppd ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf2,0xf9,0x1f,0x42,0xd3]
          vgetexppd ymm2 {k7}, ymm3, {sae}

// CHECK: vgetexppd ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf2,0xf9,0x9f,0x42,0xd3]
          vgetexppd ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vgetexpph ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf6,0x79,0x18,0x42,0xd3]
          vgetexpph ymm2, ymm3, {sae}

// CHECK: vgetexpph ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf6,0x79,0x1f,0x42,0xd3]
          vgetexpph ymm2 {k7}, ymm3, {sae}

// CHECK: vgetexpph ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf6,0x79,0x9f,0x42,0xd3]
          vgetexpph ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vgetexpps ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf2,0x79,0x18,0x42,0xd3]
          vgetexpps ymm2, ymm3, {sae}

// CHECK: vgetexpps ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf2,0x79,0x1f,0x42,0xd3]
          vgetexpps ymm2 {k7}, ymm3, {sae}

// CHECK: vgetexpps ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf2,0x79,0x9f,0x42,0xd3]
          vgetexpps ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vgetmantpd ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x18,0x26,0xd3,0x7b]
          vgetmantpd ymm2, ymm3, {sae}, 123

// CHECK: vgetmantpd ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x1f,0x26,0xd3,0x7b]
          vgetmantpd ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vgetmantpd ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x9f,0x26,0xd3,0x7b]
          vgetmantpd ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vgetmantph ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x18,0x26,0xd3,0x7b]
          vgetmantph ymm2, ymm3, {sae}, 123

// CHECK: vgetmantph ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x1f,0x26,0xd3,0x7b]
          vgetmantph ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vgetmantph ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x9f,0x26,0xd3,0x7b]
          vgetmantph ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vgetmantps ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x18,0x26,0xd3,0x7b]
          vgetmantps ymm2, ymm3, {sae}, 123

// CHECK: vgetmantps ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x1f,0x26,0xd3,0x7b]
          vgetmantps ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vgetmantps ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x9f,0x26,0xd3,0x7b]
          vgetmantps ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vmaxpd ymm2, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0x5f,0xd4]
          vmaxpd ymm2, ymm3, ymm4, {sae}

// CHECK: vmaxpd ymm2 {k7}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x1f,0x5f,0xd4]
          vmaxpd ymm2 {k7}, ymm3, ymm4, {sae}

// CHECK: vmaxpd ymm2 {k7} {z}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x9f,0x5f,0xd4]
          vmaxpd ymm2 {k7} {z}, ymm3, ymm4, {sae}

// CHECK: vmaxph ymm2, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x18,0x5f,0xd4]
          vmaxph ymm2, ymm3, ymm4, {sae}

// CHECK: vmaxph ymm2 {k7}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x1f,0x5f,0xd4]
          vmaxph ymm2 {k7}, ymm3, ymm4, {sae}

// CHECK: vmaxph ymm2 {k7} {z}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x9f,0x5f,0xd4]
          vmaxph ymm2 {k7} {z}, ymm3, ymm4, {sae}

// CHECK: vmaxps ymm2, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0x5f,0xd4]
          vmaxps ymm2, ymm3, ymm4, {sae}

// CHECK: vmaxps ymm2 {k7}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x1f,0x5f,0xd4]
          vmaxps ymm2 {k7}, ymm3, ymm4, {sae}

// CHECK: vmaxps ymm2 {k7} {z}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x9f,0x5f,0xd4]
          vmaxps ymm2 {k7} {z}, ymm3, ymm4, {sae}

// CHECK: vminpd ymm2, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0x5d,0xd4]
          vminpd ymm2, ymm3, ymm4, {sae}

// CHECK: vminpd ymm2 {k7}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x1f,0x5d,0xd4]
          vminpd ymm2 {k7}, ymm3, ymm4, {sae}

// CHECK: vminpd ymm2 {k7} {z}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x9f,0x5d,0xd4]
          vminpd ymm2 {k7} {z}, ymm3, ymm4, {sae}

// CHECK: vminph ymm2, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x18,0x5d,0xd4]
          vminph ymm2, ymm3, ymm4, {sae}

// CHECK: vminph ymm2 {k7}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x1f,0x5d,0xd4]
          vminph ymm2 {k7}, ymm3, ymm4, {sae}

// CHECK: vminph ymm2 {k7} {z}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x9f,0x5d,0xd4]
          vminph ymm2 {k7} {z}, ymm3, ymm4, {sae}

// CHECK: vminps ymm2, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0x5d,0xd4]
          vminps ymm2, ymm3, ymm4, {sae}

// CHECK: vminps ymm2 {k7}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x1f,0x5d,0xd4]
          vminps ymm2 {k7}, ymm3, ymm4, {sae}

// CHECK: vminps ymm2 {k7} {z}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x9f,0x5d,0xd4]
          vminps ymm2 {k7} {z}, ymm3, ymm4, {sae}

// CHECK: vmulpd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0x59,0xd4]
          vmulpd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vmulpd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x3f,0x59,0xd4]
          vmulpd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vmulpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0xff,0x59,0xd4]
          vmulpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vmulph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x18,0x59,0xd4]
          vmulph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vmulph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x3f,0x59,0xd4]
          vmulph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vmulph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0xff,0x59,0xd4]
          vmulph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vmulps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0x59,0xd4]
          vmulps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vmulps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x3f,0x59,0xd4]
          vmulps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vmulps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0xff,0x59,0xd4]
          vmulps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vrangepd ymm2, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x18,0x50,0xd4,0x7b]
          vrangepd ymm2, ymm3, ymm4, {sae}, 123

// CHECK: vrangepd ymm2 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x1f,0x50,0xd4,0x7b]
          vrangepd ymm2 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vrangepd ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x9f,0x50,0xd4,0x7b]
          vrangepd ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123

// CHECK: vrangeps ymm2, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x18,0x50,0xd4,0x7b]
          vrangeps ymm2, ymm3, ymm4, {sae}, 123

// CHECK: vrangeps ymm2 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x1f,0x50,0xd4,0x7b]
          vrangeps ymm2 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vrangeps ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x9f,0x50,0xd4,0x7b]
          vrangeps ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123

// CHECK: vreducepd ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x18,0x56,0xd3,0x7b]
          vreducepd ymm2, ymm3, {sae}, 123

// CHECK: vreducepd ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x1f,0x56,0xd3,0x7b]
          vreducepd ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vreducepd ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x9f,0x56,0xd3,0x7b]
          vreducepd ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vreduceph ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x18,0x56,0xd3,0x7b]
          vreduceph ymm2, ymm3, {sae}, 123

// CHECK: vreduceph ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x1f,0x56,0xd3,0x7b]
          vreduceph ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vreduceph ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x9f,0x56,0xd3,0x7b]
          vreduceph ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vreduceps ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x18,0x56,0xd3,0x7b]
          vreduceps ymm2, ymm3, {sae}, 123

// CHECK: vreduceps ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x1f,0x56,0xd3,0x7b]
          vreduceps ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vreduceps ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x9f,0x56,0xd3,0x7b]
          vreduceps ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vrndscalepd ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x18,0x09,0xd3,0x7b]
          vrndscalepd ymm2, ymm3, {sae}, 123

// CHECK: vrndscalepd ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x1f,0x09,0xd3,0x7b]
          vrndscalepd ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vrndscalepd ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x9f,0x09,0xd3,0x7b]
          vrndscalepd ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vrndscaleph ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x18,0x08,0xd3,0x7b]
          vrndscaleph ymm2, ymm3, {sae}, 123

// CHECK: vrndscaleph ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x1f,0x08,0xd3,0x7b]
          vrndscaleph ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vrndscaleph ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x9f,0x08,0xd3,0x7b]
          vrndscaleph ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vrndscaleps ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x18,0x08,0xd3,0x7b]
          vrndscaleps ymm2, ymm3, {sae}, 123

// CHECK: vrndscaleps ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x1f,0x08,0xd3,0x7b]
          vrndscaleps ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vrndscaleps ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x9f,0x08,0xd3,0x7b]
          vrndscaleps ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vscalefpd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x2c,0xd4]
          vscalefpd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vscalefpd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x2c,0xd4]
          vscalefpd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vscalefpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x2c,0xd4]
          vscalefpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vscalefph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x2c,0xd4]
          vscalefph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vscalefph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x2c,0xd4]
          vscalefph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vscalefph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x2c,0xd4]
          vscalefph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vscalefps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x2c,0xd4]
          vscalefps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vscalefps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x2c,0xd4]
          vscalefps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vscalefps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x2c,0xd4]
          vscalefps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vsqrtpd ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0x51,0xd3]
          vsqrtpd ymm2, ymm3, {rn-sae}

// CHECK: vsqrtpd ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x3f,0x51,0xd3]
          vsqrtpd ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vsqrtpd ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0xff,0x51,0xd3]
          vsqrtpd ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vsqrtph ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x51,0xd3]
          vsqrtph ymm2, ymm3, {rn-sae}

// CHECK: vsqrtph ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x3f,0x51,0xd3]
          vsqrtph ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vsqrtph ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0xff,0x51,0xd3]
          vsqrtph ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vsqrtps ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x18,0x51,0xd3]
          vsqrtps ymm2, ymm3, {rn-sae}

// CHECK: vsqrtps ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x3f,0x51,0xd3]
          vsqrtps ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vsqrtps ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0xff,0x51,0xd3]
          vsqrtps ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vsubpd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0x5c,0xd4]
          vsubpd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vsubpd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x3f,0x5c,0xd4]
          vsubpd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vsubpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0xff,0x5c,0xd4]
          vsubpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vsubph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x18,0x5c,0xd4]
          vsubph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vsubph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x3f,0x5c,0xd4]
          vsubph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vsubph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0xff,0x5c,0xd4]
          vsubph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vsubps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0x5c,0xd4]
          vsubps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vsubps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x3f,0x5c,0xd4]
          vsubps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vsubps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0xff,0x5c,0xd4]
          vsubps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
