// RUN: llvm-mc -triple i386-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vp2intersectd k6, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x68,0xf4]
          vp2intersectd k6, ymm3, ymm4

// CHECK: vp2intersectd k6, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x68,0xf4]
          vp2intersectd k6, xmm3, xmm4

// CHECK: vp2intersectd k6, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x68,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vp2intersectd k6, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vp2intersectd k6, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x68,0xb4,0x87,0x23,0x01,0x00,0x00]
          vp2intersectd k6, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vp2intersectd k6, ymm3, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0x67,0x38,0x68,0x30]
          vp2intersectd k6, ymm3, dword ptr [eax]{1to8}

// CHECK: vp2intersectd k6, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x68,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vp2intersectd k6, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vp2intersectd k6, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x68,0x71,0x7f]
          vp2intersectd k6, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vp2intersectd k6, ymm3, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x67,0x38,0x68,0x72,0x80]
          vp2intersectd k6, ymm3, dword ptr [edx - 512]{1to8}

// CHECK: vp2intersectd k6, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x68,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vp2intersectd k6, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vp2intersectd k6, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x68,0xb4,0x87,0x23,0x01,0x00,0x00]
          vp2intersectd k6, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vp2intersectd k6, xmm3, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0x67,0x18,0x68,0x30]
          vp2intersectd k6, xmm3, dword ptr [eax]{1to4}

// CHECK: vp2intersectd k6, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x68,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vp2intersectd k6, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vp2intersectd k6, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x68,0x71,0x7f]
          vp2intersectd k6, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vp2intersectd k6, xmm3, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x67,0x18,0x68,0x72,0x80]
          vp2intersectd k6, xmm3, dword ptr [edx - 512]{1to4}

// CHECK: vp2intersectq k6, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0xe7,0x28,0x68,0xf4]
          vp2intersectq k6, ymm3, ymm4

// CHECK: vp2intersectq k6, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0xe7,0x08,0x68,0xf4]
          vp2intersectq k6, xmm3, xmm4

// CHECK: vp2intersectq k6, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0xe7,0x28,0x68,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vp2intersectq k6, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vp2intersectq k6, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0xe7,0x28,0x68,0xb4,0x87,0x23,0x01,0x00,0x00]
          vp2intersectq k6, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: vp2intersectq k6, ymm3, qword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf2,0xe7,0x38,0x68,0x30]
          vp2intersectq k6, ymm3, qword ptr [eax]{1to4}

// CHECK: vp2intersectq k6, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf2,0xe7,0x28,0x68,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vp2intersectq k6, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: vp2intersectq k6, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0xe7,0x28,0x68,0x71,0x7f]
          vp2intersectq k6, ymm3, ymmword ptr [ecx + 4064]

// CHECK: vp2intersectq k6, ymm3, qword ptr [edx - 1024]{1to4}
// CHECK: encoding: [0x62,0xf2,0xe7,0x38,0x68,0x72,0x80]
          vp2intersectq k6, ymm3, qword ptr [edx - 1024]{1to4}

// CHECK: vp2intersectq k6, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0xe7,0x08,0x68,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vp2intersectq k6, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vp2intersectq k6, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0xe7,0x08,0x68,0xb4,0x87,0x23,0x01,0x00,0x00]
          vp2intersectq k6, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: vp2intersectq k6, xmm3, qword ptr [eax]{1to2}
// CHECK: encoding: [0x62,0xf2,0xe7,0x18,0x68,0x30]
          vp2intersectq k6, xmm3, qword ptr [eax]{1to2}

// CHECK: vp2intersectq k6, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf2,0xe7,0x08,0x68,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vp2intersectq k6, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: vp2intersectq k6, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0xe7,0x08,0x68,0x71,0x7f]
          vp2intersectq k6, xmm3, xmmword ptr [ecx + 2032]

// CHECK: vp2intersectq k6, xmm3, qword ptr [edx - 1024]{1to2}
// CHECK: encoding: [0x62,0xf2,0xe7,0x18,0x68,0x72,0x80]
          vp2intersectq k6, xmm3, qword ptr [edx - 1024]{1to2}
