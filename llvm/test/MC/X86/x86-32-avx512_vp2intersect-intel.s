// RUN: llvm-mc -triple i386-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s
// CHECK: vp2intersectd k4, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x68,0xe4]
          vp2intersectd k4, zmm3, zmm4

// CHECK: vp2intersectd k4, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x68,0xa4,0xf4,0x00,0x00,0x00,0x10]
          vp2intersectd k4, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vp2intersectd k4, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x68,0xa4,0x87,0x23,0x01,0x00,0x00]
          vp2intersectd k4, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vp2intersectd k4, zmm3, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x67,0x58,0x68,0x20]
          vp2intersectd k4, zmm3, dword ptr [eax]{1to16}

// CHECK: vp2intersectd k4, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x68,0x24,0x6d,0x00,0xf8,0xff,0xff]
          vp2intersectd k4, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vp2intersectd k4, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x68,0x61,0x7f]
          vp2intersectd k4, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vp2intersectd k4, zmm3, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x67,0x58,0x68,0x62,0x80]
          vp2intersectd k4, zmm3, dword ptr [edx - 512]{1to16}

// CHECK: vp2intersectq k4, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0xe7,0x48,0x68,0xe4]
          vp2intersectq k4, zmm3, zmm4

// CHECK: vp2intersectq k4, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0xe7,0x48,0x68,0xa4,0xf4,0x00,0x00,0x00,0x10]
          vp2intersectq k4, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vp2intersectq k4, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0xe7,0x48,0x68,0xa4,0x87,0x23,0x01,0x00,0x00]
          vp2intersectq k4, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK: vp2intersectq k4, zmm3, qword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf2,0xe7,0x58,0x68,0x20]
          vp2intersectq k4, zmm3, qword ptr [eax]{1to8}

// CHECK: vp2intersectq k4, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0xe7,0x48,0x68,0x24,0x6d,0x00,0xf8,0xff,0xff]
          vp2intersectq k4, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK: vp2intersectq k4, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0xe7,0x48,0x68,0x61,0x7f]
          vp2intersectq k4, zmm3, zmmword ptr [ecx + 8128]

// CHECK: vp2intersectq k4, zmm3, qword ptr [edx - 1024]{1to8}
// CHECK: encoding: [0x62,0xf2,0xe7,0x58,0x68,0x62,0x80]
          vp2intersectq k4, zmm3, qword ptr [edx - 1024]{1to8}

