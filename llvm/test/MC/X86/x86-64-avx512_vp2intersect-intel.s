// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s
// CHECK: vp2intersectd k4, zmm23, zmm24
// CHECK: encoding: [0x62,0x92,0x47,0x40,0x68,0xe0]
          vp2intersectd k4, zmm23, zmm24

// CHECK: vp2intersectd k4, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xb2,0x47,0x40,0x68,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vp2intersectd k4, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vp2intersectd k4, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xd2,0x47,0x40,0x68,0xa4,0x80,0x23,0x01,0x00,0x00]
          vp2intersectd k4, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vp2intersectd k4, zmm23, dword ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xf2,0x47,0x50,0x68,0x25,0x00,0x00,0x00,0x00]
          vp2intersectd k4, zmm23, dword ptr [rip]{1to16}

// CHECK: vp2intersectd k4, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xf2,0x47,0x40,0x68,0x24,0x6d,0x00,0xf8,0xff,0xff]
          vp2intersectd k4, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vp2intersectd k4, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xf2,0x47,0x40,0x68,0x61,0x7f]
          vp2intersectd k4, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vp2intersectd k4, zmm23, dword ptr [rdx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x47,0x50,0x68,0x62,0x80]
          vp2intersectd k4, zmm23, dword ptr [rdx - 512]{1to16}

// CHECK: vp2intersectq k4, zmm23, zmm24
// CHECK: encoding: [0x62,0x92,0xc7,0x40,0x68,0xe0]
          vp2intersectq k4, zmm23, zmm24

// CHECK: vp2intersectq k4, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xb2,0xc7,0x40,0x68,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vp2intersectq k4, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vp2intersectq k4, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xd2,0xc7,0x40,0x68,0xa4,0x80,0x23,0x01,0x00,0x00]
          vp2intersectq k4, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vp2intersectq k4, zmm23, qword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xf2,0xc7,0x50,0x68,0x25,0x00,0x00,0x00,0x00]
          vp2intersectq k4, zmm23, qword ptr [rip]{1to8}

// CHECK: vp2intersectq k4, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xf2,0xc7,0x40,0x68,0x24,0x6d,0x00,0xf8,0xff,0xff]
          vp2intersectq k4, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vp2intersectq k4, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xf2,0xc7,0x40,0x68,0x61,0x7f]
          vp2intersectq k4, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vp2intersectq k4, zmm23, qword ptr [rdx - 1024]{1to8}
// CHECK: encoding: [0x62,0xf2,0xc7,0x50,0x68,0x62,0x80]
          vp2intersectq k4, zmm23, qword ptr [rdx - 1024]{1to8}

