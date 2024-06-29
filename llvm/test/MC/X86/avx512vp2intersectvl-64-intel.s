// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vp2intersectd k6, ymm23, ymm24
// CHECK: encoding: [0x62,0x92,0x47,0x20,0x68,0xf0]
          vp2intersectd k6, ymm23, ymm24

// CHECK: vp2intersectd k6, xmm23, xmm24
// CHECK: encoding: [0x62,0x92,0x47,0x00,0x68,0xf0]
          vp2intersectd k6, xmm23, xmm24

// CHECK: vp2intersectd k6, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xb2,0x47,0x20,0x68,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vp2intersectd k6, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vp2intersectd k6, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xd2,0x47,0x20,0x68,0xb4,0x80,0x23,0x01,0x00,0x00]
          vp2intersectd k6, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vp2intersectd k6, ymm23, dword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xf2,0x47,0x30,0x68,0x35,0x00,0x00,0x00,0x00]
          vp2intersectd k6, ymm23, dword ptr [rip]{1to8}

// CHECK: vp2intersectd k6, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xf2,0x47,0x20,0x68,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vp2intersectd k6, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vp2intersectd k6, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xf2,0x47,0x20,0x68,0x71,0x7f]
          vp2intersectd k6, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vp2intersectd k6, ymm23, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0xf2,0x47,0x30,0x68,0x72,0x80]
          vp2intersectd k6, ymm23, dword ptr [rdx - 512]{1to8}

// CHECK: vp2intersectd k6, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xb2,0x47,0x00,0x68,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vp2intersectd k6, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vp2intersectd k6, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xd2,0x47,0x00,0x68,0xb4,0x80,0x23,0x01,0x00,0x00]
          vp2intersectd k6, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vp2intersectd k6, xmm23, dword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xf2,0x47,0x10,0x68,0x35,0x00,0x00,0x00,0x00]
          vp2intersectd k6, xmm23, dword ptr [rip]{1to4}

// CHECK: vp2intersectd k6, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xf2,0x47,0x00,0x68,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vp2intersectd k6, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vp2intersectd k6, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xf2,0x47,0x00,0x68,0x71,0x7f]
          vp2intersectd k6, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vp2intersectd k6, xmm23, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0xf2,0x47,0x10,0x68,0x72,0x80]
          vp2intersectd k6, xmm23, dword ptr [rdx - 512]{1to4}

// CHECK: vp2intersectq k6, ymm23, ymm24
// CHECK: encoding: [0x62,0x92,0xc7,0x20,0x68,0xf0]
          vp2intersectq k6, ymm23, ymm24

// CHECK: vp2intersectq k6, xmm23, xmm24
// CHECK: encoding: [0x62,0x92,0xc7,0x00,0x68,0xf0]
          vp2intersectq k6, xmm23, xmm24

// CHECK: vp2intersectq k6, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xb2,0xc7,0x20,0x68,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vp2intersectq k6, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vp2intersectq k6, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xd2,0xc7,0x20,0x68,0xb4,0x80,0x23,0x01,0x00,0x00]
          vp2intersectq k6, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vp2intersectq k6, ymm23, qword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xf2,0xc7,0x30,0x68,0x35,0x00,0x00,0x00,0x00]
          vp2intersectq k6, ymm23, qword ptr [rip]{1to4}

// CHECK: vp2intersectq k6, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xf2,0xc7,0x20,0x68,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vp2intersectq k6, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vp2intersectq k6, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xf2,0xc7,0x20,0x68,0x71,0x7f]
          vp2intersectq k6, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vp2intersectq k6, ymm23, qword ptr [rdx - 1024]{1to4}
// CHECK: encoding: [0x62,0xf2,0xc7,0x30,0x68,0x72,0x80]
          vp2intersectq k6, ymm23, qword ptr [rdx - 1024]{1to4}

// CHECK: vp2intersectq k6, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xb2,0xc7,0x00,0x68,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vp2intersectq k6, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vp2intersectq k6, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xd2,0xc7,0x00,0x68,0xb4,0x80,0x23,0x01,0x00,0x00]
          vp2intersectq k6, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vp2intersectq k6, xmm23, qword ptr [rip]{1to2}
// CHECK: encoding: [0x62,0xf2,0xc7,0x10,0x68,0x35,0x00,0x00,0x00,0x00]
          vp2intersectq k6, xmm23, qword ptr [rip]{1to2}

// CHECK: vp2intersectq k6, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xf2,0xc7,0x00,0x68,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vp2intersectq k6, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vp2intersectq k6, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xf2,0xc7,0x00,0x68,0x71,0x7f]
          vp2intersectq k6, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vp2intersectq k6, xmm23, qword ptr [rdx - 1024]{1to2}
// CHECK: encoding: [0x62,0xf2,0xc7,0x10,0x68,0x72,0x80]
          vp2intersectq k6, xmm23, qword ptr [rdx - 1024]{1to2}
