// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK:      vsm4key4 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x82,0x46,0x40,0xda,0xf0]
               vsm4key4 zmm22, zmm23, zmm24

// CHECK:      vsm4key4 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x46,0x40,0xda,0xb4,0xf5,0x00,0x00,0x00,0x10]
               vsm4key4 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK:      vsm4key4 zmm22, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x46,0x40,0xda,0xb4,0x80,0x23,0x01,0x00,0x00]
               vsm4key4 zmm22, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK:      vsm4key4 zmm22, zmm23, zmmword ptr [rip]
// CHECK: encoding: [0x62,0xe2,0x46,0x40,0xda,0x35,0x00,0x00,0x00,0x00]
               vsm4key4 zmm22, zmm23, zmmword ptr [rip]

// CHECK:      vsm4key4 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe2,0x46,0x40,0xda,0x34,0x6d,0x00,0xf8,0xff,0xff]
               vsm4key4 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK:      vsm4key4 zmm22, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe2,0x46,0x40,0xda,0x71,0x7f]
               vsm4key4 zmm22, zmm23, zmmword ptr [rcx + 8128]

// CHECK:      vsm4key4 zmm22, zmm23, zmmword ptr [rdx - 8192]
// CHECK: encoding: [0x62,0xe2,0x46,0x40,0xda,0x72,0x80]
               vsm4key4 zmm22, zmm23, zmmword ptr [rdx - 8192]

// CHECK:      vsm4rnds4 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x82,0x47,0x40,0xda,0xf0]
               vsm4rnds4 zmm22, zmm23, zmm24

// CHECK:      vsm4rnds4 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x47,0x40,0xda,0xb4,0xf5,0x00,0x00,0x00,0x10]
               vsm4rnds4 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK:      vsm4rnds4 zmm22, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x47,0x40,0xda,0xb4,0x80,0x23,0x01,0x00,0x00]
               vsm4rnds4 zmm22, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK:      vsm4rnds4 zmm22, zmm23, zmmword ptr [rip]
// CHECK: encoding: [0x62,0xe2,0x47,0x40,0xda,0x35,0x00,0x00,0x00,0x00]
               vsm4rnds4 zmm22, zmm23, zmmword ptr [rip]

// CHECK:      vsm4rnds4 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe2,0x47,0x40,0xda,0x34,0x6d,0x00,0xf8,0xff,0xff]
               vsm4rnds4 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK:      vsm4rnds4 zmm22, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe2,0x47,0x40,0xda,0x71,0x7f]
               vsm4rnds4 zmm22, zmm23, zmmword ptr [rcx + 8128]

// CHECK:      vsm4rnds4 zmm22, zmm23, zmmword ptr [rdx - 8192]
// CHECK: encoding: [0x62,0xe2,0x47,0x40,0xda,0x72,0x80]
               vsm4rnds4 zmm22, zmm23, zmmword ptr [rdx - 8192]

// CHECK:      vsm4key4 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x82,0x46,0x20,0xda,0xf0]
               vsm4key4 ymm22, ymm23, ymm24

// CHECK:      vsm4key4 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x82,0x46,0x00,0xda,0xf0]
               vsm4key4 xmm22, xmm23, xmm24

// CHECK:      vsm4key4 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x46,0x20,0xda,0xb4,0xf5,0x00,0x00,0x00,0x10]
               vsm4key4 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK:      vsm4key4 ymm22, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x46,0x20,0xda,0xb4,0x80,0x23,0x01,0x00,0x00]
               vsm4key4 ymm22, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK:      vsm4key4 ymm22, ymm23, ymmword ptr [rip]
// CHECK: encoding: [0x62,0xe2,0x46,0x20,0xda,0x35,0x00,0x00,0x00,0x00]
               vsm4key4 ymm22, ymm23, ymmword ptr [rip]

// CHECK:      vsm4key4 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe2,0x46,0x20,0xda,0x34,0x6d,0x00,0xfc,0xff,0xff]
               vsm4key4 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK:      vsm4key4 ymm22, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe2,0x46,0x20,0xda,0x71,0x7f]
               vsm4key4 ymm22, ymm23, ymmword ptr [rcx + 4064]

// CHECK:      vsm4key4 ymm22, ymm23, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0x62,0xe2,0x46,0x20,0xda,0x72,0x80]
               vsm4key4 ymm22, ymm23, ymmword ptr [rdx - 4096]

// CHECK:      vsm4key4 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x46,0x00,0xda,0xb4,0xf5,0x00,0x00,0x00,0x10]
               vsm4key4 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK:      vsm4key4 xmm22, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x46,0x00,0xda,0xb4,0x80,0x23,0x01,0x00,0x00]
               vsm4key4 xmm22, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK:      vsm4key4 xmm22, xmm23, xmmword ptr [rip]
// CHECK: encoding: [0x62,0xe2,0x46,0x00,0xda,0x35,0x00,0x00,0x00,0x00]
               vsm4key4 xmm22, xmm23, xmmword ptr [rip]

// CHECK:      vsm4key4 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe2,0x46,0x00,0xda,0x34,0x6d,0x00,0xfe,0xff,0xff]
               vsm4key4 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK:      vsm4key4 xmm22, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe2,0x46,0x00,0xda,0x71,0x7f]
               vsm4key4 xmm22, xmm23, xmmword ptr [rcx + 2032]

// CHECK:      vsm4key4 xmm22, xmm23, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0x62,0xe2,0x46,0x00,0xda,0x72,0x80]
               vsm4key4 xmm22, xmm23, xmmword ptr [rdx - 2048]

// CHECK:      vsm4rnds4 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x82,0x47,0x20,0xda,0xf0]
               vsm4rnds4 ymm22, ymm23, ymm24

// CHECK:      vsm4rnds4 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x82,0x47,0x00,0xda,0xf0]
               vsm4rnds4 xmm22, xmm23, xmm24

// CHECK:      vsm4rnds4 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x47,0x20,0xda,0xb4,0xf5,0x00,0x00,0x00,0x10]
               vsm4rnds4 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK:      vsm4rnds4 ymm22, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x47,0x20,0xda,0xb4,0x80,0x23,0x01,0x00,0x00]
               vsm4rnds4 ymm22, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK:      vsm4rnds4 ymm22, ymm23, ymmword ptr [rip]
// CHECK: encoding: [0x62,0xe2,0x47,0x20,0xda,0x35,0x00,0x00,0x00,0x00]
               vsm4rnds4 ymm22, ymm23, ymmword ptr [rip]

// CHECK:      vsm4rnds4 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe2,0x47,0x20,0xda,0x34,0x6d,0x00,0xfc,0xff,0xff]
               vsm4rnds4 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK:      vsm4rnds4 ymm22, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe2,0x47,0x20,0xda,0x71,0x7f]
               vsm4rnds4 ymm22, ymm23, ymmword ptr [rcx + 4064]

// CHECK:      vsm4rnds4 ymm22, ymm23, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0x62,0xe2,0x47,0x20,0xda,0x72,0x80]
               vsm4rnds4 ymm22, ymm23, ymmword ptr [rdx - 4096]

// CHECK:      vsm4rnds4 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x47,0x00,0xda,0xb4,0xf5,0x00,0x00,0x00,0x10]
               vsm4rnds4 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK:      vsm4rnds4 xmm22, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x47,0x00,0xda,0xb4,0x80,0x23,0x01,0x00,0x00]
               vsm4rnds4 xmm22, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK:      vsm4rnds4 xmm22, xmm23, xmmword ptr [rip]
// CHECK: encoding: [0x62,0xe2,0x47,0x00,0xda,0x35,0x00,0x00,0x00,0x00]
               vsm4rnds4 xmm22, xmm23, xmmword ptr [rip]

// CHECK:      vsm4rnds4 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe2,0x47,0x00,0xda,0x34,0x6d,0x00,0xfe,0xff,0xff]
               vsm4rnds4 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK:      vsm4rnds4 xmm22, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe2,0x47,0x00,0xda,0x71,0x7f]
               vsm4rnds4 xmm22, xmm23, xmmword ptr [rcx + 2032]

// CHECK:      vsm4rnds4 xmm22, xmm23, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0x62,0xe2,0x47,0x00,0xda,0x72,0x80]
               vsm4rnds4 xmm22, xmm23, xmmword ptr [rdx - 2048]
