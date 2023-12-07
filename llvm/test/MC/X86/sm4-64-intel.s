// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vsm4key4 ymm12, ymm13, ymm4
// CHECK: encoding: [0xc4,0x62,0x16,0xda,0xe4]
          vsm4key4 ymm12, ymm13, ymm4

// CHECK: vsm4key4 xmm12, xmm13, xmm4
// CHECK: encoding: [0xc4,0x62,0x12,0xda,0xe4]
          vsm4key4 xmm12, xmm13, xmm4

// CHECK: vsm4key4 ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x16,0xda,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vsm4key4 ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsm4key4 ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x16,0xda,0xa4,0x80,0x23,0x01,0x00,0x00]
          vsm4key4 ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vsm4key4 ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x16,0xda,0x25,0x00,0x00,0x00,0x00]
          vsm4key4 ymm12, ymm13, ymmword ptr [rip]

// CHECK: vsm4key4 ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x16,0xda,0x24,0x6d,0x00,0xfc,0xff,0xff]
          vsm4key4 ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vsm4key4 ymm12, ymm13, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0xc4,0x62,0x16,0xda,0xa1,0xe0,0x0f,0x00,0x00]
          vsm4key4 ymm12, ymm13, ymmword ptr [rcx + 4064]

// CHECK: vsm4key4 ymm12, ymm13, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0xc4,0x62,0x16,0xda,0xa2,0x00,0xf0,0xff,0xff]
          vsm4key4 ymm12, ymm13, ymmword ptr [rdx - 4096]

// CHECK: vsm4key4 xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x12,0xda,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vsm4key4 xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsm4key4 xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x12,0xda,0xa4,0x80,0x23,0x01,0x00,0x00]
          vsm4key4 xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vsm4key4 xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x12,0xda,0x25,0x00,0x00,0x00,0x00]
          vsm4key4 xmm12, xmm13, xmmword ptr [rip]

// CHECK: vsm4key4 xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x12,0xda,0x24,0x6d,0x00,0xfe,0xff,0xff]
          vsm4key4 xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vsm4key4 xmm12, xmm13, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0xc4,0x62,0x12,0xda,0xa1,0xf0,0x07,0x00,0x00]
          vsm4key4 xmm12, xmm13, xmmword ptr [rcx + 2032]

// CHECK: vsm4key4 xmm12, xmm13, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0xc4,0x62,0x12,0xda,0xa2,0x00,0xf8,0xff,0xff]
          vsm4key4 xmm12, xmm13, xmmword ptr [rdx - 2048]

// CHECK: vsm4rnds4 ymm12, ymm13, ymm4
// CHECK: encoding: [0xc4,0x62,0x17,0xda,0xe4]
          vsm4rnds4 ymm12, ymm13, ymm4

// CHECK: vsm4rnds4 xmm12, xmm13, xmm4
// CHECK: encoding: [0xc4,0x62,0x13,0xda,0xe4]
          vsm4rnds4 xmm12, xmm13, xmm4

// CHECK: vsm4rnds4 ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x17,0xda,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vsm4rnds4 ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsm4rnds4 ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x17,0xda,0xa4,0x80,0x23,0x01,0x00,0x00]
          vsm4rnds4 ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vsm4rnds4 ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x17,0xda,0x25,0x00,0x00,0x00,0x00]
          vsm4rnds4 ymm12, ymm13, ymmword ptr [rip]

// CHECK: vsm4rnds4 ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x17,0xda,0x24,0x6d,0x00,0xfc,0xff,0xff]
          vsm4rnds4 ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vsm4rnds4 ymm12, ymm13, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0xc4,0x62,0x17,0xda,0xa1,0xe0,0x0f,0x00,0x00]
          vsm4rnds4 ymm12, ymm13, ymmword ptr [rcx + 4064]

// CHECK: vsm4rnds4 ymm12, ymm13, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0xc4,0x62,0x17,0xda,0xa2,0x00,0xf0,0xff,0xff]
          vsm4rnds4 ymm12, ymm13, ymmword ptr [rdx - 4096]

// CHECK: vsm4rnds4 xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x13,0xda,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vsm4rnds4 xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsm4rnds4 xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x13,0xda,0xa4,0x80,0x23,0x01,0x00,0x00]
          vsm4rnds4 xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vsm4rnds4 xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x13,0xda,0x25,0x00,0x00,0x00,0x00]
          vsm4rnds4 xmm12, xmm13, xmmword ptr [rip]

// CHECK: vsm4rnds4 xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x13,0xda,0x24,0x6d,0x00,0xfe,0xff,0xff]
          vsm4rnds4 xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vsm4rnds4 xmm12, xmm13, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0xc4,0x62,0x13,0xda,0xa1,0xf0,0x07,0x00,0x00]
          vsm4rnds4 xmm12, xmm13, xmmword ptr [rcx + 2032]

// CHECK: vsm4rnds4 xmm12, xmm13, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0xc4,0x62,0x13,0xda,0xa2,0x00,0xf8,0xff,0xff]
          vsm4rnds4 xmm12, xmm13, xmmword ptr [rdx - 2048]

