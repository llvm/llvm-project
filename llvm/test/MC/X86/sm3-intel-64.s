// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vsm3msg1 xmm12, xmm13, xmm4
// CHECK: encoding: [0xc4,0x62,0x10,0xda,0xe4]
          vsm3msg1 xmm12, xmm13, xmm4

// CHECK: vsm3msg1 xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x10,0xda,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vsm3msg1 xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsm3msg1 xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x10,0xda,0xa4,0x80,0x23,0x01,0x00,0x00]
          vsm3msg1 xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vsm3msg1 xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x10,0xda,0x25,0x00,0x00,0x00,0x00]
          vsm3msg1 xmm12, xmm13, xmmword ptr [rip]

// CHECK: vsm3msg1 xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x10,0xda,0x24,0x6d,0x00,0xfe,0xff,0xff]
          vsm3msg1 xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vsm3msg1 xmm12, xmm13, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0xc4,0x62,0x10,0xda,0xa1,0xf0,0x07,0x00,0x00]
          vsm3msg1 xmm12, xmm13, xmmword ptr [rcx + 2032]

// CHECK: vsm3msg1 xmm12, xmm13, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0xc4,0x62,0x10,0xda,0xa2,0x00,0xf8,0xff,0xff]
          vsm3msg1 xmm12, xmm13, xmmword ptr [rdx - 2048]

// CHECK: vsm3msg2 xmm12, xmm13, xmm4
// CHECK: encoding: [0xc4,0x62,0x11,0xda,0xe4]
          vsm3msg2 xmm12, xmm13, xmm4

// CHECK: vsm3msg2 xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x11,0xda,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vsm3msg2 xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsm3msg2 xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x11,0xda,0xa4,0x80,0x23,0x01,0x00,0x00]
          vsm3msg2 xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vsm3msg2 xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x11,0xda,0x25,0x00,0x00,0x00,0x00]
          vsm3msg2 xmm12, xmm13, xmmword ptr [rip]

// CHECK: vsm3msg2 xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x11,0xda,0x24,0x6d,0x00,0xfe,0xff,0xff]
          vsm3msg2 xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vsm3msg2 xmm12, xmm13, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0xc4,0x62,0x11,0xda,0xa1,0xf0,0x07,0x00,0x00]
          vsm3msg2 xmm12, xmm13, xmmword ptr [rcx + 2032]

// CHECK: vsm3msg2 xmm12, xmm13, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0xc4,0x62,0x11,0xda,0xa2,0x00,0xf8,0xff,0xff]
          vsm3msg2 xmm12, xmm13, xmmword ptr [rdx - 2048]

// CHECK: vsm3rnds2 xmm12, xmm13, xmm4, 123
// CHECK: encoding: [0xc4,0x63,0x11,0xde,0xe4,0x7b]
          vsm3rnds2 xmm12, xmm13, xmm4, 123

// CHECK: vsm3rnds2 xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0xc4,0x23,0x11,0xde,0xa4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vsm3rnds2 xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vsm3rnds2 xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0xc4,0x43,0x11,0xde,0xa4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vsm3rnds2 xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vsm3rnds2 xmm12, xmm13, xmmword ptr [rip], 123
// CHECK: encoding: [0xc4,0x63,0x11,0xde,0x25,0x00,0x00,0x00,0x00,0x7b]
          vsm3rnds2 xmm12, xmm13, xmmword ptr [rip], 123

// CHECK: vsm3rnds2 xmm12, xmm13, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0xc4,0x63,0x11,0xde,0x24,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vsm3rnds2 xmm12, xmm13, xmmword ptr [2*rbp - 512], 123

// CHECK: vsm3rnds2 xmm12, xmm13, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0xc4,0x63,0x11,0xde,0xa1,0xf0,0x07,0x00,0x00,0x7b]
          vsm3rnds2 xmm12, xmm13, xmmword ptr [rcx + 2032], 123

// CHECK: vsm3rnds2 xmm12, xmm13, xmmword ptr [rdx - 2048], 123
// CHECK: encoding: [0xc4,0x63,0x11,0xde,0xa2,0x00,0xf8,0xff,0xff,0x7b]
          vsm3rnds2 xmm12, xmm13, xmmword ptr [rdx - 2048], 123

