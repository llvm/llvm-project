// RUN: llvm-mc -triple x86_64-unknown-unknown -mattr=+avxvnniint8 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vpdpbssd ymm12, ymm13, ymm14
// CHECK: encoding: [0xc4,0x42,0x17,0x50,0xe6]
     vpdpbssd ymm12, ymm13, ymm14

// CHECK: vpdpbssd xmm12, xmm13, xmm14
// CHECK: encoding: [0xc4,0x42,0x13,0x50,0xe6]
     vpdpbssd xmm12, xmm13, xmm14

// CHECK: vpdpbssd ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x17,0x50,0xa4,0xf5,0x00,0x00,0x00,0x10]
     vpdpbssd ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpbssd ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x17,0x50,0xa4,0x80,0x23,0x01,0x00,0x00]
     vpdpbssd ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpbssd ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x17,0x50,0x25,0x00,0x00,0x00,0x00]
     vpdpbssd ymm12, ymm13, ymmword ptr [rip]

// CHECK: vpdpbssd ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x17,0x50,0x24,0x6d,0x00,0xfc,0xff,0xff]
     vpdpbssd ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vpdpbssd xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x13,0x50,0xa4,0xf5,0x00,0x00,0x00,0x10]
     vpdpbssd xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpbssd xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x13,0x50,0xa4,0x80,0x23,0x01,0x00,0x00]
     vpdpbssd xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpbssd xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x13,0x50,0x25,0x00,0x00,0x00,0x00]
     vpdpbssd xmm12, xmm13, xmmword ptr [rip]

// CHECK: vpdpbssd xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x13,0x50,0x24,0x6d,0x00,0xfe,0xff,0xff]
     vpdpbssd xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vpdpbssds ymm12, ymm13, ymm14
// CHECK: encoding: [0xc4,0x42,0x17,0x51,0xe6]
     vpdpbssds ymm12, ymm13, ymm14

// CHECK: vpdpbssds xmm12, xmm13, xmm14
// CHECK: encoding: [0xc4,0x42,0x13,0x51,0xe6]
     vpdpbssds xmm12, xmm13, xmm14

// CHECK: vpdpbssds ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x17,0x51,0xa4,0xf5,0x00,0x00,0x00,0x10]
     vpdpbssds ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpbssds ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x17,0x51,0xa4,0x80,0x23,0x01,0x00,0x00]
     vpdpbssds ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpbssds ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x17,0x51,0x25,0x00,0x00,0x00,0x00]
     vpdpbssds ymm12, ymm13, ymmword ptr [rip]

// CHECK: vpdpbssds ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x17,0x51,0x24,0x6d,0x00,0xfc,0xff,0xff]
     vpdpbssds ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vpdpbssds xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x13,0x51,0xa4,0xf5,0x00,0x00,0x00,0x10]
     vpdpbssds xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpbssds xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x13,0x51,0xa4,0x80,0x23,0x01,0x00,0x00]
     vpdpbssds xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpbssds xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x13,0x51,0x25,0x00,0x00,0x00,0x00]
     vpdpbssds xmm12, xmm13, xmmword ptr [rip]

// CHECK: vpdpbssds xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x13,0x51,0x24,0x6d,0x00,0xfe,0xff,0xff]
     vpdpbssds xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vpdpbsud ymm12, ymm13, ymm14
// CHECK: encoding: [0xc4,0x42,0x16,0x50,0xe6]
     vpdpbsud ymm12, ymm13, ymm14

// CHECK: vpdpbsud xmm12, xmm13, xmm14
// CHECK: encoding: [0xc4,0x42,0x12,0x50,0xe6]
     vpdpbsud xmm12, xmm13, xmm14

// CHECK: vpdpbsud ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x16,0x50,0xa4,0xf5,0x00,0x00,0x00,0x10]
     vpdpbsud ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpbsud ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x16,0x50,0xa4,0x80,0x23,0x01,0x00,0x00]
     vpdpbsud ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpbsud ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x16,0x50,0x25,0x00,0x00,0x00,0x00]
     vpdpbsud ymm12, ymm13, ymmword ptr [rip]

// CHECK: vpdpbsud ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x16,0x50,0x24,0x6d,0x00,0xfc,0xff,0xff]
     vpdpbsud ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vpdpbsud xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x12,0x50,0xa4,0xf5,0x00,0x00,0x00,0x10]
     vpdpbsud xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpbsud xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x12,0x50,0xa4,0x80,0x23,0x01,0x00,0x00]
     vpdpbsud xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpbsud xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x12,0x50,0x25,0x00,0x00,0x00,0x00]
     vpdpbsud xmm12, xmm13, xmmword ptr [rip]

// CHECK: vpdpbsud xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x12,0x50,0x24,0x6d,0x00,0xfe,0xff,0xff]
     vpdpbsud xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vpdpbsuds ymm12, ymm13, ymm14
// CHECK: encoding: [0xc4,0x42,0x16,0x51,0xe6]
     vpdpbsuds ymm12, ymm13, ymm14

// CHECK: vpdpbsuds xmm12, xmm13, xmm14
// CHECK: encoding: [0xc4,0x42,0x12,0x51,0xe6]
     vpdpbsuds xmm12, xmm13, xmm14

// CHECK: vpdpbsuds ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x16,0x51,0xa4,0xf5,0x00,0x00,0x00,0x10]
     vpdpbsuds ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpbsuds ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x16,0x51,0xa4,0x80,0x23,0x01,0x00,0x00]
     vpdpbsuds ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpbsuds ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x16,0x51,0x25,0x00,0x00,0x00,0x00]
     vpdpbsuds ymm12, ymm13, ymmword ptr [rip]

// CHECK: vpdpbsuds ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x16,0x51,0x24,0x6d,0x00,0xfc,0xff,0xff]
     vpdpbsuds ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vpdpbsuds xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x12,0x51,0xa4,0xf5,0x00,0x00,0x00,0x10]
     vpdpbsuds xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpbsuds xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x12,0x51,0xa4,0x80,0x23,0x01,0x00,0x00]
     vpdpbsuds xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpbsuds xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x12,0x51,0x25,0x00,0x00,0x00,0x00]
     vpdpbsuds xmm12, xmm13, xmmword ptr [rip]

// CHECK: vpdpbsuds xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x12,0x51,0x24,0x6d,0x00,0xfe,0xff,0xff]
     vpdpbsuds xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vpdpbuud ymm12, ymm13, ymm14
// CHECK: encoding: [0xc4,0x42,0x14,0x50,0xe6]
     vpdpbuud ymm12, ymm13, ymm14

// CHECK: vpdpbuud xmm12, xmm13, xmm14
// CHECK: encoding: [0xc4,0x42,0x10,0x50,0xe6]
     vpdpbuud xmm12, xmm13, xmm14

// CHECK: vpdpbuud ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x14,0x50,0xa4,0xf5,0x00,0x00,0x00,0x10]
     vpdpbuud ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpbuud ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x14,0x50,0xa4,0x80,0x23,0x01,0x00,0x00]
     vpdpbuud ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpbuud ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x14,0x50,0x25,0x00,0x00,0x00,0x00]
     vpdpbuud ymm12, ymm13, ymmword ptr [rip]

// CHECK: vpdpbuud ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x14,0x50,0x24,0x6d,0x00,0xfc,0xff,0xff]
     vpdpbuud ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vpdpbuud xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x10,0x50,0xa4,0xf5,0x00,0x00,0x00,0x10]
     vpdpbuud xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpbuud xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x10,0x50,0xa4,0x80,0x23,0x01,0x00,0x00]
     vpdpbuud xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpbuud xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x10,0x50,0x25,0x00,0x00,0x00,0x00]
     vpdpbuud xmm12, xmm13, xmmword ptr [rip]

// CHECK: vpdpbuud xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x10,0x50,0x24,0x6d,0x00,0xfe,0xff,0xff]
     vpdpbuud xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vpdpbuuds ymm12, ymm13, ymm14
// CHECK: encoding: [0xc4,0x42,0x14,0x51,0xe6]
     vpdpbuuds ymm12, ymm13, ymm14

// CHECK: vpdpbuuds xmm12, xmm13, xmm14
// CHECK: encoding: [0xc4,0x42,0x10,0x51,0xe6]
     vpdpbuuds xmm12, xmm13, xmm14

// CHECK: vpdpbuuds ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x14,0x51,0xa4,0xf5,0x00,0x00,0x00,0x10]
     vpdpbuuds ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpbuuds ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x14,0x51,0xa4,0x80,0x23,0x01,0x00,0x00]
     vpdpbuuds ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpbuuds ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x14,0x51,0x25,0x00,0x00,0x00,0x00]
     vpdpbuuds ymm12, ymm13, ymmword ptr [rip]

// CHECK: vpdpbuuds ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x14,0x51,0x24,0x6d,0x00,0xfc,0xff,0xff]
     vpdpbuuds ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vpdpbuuds xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x10,0x51,0xa4,0xf5,0x00,0x00,0x00,0x10]
     vpdpbuuds xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpbuuds xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x10,0x51,0xa4,0x80,0x23,0x01,0x00,0x00]
     vpdpbuuds xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpbuuds xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x10,0x51,0x25,0x00,0x00,0x00,0x00]
     vpdpbuuds xmm12, xmm13, xmmword ptr [rip]

// CHECK: vpdpbuuds xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x10,0x51,0x24,0x6d,0x00,0xfe,0xff,0xff]
     vpdpbuuds xmm12, xmm13, xmmword ptr [2*rbp - 512]

