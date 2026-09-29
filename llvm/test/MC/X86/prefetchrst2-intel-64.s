// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: prefetchrst2 byte ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x42,0x0f,0x18,0xa4,0xf5,0x00,0x00,0x00,0x10]
          prefetchrst2 byte ptr [rbp + 8*r14 + 268435456]

// CHECK: prefetchrst2 byte ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x41,0x0f,0x18,0xa4,0x80,0x23,0x01,0x00,0x00]
          prefetchrst2 byte ptr [r8 + 4*rax + 291]

// CHECK: prefetchrst2 byte ptr [rip]
// CHECK: encoding: [0x0f,0x18,0x25,0x00,0x00,0x00,0x00]
          prefetchrst2 byte ptr [rip]

// CHECK: prefetchrst2 byte ptr [2*rbp - 32]
// CHECK: encoding: [0x0f,0x18,0x24,0x6d,0xe0,0xff,0xff,0xff]
          prefetchrst2 byte ptr [2*rbp - 32]

// CHECK: prefetchrst2 byte ptr [rcx + 127]
// CHECK: encoding: [0x0f,0x18,0x61,0x7f]
          prefetchrst2 byte ptr [rcx + 127]

// CHECK: prefetchrst2 byte ptr [rdx - 128]
// CHECK: encoding: [0x0f,0x18,0x62,0x80]
          prefetchrst2 byte ptr [rdx - 128]