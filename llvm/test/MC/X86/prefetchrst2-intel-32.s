// RUN: llvm-mc -triple i386-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: prefetchrst2 byte ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x0f,0x18,0xa4,0xf4,0x00,0x00,0x00,0x10]
prefetchrst2 byte ptr [esp + 8*esi + 268435456]

// CHECK: prefetchrst2 byte ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x0f,0x18,0xa4,0x87,0x23,0x01,0x00,0x00]
prefetchrst2 byte ptr [edi + 4*eax + 291]

// CHECK: prefetchrst2 byte ptr [eax]
// CHECK: encoding: [0x0f,0x18,0x20]
prefetchrst2 byte ptr [eax]

// CHECK: prefetchrst2 byte ptr [2*ebp - 32]
// CHECK: encoding: [0x0f,0x18,0x24,0x6d,0xe0,0xff,0xff,0xff]
prefetchrst2 byte ptr [2*ebp - 32]

// CHECK: prefetchrst2 byte ptr [ecx + 127]
// CHECK: encoding: [0x0f,0x18,0x61,0x7f]
prefetchrst2 byte ptr [ecx + 127]

// CHECK: prefetchrst2 byte ptr [edx - 128]
// CHECK: encoding: [0x0f,0x18,0x62,0x80]
prefetchrst2 byte ptr [edx - 128]