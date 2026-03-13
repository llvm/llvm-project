// RUN: llvm-mc -triple i386-unknown-unknown-code16 --x86-asm-syntax=intel --show-encoding %s | FileCheck %s

// CHECK: pushl $8
// CHECK: encoding: [0x66,0x6a,0x08]
          data32 push 8

// CHECK: pushw $8
// CHECK: encoding: [0x6a,0x08]
          push 8

// CHECK: lretl
// CHECK: encoding: [0x66,0xcb]
          data32 retf
