// RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: rdmsrlist
// CHECK: encoding: [0xf2,0x0f,0x01,0xc6]
          rdmsrlist

// CHECK: wrmsrlist
// CHECK: encoding: [0xf3,0x0f,0x01,0xc6]
          wrmsrlist
