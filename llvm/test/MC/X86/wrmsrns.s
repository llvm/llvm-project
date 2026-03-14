// RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s
// RUN: llvm-mc -triple i386 --show-encoding %s | FileCheck %s
// RUN: llvm-mc -triple i386 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: wrmsrns
// CHECK: encoding: [0x0f,0x01,0xc6]
          wrmsrns
