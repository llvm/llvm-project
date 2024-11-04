// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: erets
// CHECK: encoding: [0xf2,0x0f,0x01,0xca]
          erets

// CHECK: eretu
// CHECK: encoding: [0xf3,0x0f,0x01,0xca]
          eretu

