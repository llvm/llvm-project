// RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
// RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

// ERROR-COUNT-2: error:
// ERROR-NOT: error:

// CHECK: erets
// CHECK: encoding: [0xf2,0x0f,0x01,0xca]
          erets

// CHECK: eretu
// CHECK: encoding: [0xf3,0x0f,0x01,0xca]
          eretu

