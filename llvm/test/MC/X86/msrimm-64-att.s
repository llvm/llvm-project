// RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
// RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

// ERROR-COUNT-2: error:
// ERROR-NOT: error:

// CHECK: rdmsr $123, %r9
// CHECK: encoding: [0xc4,0xc7,0x7b,0xf6,0xc1,0x7b,0x00,0x00,0x00]
          rdmsr $123, %r9

// CHECK: wrmsrns %r9, $123
// CHECK: encoding: [0xc4,0xc7,0x7a,0xf6,0xc1,0x7b,0x00,0x00,0x00]
          wrmsrns %r9, $123

