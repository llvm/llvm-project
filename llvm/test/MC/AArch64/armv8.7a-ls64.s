// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+ls64 < %s 2>%t | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERR --check-prefix=CHECK-LS64-ERR %s < %t
// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERR --check-prefix=CHECK-NO-LS64-ERR %s < %t

  ld64b x0, [x13]
  st64b x14, [x13]
  st64bv x1, x20, [x13]
  st64bv0 x1, x22, [x13]
// CHECK: ld64b x0, [x13]        // encoding: [0xa0,0xd1,0x3f,0xf8]
// CHECK: st64b x14, [x13]       // encoding: [0xae,0x91,0x3f,0xf8]
// CHECK: st64bv x1, x20, [x13]  // encoding: [0xb4,0xb1,0x21,0xf8]
// CHECK: st64bv0 x1, x22, [x13] // encoding: [0xb6,0xa1,0x21,0xf8]
// CHECK-NO-LS64-ERR: [[@LINE-8]]:3: error: instruction requires: ls64
// CHECK-NO-LS64-ERR: [[@LINE-8]]:3: error: instruction requires: ls64
// CHECK-NO-LS64-ERR: [[@LINE-8]]:3: error: instruction requires: ls64
// CHECK-NO-LS64-ERR: [[@LINE-8]]:3: error: instruction requires: ls64

  ld64b x0, [sp]
  st64b x14, [sp]
  st64bv x1, x20, [sp]
  st64bv0 x1, x22, [sp]
// CHECK: ld64b x0, [sp]         // encoding: [0xe0,0xd3,0x3f,0xf8]
// CHECK: st64b x14, [sp]        // encoding: [0xee,0x93,0x3f,0xf8]
// CHECK: st64bv x1, x20, [sp]   // encoding: [0xf4,0xb3,0x21,0xf8]
// CHECK: st64bv0 x1, x22, [sp]  // encoding: [0xf6,0xa3,0x21,0xf8]

  ld64b x1, [x13]
  ld64b x24, [x13]
// CHECK-ERR: [[@LINE-2]]:9: error: expected an even-numbered x-register in the range [x0,x22]
// CHECK-ERR: [[@LINE-2]]:9: error: expected an even-numbered x-register in the range [x0,x22]

  mrs x0, accdata_el1
  msr accdata_el1, x0
// CHECK: mrs x0, ACCDATA_EL1    // encoding: [0xa0,0xd0,0x38,0xd5]
// CHECK: msr ACCDATA_EL1, x0    // encoding: [0xa0,0xd0,0x18,0xd5]
// CHECK-NO-LS64-ERR: [[@LINE-4]]:11: error: expected readable system register
// CHECK-NO-LS64-ERR: [[@LINE-4]]:7: error: expected writable system register
