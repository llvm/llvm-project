// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+rand  < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+v8.5a < %s 2>&1 | FileCheck %s --check-prefix=NORAND
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=-rand  < %s 2>&1 | FileCheck %s --check-prefix=NORAND

mrs x0, rndr
mrs x1, rndrrs

// CHECK: mrs x0, RNDR      // encoding: [0x00,0x24,0x3b,0xd5]
// CHECK: mrs x1, RNDRRS    // encoding: [0x21,0x24,0x3b,0xd5]

// NORAND: expected readable system register
// NORAND-NEXT: rndr
// NORAND: expected readable system register
// NORAND-NEXT: rndrrs
