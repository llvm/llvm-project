// RUN: llvm-mc -triple aarch64 -show-encoding < %s | FileCheck %s

mrs x0, HACDBSBR_EL2
// CHECK: mrs x0, HACDBSBR_EL2                  // encoding: [0x80,0x23,0x3c,0xd5]
msr HACDBSBR_EL2, x0
// CHECK: msr HACDBSBR_EL2, x0                  // encoding: [0x80,0x23,0x1c,0xd5]

mrs x0, HACDBSCONS_EL2
// CHECK: mrs x0, HACDBSCONS_EL2                  // encoding: [0xa0,0x23,0x3c,0xd5]
msr HACDBSCONS_EL2, x0
// CHECK: msr HACDBSCONS_EL2, x0                  // encoding: [0xa0,0x23,0x1c,0xd5]

