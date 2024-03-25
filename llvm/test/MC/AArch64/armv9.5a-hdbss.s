// RUN: llvm-mc -triple aarch64 -show-encoding < %s | FileCheck %s

mrs x0, HDBSSBR_EL2
// CHECK: mrs x0, HDBSSBR_EL2                  // encoding: [0x40,0x23,0x3c,0xd5]
msr HDBSSBR_EL2, x0
// CHECK: msr HDBSSBR_EL2, x0                  // encoding: [0x40,0x23,0x1c,0xd5]

mrs x0, HDBSSPROD_EL2
// CHECK: mrs x0, HDBSSPROD_EL2                  // encoding: [0x60,0x23,0x3c,0xd5]
msr HDBSSPROD_EL2, x0
// CHECK: msr HDBSSPROD_EL2, x0                  // encoding: [0x60,0x23,0x1c,0xd5]

