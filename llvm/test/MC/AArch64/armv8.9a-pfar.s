// RUN: llvm-mc -triple aarch64 -show-encoding < %s | FileCheck %s

mrs x0, PFAR_EL1
// CHECK: mrs x0, PFAR_EL1                    // encoding: [0xa0,0x60,0x38,0xd5]
msr PFAR_EL1, x0
// CHECK: msr PFAR_EL1, x0                    // encoding: [0xa0,0x60,0x18,0xd5]

mrs x0, PFAR_EL2
// CHECK: mrs x0, PFAR_EL2                    // encoding: [0xa0,0x60,0x3c,0xd5]
msr PFAR_EL2, x0
// CHECK: msr PFAR_EL2, x0                    // encoding: [0xa0,0x60,0x1c,0xd5]

mrs x0, PFAR_EL12
// CHECK: mrs x0, PFAR_EL12                   // encoding: [0xa0,0x60,0x3d,0xd5]
msr PFAR_EL12, x0
// CHECK: msr PFAR_EL12, x0                   // encoding: [0xa0,0x60,0x1d,0xd5]

mrs x0, MFAR_EL3
// CHECK: mrs x0, MFAR_EL3                    // encoding: [0xa0,0x60,0x3e,0xd5]
msr MFAR_EL3, x0
// CHECK: msr MFAR_EL3, x0                    // encoding: [0xa0,0x60,0x1e,0xd5]
