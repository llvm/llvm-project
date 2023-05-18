// TRBE System registers
//
// RUN:     llvm-mc -triple aarch64 -show-encoding < %s      | FileCheck %s

// Read from system register
mrs x0, TRBLIMITR_EL1
mrs x0, TRBPTR_EL1
mrs x0, TRBBASER_EL1
mrs x0, TRBSR_EL1
mrs x0, TRBMAR_EL1
mrs x0, TRBTRG_EL1
mrs x0, TRBIDR_EL1

// CHECK: mrs x0, TRBLIMITR_EL1 // encoding: [0x00,0x9b,0x38,0xd5]
// CHECK: mrs x0, TRBPTR_EL1    // encoding: [0x20,0x9b,0x38,0xd5]
// CHECK: mrs x0, TRBBASER_EL1  // encoding: [0x40,0x9b,0x38,0xd5]
// CHECK: mrs x0, TRBSR_EL1     // encoding: [0x60,0x9b,0x38,0xd5]
// CHECK: mrs x0, TRBMAR_EL1    // encoding: [0x80,0x9b,0x38,0xd5]
// CHECK: mrs x0, TRBTRG_EL1    // encoding: [0xc0,0x9b,0x38,0xd5]
// CHECK: mrs x0, TRBIDR_EL1    // encoding: [0xe0,0x9b,0x38,0xd5]

// Write to system register
msr TRBLIMITR_EL1, x0
msr TRBPTR_EL1, x0
msr TRBBASER_EL1, x0
msr TRBSR_EL1, x0
msr TRBMAR_EL1, x0
msr TRBTRG_EL1, x0

// CHECK: msr TRBLIMITR_EL1, x0 // encoding: [0x00,0x9b,0x18,0xd5]
// CHECK: msr TRBPTR_EL1, x0    // encoding: [0x20,0x9b,0x18,0xd5]
// CHECK: msr TRBBASER_EL1, x0  // encoding: [0x40,0x9b,0x18,0xd5]
// CHECK: msr TRBSR_EL1, x0     // encoding: [0x60,0x9b,0x18,0xd5]
// CHECK: msr TRBMAR_EL1, x0    // encoding: [0x80,0x9b,0x18,0xd5]
// CHECK: msr TRBTRG_EL1, x0    // encoding: [0xc0,0x9b,0x18,0xd5]
