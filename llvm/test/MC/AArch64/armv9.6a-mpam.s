// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding < %s 2> %t | FileCheck %s --check-prefix=CHECK
// RUN: FileCheck --check-prefix=CHECK-RO < %t %s

//------------------------------------------------------------------------------
// Armv9.6-A FEAT_MPAM Extensions
//------------------------------------------------------------------------------

msr MPAMBWIDR_EL1, x0
msr MPAMBW3_EL3, x0
msr MPAMBW2_EL2, x0
msr MPAMBW1_EL1, x0
msr MPAMBW1_EL12, x0
msr MPAMBW0_EL1, x0
msr MPAMBWCAP_EL2, x0
msr MPAMBWSM_EL1, x0

mrs x0, MPAMBWIDR_EL1
mrs x0, MPAMBW3_EL3
mrs x0, MPAMBW2_EL2
mrs x0, MPAMBW1_EL1
mrs x0, MPAMBW1_EL12
mrs x0, MPAMBW0_EL1
mrs x0, MPAMBWCAP_EL2
mrs x0, MPAMBWSM_EL1

//CHECK: msr     MPAMBW3_EL3, x0                 // encoding: [0x80,0xa5,0x1e,0xd5]
//CHECK: msr     MPAMBW2_EL2, x0                 // encoding: [0x80,0xa5,0x1c,0xd5]
//CHECK: msr     MPAMBW1_EL1, x0                 // encoding: [0x80,0xa5,0x18,0xd5]
//CHECK: msr     MPAMBW1_EL12, x0                // encoding: [0x80,0xa5,0x1d,0xd5]
//CHECK: msr     MPAMBW0_EL1, x0                 // encoding: [0xa0,0xa5,0x18,0xd5]
//CHECK: msr     MPAMBWCAP_EL2, x0               // encoding: [0xc0,0xa5,0x1c,0xd5]
//CHECK: msr     MPAMBWSM_EL1, x0                // encoding: [0xe0,0xa5,0x18,0xd5]

//CHECK-RO: error: expected writable system register or pstate
//CHECK-RO: msr MPAMBWIDR_EL1, x0
//CHECK-RO:     ^

//CHECK: mrs     x0, MPAMBWIDR_EL1               // encoding: [0xa0,0xa4,0x38,0xd5]
//CHECK: mrs     x0, MPAMBW3_EL3                 // encoding: [0x80,0xa5,0x3e,0xd5]
//CHECK: mrs     x0, MPAMBW2_EL2                 // encoding: [0x80,0xa5,0x3c,0xd5]
//CHECK: mrs     x0, MPAMBW1_EL1                 // encoding: [0x80,0xa5,0x38,0xd5]
//CHECK: mrs     x0, MPAMBW1_EL12                // encoding: [0x80,0xa5,0x3d,0xd5]
//CHECK: mrs     x0, MPAMBW0_EL1                 // encoding: [0xa0,0xa5,0x38,0xd5]
//CHECK: mrs     x0, MPAMBWCAP_EL2               // encoding: [0xc0,0xa5,0x3c,0xd5]
//CHECK: mrs     x0, MPAMBWSM_EL1                // encoding: [0xe0,0xa5,0x38,0xd5]
