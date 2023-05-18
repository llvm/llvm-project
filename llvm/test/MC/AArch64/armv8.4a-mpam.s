// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a < %s 2> %t | FileCheck %s --check-prefix=CHECK
// RUN: FileCheck --check-prefix=CHECK-RO < %t %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.4a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

//------------------------------------------------------------------------------
// ARMV8.4-A MPAM Extensions
//------------------------------------------------------------------------------

msr MPAM0_EL1, x0
msr MPAM1_EL1, x0
msr MPAM2_EL2, x0
msr MPAM3_EL3, x0
msr MPAM1_EL12, x0
msr MPAMHCR_EL2, x0
msr MPAMVPMV_EL2, x0
msr MPAMVPM0_EL2, x0
msr MPAMVPM1_EL2, x0
msr MPAMVPM2_EL2, x0
msr MPAMVPM3_EL2, x0
msr MPAMVPM4_EL2, x0
msr MPAMVPM5_EL2, x0
msr MPAMVPM6_EL2, x0
msr MPAMVPM7_EL2, x0
msr MPAMIDR_EL1, x0

mrs x0, MPAM0_EL1
mrs x0, MPAM1_EL1
mrs x0, MPAM2_EL2
mrs x0, MPAM3_EL3
mrs x0, MPAM1_EL12
mrs x0, MPAMHCR_EL2
mrs x0, MPAMVPMV_EL2
mrs x0, MPAMVPM0_EL2
mrs x0, MPAMVPM1_EL2
mrs x0, MPAMVPM2_EL2
mrs x0, MPAMVPM3_EL2
mrs x0, MPAMVPM4_EL2
mrs x0, MPAMVPM5_EL2
mrs x0, MPAMVPM6_EL2
mrs x0, MPAMVPM7_EL2
mrs x0, MPAMIDR_EL1

//CHECK:  msr MPAM0_EL1, x0           // encoding: [0x20,0xa5,0x18,0xd5]
//CHECK:  msr MPAM1_EL1, x0           // encoding: [0x00,0xa5,0x18,0xd5]
//CHECK:  msr MPAM2_EL2, x0           // encoding: [0x00,0xa5,0x1c,0xd5]
//CHECK:  msr MPAM3_EL3, x0           // encoding: [0x00,0xa5,0x1e,0xd5]
//CHECK:  msr MPAM1_EL12, x0          // encoding: [0x00,0xa5,0x1d,0xd5]
//CHECK:  msr MPAMHCR_EL2, x0         // encoding: [0x00,0xa4,0x1c,0xd5]
//CHECK:  msr MPAMVPMV_EL2, x0        // encoding: [0x20,0xa4,0x1c,0xd5]
//CHECK:  msr MPAMVPM0_EL2, x0        // encoding: [0x00,0xa6,0x1c,0xd5]
//CHECK:  msr MPAMVPM1_EL2, x0        // encoding: [0x20,0xa6,0x1c,0xd5]
//CHECK:  msr MPAMVPM2_EL2, x0        // encoding: [0x40,0xa6,0x1c,0xd5]
//CHECK:  msr MPAMVPM3_EL2, x0        // encoding: [0x60,0xa6,0x1c,0xd5]
//CHECK:  msr MPAMVPM4_EL2, x0        // encoding: [0x80,0xa6,0x1c,0xd5]
//CHECK:  msr MPAMVPM5_EL2, x0        // encoding: [0xa0,0xa6,0x1c,0xd5]
//CHECK:  msr MPAMVPM6_EL2, x0        // encoding: [0xc0,0xa6,0x1c,0xd5]
//CHECK:  msr MPAMVPM7_EL2, x0        // encoding: [0xe0,0xa6,0x1c,0xd5]

//CHECK-RO: error: expected writable system register or pstate
//CHECK-RO: msr MPAMIDR_EL1, x0
//CHECK-RO:     ^

//CHECK:  mrs x0, MPAM0_EL1           // encoding: [0x20,0xa5,0x38,0xd5]
//CHECK:  mrs x0, MPAM1_EL1           // encoding: [0x00,0xa5,0x38,0xd5]
//CHECK:  mrs x0, MPAM2_EL2           // encoding: [0x00,0xa5,0x3c,0xd5]
//CHECK:  mrs x0, MPAM3_EL3           // encoding: [0x00,0xa5,0x3e,0xd5]
//CHECK:  mrs x0, MPAM1_EL12          // encoding: [0x00,0xa5,0x3d,0xd5]
//CHECK:  mrs x0, MPAMHCR_EL2         // encoding: [0x00,0xa4,0x3c,0xd5]
//CHECK:  mrs x0, MPAMVPMV_EL2        // encoding: [0x20,0xa4,0x3c,0xd5]
//CHECK:  mrs x0, MPAMVPM0_EL2        // encoding: [0x00,0xa6,0x3c,0xd5]
//CHECK:  mrs x0, MPAMVPM1_EL2        // encoding: [0x20,0xa6,0x3c,0xd5]
//CHECK:  mrs x0, MPAMVPM2_EL2        // encoding: [0x40,0xa6,0x3c,0xd5]
//CHECK:  mrs x0, MPAMVPM3_EL2        // encoding: [0x60,0xa6,0x3c,0xd5]
//CHECK:  mrs x0, MPAMVPM4_EL2        // encoding: [0x80,0xa6,0x3c,0xd5]
//CHECK:  mrs x0, MPAMVPM5_EL2        // encoding: [0xa0,0xa6,0x3c,0xd5]
//CHECK:  mrs x0, MPAMVPM6_EL2        // encoding: [0xc0,0xa6,0x3c,0xd5]
//CHECK:  mrs x0, MPAMVPM7_EL2        // encoding: [0xe0,0xa6,0x3c,0xd5]
//CHECK:  mrs x0, MPAMIDR_EL1         // encoding: [0x80,0xa4,0x38,0xd5]

//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAM0_EL1, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAM1_EL1, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAM2_EL2, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAM3_EL3, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAM1_EL12, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAMHCR_EL2, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAMVPMV_EL2, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAMVPM0_EL2, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAMVPM1_EL2, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAMVPM2_EL2, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAMVPM3_EL2, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAMVPM4_EL2, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAMVPM5_EL2, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAMVPM6_EL2, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAMVPM7_EL2, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr MPAMIDR_EL1, x0
//CHECK-ERROR:     ^

//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAM0_EL1
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAM1_EL1
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAM2_EL2
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAM3_EL3
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAM1_EL12
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAMHCR_EL2
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAMVPMV_EL2
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAMVPM0_EL2
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAMVPM1_EL2
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAMVPM2_EL2
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAMVPM3_EL2
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAMVPM4_EL2
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAMVPM5_EL2
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAMVPM6_EL2
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAMVPM7_EL2
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, MPAMIDR_EL1
//CHECK-ERROR:         ^
