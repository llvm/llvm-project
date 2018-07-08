// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a < %s  | FileCheck %s --check-prefix=CHECK
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.4a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

//------------------------------------------------------------------------------
// ARMV8.4-A Debug, Trace and PMU Extensions
//------------------------------------------------------------------------------

msr TRFCR_EL1, x0
msr TRFCR_EL2, x0
msr TRFCR_EL12, x0

mrs x0, TRFCR_EL1
mrs x0, TRFCR_EL2
mrs x0, TRFCR_EL12

tsb csync

//CHECK:  msr TRFCR_EL1, x0           // encoding: [0x20,0x12,0x18,0xd5]
//CHECK:  msr TRFCR_EL2, x0           // encoding: [0x20,0x12,0x1c,0xd5]
//CHECK:  msr TRFCR_EL12, x0          // encoding: [0x20,0x12,0x1d,0xd5]

//CHECK:  mrs x0, TRFCR_EL1           // encoding: [0x20,0x12,0x38,0xd5]
//CHECK:  mrs x0, TRFCR_EL2           // encoding: [0x20,0x12,0x3c,0xd5]
//CHECK:  mrs x0, TRFCR_EL12          // encoding: [0x20,0x12,0x3d,0xd5]

//CHECK:  tsb csync                   // encoding: [0x5f,0x22,0x03,0xd5]

//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr TRFCR_EL1, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr TRFCR_EL2, x0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: msr TRFCR_EL12, x0
//CHECK-ERROR:     ^

//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, TRFCR_EL1
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, TRFCR_EL2
//CHECK-ERROR:         ^
//CHECK-ERROR: error: expected readable system register
//CHECK-ERROR: mrs x0, TRFCR_EL12
//CHECK-ERROR:         ^

//CHECK-ERROR: error: instruction requires: armv8.4a
