// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a < %s 2> %t  | FileCheck %s --check-prefix=CHECK
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.4a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-V84

//------------------------------------------------------------------------------
// ARMV8.4-A Enhanced Support for Nested Virtualization
//------------------------------------------------------------------------------

mrs x0, VNCR_EL2
msr VNCR_EL2, x0

// CHECK: mrs x0, VNCR_EL2    // encoding: [0x00,0x22,0x3c,0xd5]
// CHECK: msr VNCR_EL2, x0    // encoding: [0x00,0x22,0x1c,0xd5]

//CHECK-NO-V84:      error: expected readable system register
//CHECK-NO-V84-NEXT: mrs x0, VNCR_EL2
//CHECK-NO-V84-NEXT:         ^
//CHECK-NO-V84-NEXT: error: expected writable system register or pstate
//CHECK-NO-V84-NEXT: msr VNCR_EL2, x0
//CHECK-NO-V84-NEXT:     ^
