// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a < %s 2> %t  | FileCheck %s --check-prefix=CHECK
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.4a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-V84

//------------------------------------------------------------------------------
// ARMV8.4-A Timing insensitivity of data processing instructions
//------------------------------------------------------------------------------

msr DIT, #1
msr DIT, x0
mrs x0, DIT

//CHECK:      msr DIT, #1                 // encoding: [0x5f,0x41,0x03,0xd5]
//CHECK-NEXT: msr DIT, x0                 // encoding: [0xa0,0x42,0x1b,0xd5]
//CHECK-NEXT: mrs x0, DIT                 // encoding: [0xa0,0x42,0x3b,0xd5]

msr DIT, #16
msr DIT, #-1

//CHECK-ERROR:      error: immediate must be an integer in range [0, 15].
//CHECK-ERROR-NEXT: msr DIT, #16
//CHECK-ERROR-NEXT:          ^
//CHECK-ERROR-NEXT: error: immediate must be an integer in range [0, 15].
//CHECK-ERROR-NEXT: msr DIT, #-1
//CHECK-ERROR-NEXT:          ^

//CHECK-NO-V84:      error: expected writable system register or pstate
//CHECK-NO-V84-NEXT: msr DIT, #1
//CHECK-NO-V84-NEXT:     ^
//CHECK-NO-V84-NEXT: error: expected writable system register or pstate
//CHECK-NO-V84-NEXT: msr DIT, x0
//CHECK-NO-V84-NEXT:     ^
//CHECK-NO-V84-NEXT: error: expected readable system register
//CHECK-NO-V84-NEXT: mrs x0, DIT
//CHECK-NO-V84-NEXT:         ^
//CHECK-NO-V84-NEXT: error: expected writable system register or pstate
//CHECK-NO-V84-NEXT: msr DIT, #16
//CHECK-NO-V84-NEXT:     ^
//CHECK-NO-V84-NEXT: error: expected writable system register or pstate
//CHECK-NO-V84-NEXT: msr DIT, #-1
//CHECK-NO-V84-NEXT:     ^

