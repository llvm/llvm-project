// Tests for transaction memory extension instructions
//
// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+tme   < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=-tme   < %s 2>&1 | FileCheck %s --check-prefix=NOTME

tstart x3
ttest  x4
tcommit
tcancel #0x1234

// CHECK: tstart x3         // encoding: [0x63,0x30,0x23,0xd5]
// CHECK: ttest x4          // encoding: [0x64,0x31,0x23,0xd5]
// CHECK: tcommit           // encoding: [0x7f,0x30,0x03,0xd5]
// CHECK: tcancel #0x1234   // encoding: [0x80,0x46,0x62,0xd4]


// NOTME: instruction requires: tme
// NOTME-NEXT: tstart x3
// NOTME: instruction requires: tme
// NOTME-NEXT: ttest  x4
// NOTME: instruction requires: tme
// NOTME-NEXT: tcommit
// NOTME: instruction requires: tme
// NOTME-NEXT: tcancel #0x1234
