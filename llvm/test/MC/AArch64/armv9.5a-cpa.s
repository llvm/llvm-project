// RUN: llvm-mc -triple aarch64 -show-encoding -mattr=+cpa < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64 < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-CPA %s

addpt x0, x1, x2
// CHECK: addpt x0, x1, x2               // encoding: [0x20,0x20,0x02,0x9a]
// ERROR-NO-CPA: error: instruction requires: cpa

addpt sp, sp, x2
// CHECK: addpt sp, sp, x2               // encoding: [0xff,0x23,0x02,0x9a]
// ERROR-NO-CPA: error: instruction requires: cpa

addpt x0, x1, x2, lsl #0
// CHECK: addpt x0, x1, x2               // encoding: [0x20,0x20,0x02,0x9a]
// ERROR-NO-CPA: error: instruction requires: cpa

addpt x0, x1, x2, lsl #7
// CHECK: addpt x0, x1, x2, lsl #7       // encoding: [0x20,0x3c,0x02,0x9a]
// ERROR-NO-CPA: error: instruction requires: cpa

addpt sp, sp, x2, lsl #7
// CHECK: addpt sp, sp, x2, lsl #7       // encoding: [0xff,0x3f,0x02,0x9a]
// ERROR-NO-CPA: error: instruction requires: cpa

subpt x0, x1, x2
// CHECK: subpt x0, x1, x2               // encoding: [0x20,0x20,0x02,0xda]
// ERROR-NO-CPA: error: instruction requires: cpa

subpt sp, sp, x2
// CHECK: subpt sp, sp, x2               // encoding: [0xff,0x23,0x02,0xda]
// ERROR-NO-CPA: error: instruction requires: cpa

subpt x0, x1, x2, lsl #0
// CHECK: subpt x0, x1, x2               // encoding: [0x20,0x20,0x02,0xda]
// ERROR-NO-CPA: error: instruction requires: cpa

subpt x0, x1, x2, lsl #7
// CHECK: subpt x0, x1, x2, lsl #7       // encoding: [0x20,0x3c,0x02,0xda]
// ERROR-NO-CPA: error: instruction requires: cpa

subpt sp, sp, x2, lsl #7
// CHECK: subpt sp, sp, x2, lsl #7       // encoding: [0xff,0x3f,0x02,0xda]
// ERROR-NO-CPA: error: instruction requires: cpa

maddpt x0, x1, x2, x3
// CHECK: maddpt x0, x1, x2, x3          // encoding: [0x20,0x0c,0x62,0x9b]
// ERROR-NO-CPA: error: instruction requires: cpa

msubpt x0, x1, x2, x3
// CHECK: msubpt x0, x1, x2, x3          // encoding: [0x20,0x8c,0x62,0x9b]
// ERROR-NO-CPA: error: instruction requires: cpa
