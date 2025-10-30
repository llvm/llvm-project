// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+cpa < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+cpa < %s \
// RUN:        | llvm-objdump -d --mattr=+cpa - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+cpa < %s \
// RUN:        | llvm-objdump -d --mattr=-cpa - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+cpa < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+cpa -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



addpt x0, x1, x2
// CHECK-INST: addpt x0, x1, x2
// CHECK-ENCODING: encoding: [0x20,0x20,0x02,0x9a]
// CHECK-ERROR: error: instruction requires: cpa
// CHECK-UNKNOWN:  9a022020 <unknown>

addpt sp, sp, x2
// CHECK-INST: addpt sp, sp, x2
// CHECK-ENCODING: encoding: [0xff,0x23,0x02,0x9a]
// CHECK-ERROR: error: instruction requires: cpa
// CHECK-UNKNOWN:  9a0223ff <unknown>

addpt x0, x1, x2, lsl #0
// CHECK-INST: addpt x0, x1, x2
// CHECK-ENCODING: encoding: [0x20,0x20,0x02,0x9a]
// CHECK-ERROR: error: instruction requires: cpa
// CHECK-UNKNOWN:  9a022020 <unknown>

addpt x0, x1, x2, lsl #7
// CHECK-INST: addpt x0, x1, x2, lsl #7
// CHECK-ENCODING: encoding: [0x20,0x3c,0x02,0x9a]
// CHECK-ERROR: error: instruction requires: cpa
// CHECK-UNKNOWN:  9a023c20 <unknown>

addpt sp, sp, x2, lsl #7
// CHECK-INST: addpt sp, sp, x2, lsl #7
// CHECK-ENCODING: encoding: [0xff,0x3f,0x02,0x9a]
// CHECK-ERROR: error: instruction requires: cpa
// CHECK-UNKNOWN:  9a023fff <unknown>

subpt x0, x1, x2
// CHECK-INST: subpt x0, x1, x2
// CHECK-ENCODING: encoding: [0x20,0x20,0x02,0xda]
// CHECK-ERROR: error: instruction requires: cpa
// CHECK-UNKNOWN:  da022020 <unknown>

subpt sp, sp, x2
// CHECK-INST: subpt sp, sp, x2
// CHECK-ENCODING: encoding: [0xff,0x23,0x02,0xda]
// CHECK-ERROR: error: instruction requires: cpa
// CHECK-UNKNOWN:  da0223ff <unknown>

subpt x0, x1, x2, lsl #0
// CHECK-INST: subpt x0, x1, x2
// CHECK-ENCODING: encoding: [0x20,0x20,0x02,0xda]
// CHECK-ERROR: error: instruction requires: cpa
// CHECK-UNKNOWN:  da022020 <unknown>

subpt x0, x1, x2, lsl #7
// CHECK-INST: subpt x0, x1, x2, lsl #7
// CHECK-ENCODING: encoding: [0x20,0x3c,0x02,0xda]
// CHECK-ERROR: error: instruction requires: cpa
// CHECK-UNKNOWN:  da023c20 <unknown>

subpt sp, sp, x2, lsl #7
// CHECK-INST: subpt sp, sp, x2, lsl #7
// CHECK-ENCODING: encoding: [0xff,0x3f,0x02,0xda]
// CHECK-ERROR: error: instruction requires: cpa
// CHECK-UNKNOWN:  da023fff <unknown>

maddpt x0, x1, x2, x3
// CHECK-INST: maddpt x0, x1, x2, x3
// CHECK-ENCODING: encoding: [0x20,0x0c,0x62,0x9b]
// CHECK-ERROR: error: instruction requires: cpa
// CHECK-UNKNOWN:  9b620c20 <unknown>

msubpt x0, x1, x2, x3
// CHECK-INST: msubpt x0, x1, x2, x3
// CHECK-ENCODING: encoding: [0x20,0x8c,0x62,0x9b]
// CHECK-ERROR: error: instruction requires: cpa
// CHECK-UNKNOWN:  9b628c20 <unknown>
