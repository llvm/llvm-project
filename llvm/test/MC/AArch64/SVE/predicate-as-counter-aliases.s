// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p1 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve2p1 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p1 < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN


ldr     pn0, [x0]
// CHECK-INST: ldr     p0, [x0]
// CHECK-ENCODING: [0x00,0x00,0x80,0x85]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 85800000 <unknown>

ldr     pn5, [x10, #255, mul vl]
// CHECK-INST: ldr     p5, [x10, #255, mul vl]
// CHECK-ENCODING: [0x45,0x1d,0x9f,0x85]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 859f1d45 <unknown>


str     pn0, [x0]
// CHECK-INST: str     p0, [x0]
// CHECK-ENCODING: [0x00,0x00,0x80,0xe5]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: e5800000 <unknown>

str     pn5, [x10, #255, mul vl]
// CHECK-INST: str     p5, [x10, #255, mul vl]
// CHECK-ENCODING: [0x45,0x1d,0x9f,0xe5]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: e59f1d45 <unknown>


mov     pn0.b, pn0.b
// CHECK-INST: mov     p0.b, p0.b
// CHECK-ENCODING: [0x00,0x40,0x80,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25804000 <unknown>


pfalse pn15.b
// CHECK-INST: pfalse  p15.b
// CHECK-ENCODING: [0x0f,0xe4,0x18,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 2518e40f <unknown>
