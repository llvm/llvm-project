// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

lsr     z0.b, z0.b, #8
// CHECK-INST: lsr     z0.b, z0.b, #8
// CHECK-ENCODING: [0x00,0x94,0x28,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 94 28 04 <unknown>

lsr     z31.b, z31.b, #1
// CHECK-INST: lsr     z31.b, z31.b, #1
// CHECK-ENCODING: [0xff,0x97,0x2f,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 97 2f 04 <unknown>

lsr     z0.h, z0.h, #16
// CHECK-INST: lsr     z0.h, z0.h, #16
// CHECK-ENCODING: [0x00,0x94,0x30,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 94 30 04 <unknown>

lsr     z31.h, z31.h, #1
// CHECK-INST: lsr     z31.h, z31.h, #1
// CHECK-ENCODING: [0xff,0x97,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 97 3f 04 <unknown>

lsr     z0.s, z0.s, #32
// CHECK-INST: lsr     z0.s, z0.s, #32
// CHECK-ENCODING: [0x00,0x94,0x60,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 94 60 04 <unknown>

lsr     z31.s, z31.s, #1
// CHECK-INST: lsr     z31.s, z31.s, #1
// CHECK-ENCODING: [0xff,0x97,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 97 7f 04 <unknown>

lsr     z0.d, z0.d, #64
// CHECK-INST: lsr     z0.d, z0.d, #64
// CHECK-ENCODING: [0x00,0x94,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 94 a0 04 <unknown>

lsr     z31.d, z31.d, #1
// CHECK-INST: lsr     z31.d, z31.d, #1
// CHECK-ENCODING: [0xff,0x97,0xff,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 97 ff 04 <unknown>
