// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

eon     z5.b, z5.b, #0xf9
// CHECK-INST: eor     z5.b, z5.b, #0x6
// CHECK-ENCODING: [0x25,0x3e,0x40,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 25 3e 40 05 <unknown>

eon     z23.h, z23.h, #0xfff9
// CHECK-INST: eor     z23.h, z23.h, #0x6
// CHECK-ENCODING: [0x37,0x7c,0x40,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 37 7c 40 05 <unknown>

eon     z0.s, z0.s, #0xfffffff9
// CHECK-INST: eor     z0.s, z0.s, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x40,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f8 40 05 <unknown>

eon     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-INST: eor     z0.d, z0.d, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x43,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f8 43 05 <unknown>

eon     z5.b, z5.b, #0x6
// CHECK-INST: eor     z5.b, z5.b, #0xf9
// CHECK-ENCODING: [0xa5,0x2e,0x40,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5 2e 40 05 <unknown>

eon     z23.h, z23.h, #0x6
// CHECK-INST: eor     z23.h, z23.h, #0xfff9
// CHECK-ENCODING: [0xb7,0x6d,0x40,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 6d 40 05 <unknown>

eon     z0.s, z0.s, #0x6
// CHECK-INST: eor     z0.s, z0.s, #0xfffffff9
// CHECK-ENCODING: [0xa0,0xeb,0x40,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 eb 40 05 <unknown>

eon     z0.d, z0.d, #0x6
// CHECK-INST: eor     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-ENCODING: [0xa0,0xef,0x43,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 ef 43 05 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

eon     z0.d, z0.d, #0x6
// CHECK-INST: eor	z0.d, z0.d, #0xfffffffffffffff9
// CHECK-ENCODING: [0xa0,0xef,0x43,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 ef 43 05 <unknown>
