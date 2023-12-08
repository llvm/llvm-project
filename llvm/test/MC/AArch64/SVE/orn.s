// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

orn     z5.b, z5.b, #0xf9
// CHECK-INST: orr     z5.b, z5.b, #0x6
// CHECK-ENCODING: [0x25,0x3e,0x00,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05003e25 <unknown>

orn     z23.h, z23.h, #0xfff9
// CHECK-INST: orr     z23.h, z23.h, #0x6
// CHECK-ENCODING: [0x37,0x7c,0x00,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05007c37 <unknown>

orn     z0.s, z0.s, #0xfffffff9
// CHECK-INST: orr     z0.s, z0.s, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x00,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0500f820 <unknown>

orn     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-INST: orr     z0.d, z0.d, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x03,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0503f820 <unknown>

orn     z5.b, z5.b, #0x6
// CHECK-INST: orr     z5.b, z5.b, #0xf9
// CHECK-ENCODING: [0xa5,0x2e,0x00,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05002ea5 <unknown>

orn     z23.h, z23.h, #0x6
// CHECK-INST: orr     z23.h, z23.h, #0xfff9
// CHECK-ENCODING: [0xb7,0x6d,0x00,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05006db7 <unknown>

orn     z0.s, z0.s, #0x6
// CHECK-INST: orr     z0.s, z0.s, #0xfffffff9
// CHECK-ENCODING: [0xa0,0xeb,0x00,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0500eba0 <unknown>

orn     z0.d, z0.d, #0x6
// CHECK-INST: orr     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-ENCODING: [0xa0,0xef,0x03,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0503efa0 <unknown>

orn     p0.b, p0/z, p0.b, p0.b
// CHECK-INST: orn     p0.b, p0/z, p0.b, p0.b
// CHECK-ENCODING: [0x10,0x40,0x80,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25804010 <unknown>

orn     p15.b, p15/z, p15.b, p15.b
// CHECK-INST: orn     p15.b, p15/z, p15.b, p15.b
// CHECK-ENCODING: [0xff,0x7d,0x8f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 258f7dff <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

orn     z0.d, z0.d, #0x6
// CHECK-INST: orr	z0.d, z0.d, #0xfffffffffffffff9
// CHECK-ENCODING: [0xa0,0xef,0x03,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0503efa0 <unknown>
