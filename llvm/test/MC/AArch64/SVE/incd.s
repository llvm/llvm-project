// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// ---------------------------------------------------------------------------//
// Test vector form and aliases.
// ---------------------------------------------------------------------------//

incd    z0.d
// CHECK-INST: incd    z0.d
// CHECK-ENCODING: [0xe0,0xc3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 c3 f0 04 <unknown>

incd    z0.d, all
// CHECK-INST: incd    z0.d
// CHECK-ENCODING: [0xe0,0xc3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 c3 f0 04 <unknown>

incd    z0.d, all, mul #1
// CHECK-INST: incd    z0.d
// CHECK-ENCODING: [0xe0,0xc3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 c3 f0 04 <unknown>

incd    z0.d, all, mul #16
// CHECK-INST: incd    z0.d, all, mul #16
// CHECK-ENCODING: [0xe0,0xc3,0xff,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 c3 ff 04 <unknown>


// ---------------------------------------------------------------------------//
// Test scalar form and aliases.
// ---------------------------------------------------------------------------//

incd    x0
// CHECK-INST: incd    x0
// CHECK-ENCODING: [0xe0,0xe3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 e3 f0 04 <unknown>

incd    x0, all
// CHECK-INST: incd    x0
// CHECK-ENCODING: [0xe0,0xe3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 e3 f0 04 <unknown>

incd    x0, all, mul #1
// CHECK-INST: incd    x0
// CHECK-ENCODING: [0xe0,0xe3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 e3 f0 04 <unknown>

incd    x0, all, mul #16
// CHECK-INST: incd    x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xe3,0xff,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 e3 ff 04 <unknown>


// ---------------------------------------------------------------------------//
// Test predicate patterns
// ---------------------------------------------------------------------------//

incd    x0, pow2
// CHECK-INST: incd    x0, pow2
// CHECK-ENCODING: [0x00,0xe0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 f0 04 <unknown>

incd    x0, vl1
// CHECK-INST: incd    x0, vl1
// CHECK-ENCODING: [0x20,0xe0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 e0 f0 04 <unknown>

incd    x0, vl2
// CHECK-INST: incd    x0, vl2
// CHECK-ENCODING: [0x40,0xe0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 e0 f0 04 <unknown>

incd    x0, vl3
// CHECK-INST: incd    x0, vl3
// CHECK-ENCODING: [0x60,0xe0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 e0 f0 04 <unknown>

incd    x0, vl4
// CHECK-INST: incd    x0, vl4
// CHECK-ENCODING: [0x80,0xe0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 e0 f0 04 <unknown>

incd    x0, vl5
// CHECK-INST: incd    x0, vl5
// CHECK-ENCODING: [0xa0,0xe0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 e0 f0 04 <unknown>

incd    x0, vl6
// CHECK-INST: incd    x0, vl6
// CHECK-ENCODING: [0xc0,0xe0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 e0 f0 04 <unknown>

incd    x0, vl7
// CHECK-INST: incd    x0, vl7
// CHECK-ENCODING: [0xe0,0xe0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 e0 f0 04 <unknown>

incd    x0, vl8
// CHECK-INST: incd    x0, vl8
// CHECK-ENCODING: [0x00,0xe1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e1 f0 04 <unknown>

incd    x0, vl16
// CHECK-INST: incd    x0, vl16
// CHECK-ENCODING: [0x20,0xe1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 e1 f0 04 <unknown>

incd    x0, vl32
// CHECK-INST: incd    x0, vl32
// CHECK-ENCODING: [0x40,0xe1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 e1 f0 04 <unknown>

incd    x0, vl64
// CHECK-INST: incd    x0, vl64
// CHECK-ENCODING: [0x60,0xe1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 e1 f0 04 <unknown>

incd    x0, vl128
// CHECK-INST: incd    x0, vl128
// CHECK-ENCODING: [0x80,0xe1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 e1 f0 04 <unknown>

incd    x0, vl256
// CHECK-INST: incd    x0, vl256
// CHECK-ENCODING: [0xa0,0xe1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 e1 f0 04 <unknown>

incd    x0, #14
// CHECK-INST: incd    x0, #14
// CHECK-ENCODING: [0xc0,0xe1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 e1 f0 04 <unknown>

incd    x0, #28
// CHECK-INST: incd    x0, #28
// CHECK-ENCODING: [0x80,0xe3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 e3 f0 04 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

incd    z0.d
// CHECK-INST: incd	z0.d
// CHECK-ENCODING: [0xe0,0xc3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 c3 f0 04 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

incd    z0.d, all, mul #16
// CHECK-INST: incd	z0.d, all, mul #16
// CHECK-ENCODING: [0xe0,0xc3,0xff,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 c3 ff 04 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

incd    z0.d, all
// CHECK-INST: incd	z0.d
// CHECK-ENCODING: [0xe0,0xc3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 c3 f0 04 <unknown>
