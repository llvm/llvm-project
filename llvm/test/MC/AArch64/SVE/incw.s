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

// ---------------------------------------------------------------------------//
// Test vector form and aliases.
// ---------------------------------------------------------------------------//

incw    z0.s
// CHECK-INST: incw    z0.s
// CHECK-ENCODING: [0xe0,0xc3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0c3e0 <unknown>

incw    z0.s, all
// CHECK-INST: incw    z0.s
// CHECK-ENCODING: [0xe0,0xc3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0c3e0 <unknown>

incw    z0.s, all, mul #1
// CHECK-INST: incw    z0.s
// CHECK-ENCODING: [0xe0,0xc3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0c3e0 <unknown>

incw    z0.s, all, mul #16
// CHECK-INST: incw    z0.s, all, mul #16
// CHECK-ENCODING: [0xe0,0xc3,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04bfc3e0 <unknown>


// ---------------------------------------------------------------------------//
// Test scalar form and aliases.
// ---------------------------------------------------------------------------//

incw    x0
// CHECK-INST: incw    x0
// CHECK-ENCODING: [0xe0,0xe3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e3e0 <unknown>

incw    x0, all
// CHECK-INST: incw    x0
// CHECK-ENCODING: [0xe0,0xe3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e3e0 <unknown>

incw    x0, all, mul #1
// CHECK-INST: incw    x0
// CHECK-ENCODING: [0xe0,0xe3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e3e0 <unknown>

incw    x0, all, mul #16
// CHECK-INST: incw    x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xe3,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04bfe3e0 <unknown>


// ---------------------------------------------------------------------------//
// Test predicate patterns
// ---------------------------------------------------------------------------//


incw    x0, pow2
// CHECK-INST: incw    x0, pow2
// CHECK-ENCODING: [0x00,0xe0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e000 <unknown>

incw    x0, vl1
// CHECK-INST: incw    x0, vl1
// CHECK-ENCODING: [0x20,0xe0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e020 <unknown>

incw    x0, vl2
// CHECK-INST: incw    x0, vl2
// CHECK-ENCODING: [0x40,0xe0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e040 <unknown>

incw    x0, vl3
// CHECK-INST: incw    x0, vl3
// CHECK-ENCODING: [0x60,0xe0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e060 <unknown>

incw    x0, vl4
// CHECK-INST: incw    x0, vl4
// CHECK-ENCODING: [0x80,0xe0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e080 <unknown>

incw    x0, vl5
// CHECK-INST: incw    x0, vl5
// CHECK-ENCODING: [0xa0,0xe0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e0a0 <unknown>

incw    x0, vl6
// CHECK-INST: incw    x0, vl6
// CHECK-ENCODING: [0xc0,0xe0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e0c0 <unknown>

incw    x0, vl7
// CHECK-INST: incw    x0, vl7
// CHECK-ENCODING: [0xe0,0xe0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e0e0 <unknown>

incw    x0, vl8
// CHECK-INST: incw    x0, vl8
// CHECK-ENCODING: [0x00,0xe1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e100 <unknown>

incw    x0, vl16
// CHECK-INST: incw    x0, vl16
// CHECK-ENCODING: [0x20,0xe1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e120 <unknown>

incw    x0, vl32
// CHECK-INST: incw    x0, vl32
// CHECK-ENCODING: [0x40,0xe1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e140 <unknown>

incw    x0, vl64
// CHECK-INST: incw    x0, vl64
// CHECK-ENCODING: [0x60,0xe1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e160 <unknown>

incw    x0, vl128
// CHECK-INST: incw    x0, vl128
// CHECK-ENCODING: [0x80,0xe1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e180 <unknown>

incw    x0, vl256
// CHECK-INST: incw    x0, vl256
// CHECK-ENCODING: [0xa0,0xe1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e1a0 <unknown>

incw    x0, #14
// CHECK-INST: incw    x0, #14
// CHECK-ENCODING: [0xc0,0xe1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e1c0 <unknown>

incw    x0, #28
// CHECK-INST: incw    x0, #28
// CHECK-ENCODING: [0x80,0xe3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e380 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

incw    z0.s
// CHECK-INST: incw	z0.s
// CHECK-ENCODING: [0xe0,0xc3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0c3e0 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

incw    z0.s, all, mul #16
// CHECK-INST: incw	z0.s, all, mul #16
// CHECK-ENCODING: [0xe0,0xc3,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04bfc3e0 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

incw    z0.s, all
// CHECK-INST: incw	z0.s
// CHECK-ENCODING: [0xe0,0xc3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0c3e0 <unknown>
