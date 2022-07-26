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
// Test 64-bit form (x0) and its aliases
// ---------------------------------------------------------------------------//

sqincw  x0
// CHECK-INST: sqincw  x0
// CHECK-ENCODING: [0xe0,0xf3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f3e0 <unknown>

sqincw  x0, all
// CHECK-INST: sqincw  x0
// CHECK-ENCODING: [0xe0,0xf3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f3e0 <unknown>

sqincw  x0, all, mul #1
// CHECK-INST: sqincw  x0
// CHECK-ENCODING: [0xe0,0xf3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f3e0 <unknown>

sqincw  x0, all, mul #16
// CHECK-INST: sqincw  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xf3,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04bff3e0 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (x0, w0) and its aliases
// ---------------------------------------------------------------------------//

sqincw  x0, w0
// CHECK-INST: sqincw  x0, w0
// CHECK-ENCODING: [0xe0,0xf3,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0f3e0 <unknown>

sqincw  x0, w0, all
// CHECK-INST: sqincw  x0, w0
// CHECK-ENCODING: [0xe0,0xf3,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0f3e0 <unknown>

sqincw  x0, w0, all, mul #1
// CHECK-INST: sqincw  x0, w0
// CHECK-ENCODING: [0xe0,0xf3,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0f3e0 <unknown>

sqincw  x0, w0, all, mul #16
// CHECK-INST: sqincw  x0, w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xf3,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04aff3e0 <unknown>

sqincw  x0, w0, pow2
// CHECK-INST: sqincw  x0, w0, pow2
// CHECK-ENCODING: [0x00,0xf0,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0f000 <unknown>

sqincw  x0, w0, pow2, mul #16
// CHECK-INST: sqincw  x0, w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xf0,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04aff000 <unknown>


// ---------------------------------------------------------------------------//
// Test vector form and aliases.
// ---------------------------------------------------------------------------//
sqincw  z0.s
// CHECK-INST: sqincw  z0.s
// CHECK-ENCODING: [0xe0,0xc3,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0c3e0 <unknown>

sqincw  z0.s, all
// CHECK-INST: sqincw  z0.s
// CHECK-ENCODING: [0xe0,0xc3,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0c3e0 <unknown>

sqincw  z0.s, all, mul #1
// CHECK-INST: sqincw  z0.s
// CHECK-ENCODING: [0xe0,0xc3,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0c3e0 <unknown>

sqincw  z0.s, all, mul #16
// CHECK-INST: sqincw  z0.s, all, mul #16
// CHECK-ENCODING: [0xe0,0xc3,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04afc3e0 <unknown>

sqincw  z0.s, pow2
// CHECK-INST: sqincw  z0.s, pow2
// CHECK-ENCODING: [0x00,0xc0,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0c000 <unknown>

sqincw  z0.s, pow2, mul #16
// CHECK-INST: sqincw  z0.s, pow2, mul #16
// CHECK-ENCODING: [0x00,0xc0,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04afc000 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

sqincw  x0, pow2
// CHECK-INST: sqincw  x0, pow2
// CHECK-ENCODING: [0x00,0xf0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f000 <unknown>

sqincw  x0, vl1
// CHECK-INST: sqincw  x0, vl1
// CHECK-ENCODING: [0x20,0xf0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f020 <unknown>

sqincw  x0, vl2
// CHECK-INST: sqincw  x0, vl2
// CHECK-ENCODING: [0x40,0xf0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f040 <unknown>

sqincw  x0, vl3
// CHECK-INST: sqincw  x0, vl3
// CHECK-ENCODING: [0x60,0xf0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f060 <unknown>

sqincw  x0, vl4
// CHECK-INST: sqincw  x0, vl4
// CHECK-ENCODING: [0x80,0xf0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f080 <unknown>

sqincw  x0, vl5
// CHECK-INST: sqincw  x0, vl5
// CHECK-ENCODING: [0xa0,0xf0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f0a0 <unknown>

sqincw  x0, vl6
// CHECK-INST: sqincw  x0, vl6
// CHECK-ENCODING: [0xc0,0xf0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f0c0 <unknown>

sqincw  x0, vl7
// CHECK-INST: sqincw  x0, vl7
// CHECK-ENCODING: [0xe0,0xf0,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f0e0 <unknown>

sqincw  x0, vl8
// CHECK-INST: sqincw  x0, vl8
// CHECK-ENCODING: [0x00,0xf1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f100 <unknown>

sqincw  x0, vl16
// CHECK-INST: sqincw  x0, vl16
// CHECK-ENCODING: [0x20,0xf1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f120 <unknown>

sqincw  x0, vl32
// CHECK-INST: sqincw  x0, vl32
// CHECK-ENCODING: [0x40,0xf1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f140 <unknown>

sqincw  x0, vl64
// CHECK-INST: sqincw  x0, vl64
// CHECK-ENCODING: [0x60,0xf1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f160 <unknown>

sqincw  x0, vl128
// CHECK-INST: sqincw  x0, vl128
// CHECK-ENCODING: [0x80,0xf1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f180 <unknown>

sqincw  x0, vl256
// CHECK-INST: sqincw  x0, vl256
// CHECK-ENCODING: [0xa0,0xf1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f1a0 <unknown>

sqincw  x0, #14
// CHECK-INST: sqincw  x0, #14
// CHECK-ENCODING: [0xc0,0xf1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f1c0 <unknown>

sqincw  x0, #15
// CHECK-INST: sqincw  x0, #15
// CHECK-ENCODING: [0xe0,0xf1,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f1e0 <unknown>

sqincw  x0, #16
// CHECK-INST: sqincw  x0, #16
// CHECK-ENCODING: [0x00,0xf2,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f200 <unknown>

sqincw  x0, #17
// CHECK-INST: sqincw  x0, #17
// CHECK-ENCODING: [0x20,0xf2,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f220 <unknown>

sqincw  x0, #18
// CHECK-INST: sqincw  x0, #18
// CHECK-ENCODING: [0x40,0xf2,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f240 <unknown>

sqincw  x0, #19
// CHECK-INST: sqincw  x0, #19
// CHECK-ENCODING: [0x60,0xf2,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f260 <unknown>

sqincw  x0, #20
// CHECK-INST: sqincw  x0, #20
// CHECK-ENCODING: [0x80,0xf2,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f280 <unknown>

sqincw  x0, #21
// CHECK-INST: sqincw  x0, #21
// CHECK-ENCODING: [0xa0,0xf2,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f2a0 <unknown>

sqincw  x0, #22
// CHECK-INST: sqincw  x0, #22
// CHECK-ENCODING: [0xc0,0xf2,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f2c0 <unknown>

sqincw  x0, #23
// CHECK-INST: sqincw  x0, #23
// CHECK-ENCODING: [0xe0,0xf2,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f2e0 <unknown>

sqincw  x0, #24
// CHECK-INST: sqincw  x0, #24
// CHECK-ENCODING: [0x00,0xf3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f300 <unknown>

sqincw  x0, #25
// CHECK-INST: sqincw  x0, #25
// CHECK-ENCODING: [0x20,0xf3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f320 <unknown>

sqincw  x0, #26
// CHECK-INST: sqincw  x0, #26
// CHECK-ENCODING: [0x40,0xf3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f340 <unknown>

sqincw  x0, #27
// CHECK-INST: sqincw  x0, #27
// CHECK-ENCODING: [0x60,0xf3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f360 <unknown>

sqincw  x0, #28
// CHECK-INST: sqincw  x0, #28
// CHECK-ENCODING: [0x80,0xf3,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f380 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

sqincw  z0.s
// CHECK-INST: sqincw	z0.s
// CHECK-ENCODING: [0xe0,0xc3,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0c3e0 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

sqincw  z0.s, pow2, mul #16
// CHECK-INST: sqincw	z0.s, pow2, mul #16
// CHECK-ENCODING: [0x00,0xc0,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04afc000 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

sqincw  z0.s, pow2
// CHECK-INST: sqincw	z0.s, pow2
// CHECK-ENCODING: [0x00,0xc0,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0c000 <unknown>
