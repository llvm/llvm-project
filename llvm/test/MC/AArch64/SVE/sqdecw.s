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

sqdecw  x0
// CHECK-INST: sqdecw  x0
// CHECK-ENCODING: [0xe0,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fbe0 <unknown>

sqdecw  x0, all
// CHECK-INST: sqdecw  x0
// CHECK-ENCODING: [0xe0,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fbe0 <unknown>

sqdecw  x0, all, mul #1
// CHECK-INST: sqdecw  x0
// CHECK-ENCODING: [0xe0,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fbe0 <unknown>

sqdecw  x0, all, mul #16
// CHECK-INST: sqdecw  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xfb,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04bffbe0 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (x0, w0) and its aliases
// ---------------------------------------------------------------------------//

sqdecw  x0, w0
// CHECK-INST: sqdecw  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0fbe0 <unknown>

sqdecw  x0, w0, all
// CHECK-INST: sqdecw  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0fbe0 <unknown>

sqdecw  x0, w0, all, mul #1
// CHECK-INST: sqdecw  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0fbe0 <unknown>

sqdecw  x0, w0, all, mul #16
// CHECK-INST: sqdecw  x0, w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xfb,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04affbe0 <unknown>

sqdecw  x0, w0, pow2
// CHECK-INST: sqdecw  x0, w0, pow2
// CHECK-ENCODING: [0x00,0xf8,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0f800 <unknown>

sqdecw  x0, w0, pow2, mul #16
// CHECK-INST: sqdecw  x0, w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xf8,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04aff800 <unknown>


// ---------------------------------------------------------------------------//
// Test vector form and aliases.
// ---------------------------------------------------------------------------//
sqdecw  z0.s
// CHECK-INST: sqdecw  z0.s
// CHECK-ENCODING: [0xe0,0xcb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0cbe0 <unknown>

sqdecw  z0.s, all
// CHECK-INST: sqdecw  z0.s
// CHECK-ENCODING: [0xe0,0xcb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0cbe0 <unknown>

sqdecw  z0.s, all, mul #1
// CHECK-INST: sqdecw  z0.s
// CHECK-ENCODING: [0xe0,0xcb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0cbe0 <unknown>

sqdecw  z0.s, all, mul #16
// CHECK-INST: sqdecw  z0.s, all, mul #16
// CHECK-ENCODING: [0xe0,0xcb,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04afcbe0 <unknown>

sqdecw  z0.s, pow2
// CHECK-INST: sqdecw  z0.s, pow2
// CHECK-ENCODING: [0x00,0xc8,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0c800 <unknown>

sqdecw  z0.s, pow2, mul #16
// CHECK-INST: sqdecw  z0.s, pow2, mul #16
// CHECK-ENCODING: [0x00,0xc8,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04afc800 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

sqdecw  x0, pow2
// CHECK-INST: sqdecw  x0, pow2
// CHECK-ENCODING: [0x00,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f800 <unknown>

sqdecw  x0, vl1
// CHECK-INST: sqdecw  x0, vl1
// CHECK-ENCODING: [0x20,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f820 <unknown>

sqdecw  x0, vl2
// CHECK-INST: sqdecw  x0, vl2
// CHECK-ENCODING: [0x40,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f840 <unknown>

sqdecw  x0, vl3
// CHECK-INST: sqdecw  x0, vl3
// CHECK-ENCODING: [0x60,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f860 <unknown>

sqdecw  x0, vl4
// CHECK-INST: sqdecw  x0, vl4
// CHECK-ENCODING: [0x80,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f880 <unknown>

sqdecw  x0, vl5
// CHECK-INST: sqdecw  x0, vl5
// CHECK-ENCODING: [0xa0,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f8a0 <unknown>

sqdecw  x0, vl6
// CHECK-INST: sqdecw  x0, vl6
// CHECK-ENCODING: [0xc0,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f8c0 <unknown>

sqdecw  x0, vl7
// CHECK-INST: sqdecw  x0, vl7
// CHECK-ENCODING: [0xe0,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f8e0 <unknown>

sqdecw  x0, vl8
// CHECK-INST: sqdecw  x0, vl8
// CHECK-ENCODING: [0x00,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f900 <unknown>

sqdecw  x0, vl16
// CHECK-INST: sqdecw  x0, vl16
// CHECK-ENCODING: [0x20,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f920 <unknown>

sqdecw  x0, vl32
// CHECK-INST: sqdecw  x0, vl32
// CHECK-ENCODING: [0x40,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f940 <unknown>

sqdecw  x0, vl64
// CHECK-INST: sqdecw  x0, vl64
// CHECK-ENCODING: [0x60,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f960 <unknown>

sqdecw  x0, vl128
// CHECK-INST: sqdecw  x0, vl128
// CHECK-ENCODING: [0x80,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f980 <unknown>

sqdecw  x0, vl256
// CHECK-INST: sqdecw  x0, vl256
// CHECK-ENCODING: [0xa0,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f9a0 <unknown>

sqdecw  x0, #14
// CHECK-INST: sqdecw  x0, #14
// CHECK-ENCODING: [0xc0,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f9c0 <unknown>

sqdecw  x0, #15
// CHECK-INST: sqdecw  x0, #15
// CHECK-ENCODING: [0xe0,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0f9e0 <unknown>

sqdecw  x0, #16
// CHECK-INST: sqdecw  x0, #16
// CHECK-ENCODING: [0x00,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fa00 <unknown>

sqdecw  x0, #17
// CHECK-INST: sqdecw  x0, #17
// CHECK-ENCODING: [0x20,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fa20 <unknown>

sqdecw  x0, #18
// CHECK-INST: sqdecw  x0, #18
// CHECK-ENCODING: [0x40,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fa40 <unknown>

sqdecw  x0, #19
// CHECK-INST: sqdecw  x0, #19
// CHECK-ENCODING: [0x60,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fa60 <unknown>

sqdecw  x0, #20
// CHECK-INST: sqdecw  x0, #20
// CHECK-ENCODING: [0x80,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fa80 <unknown>

sqdecw  x0, #21
// CHECK-INST: sqdecw  x0, #21
// CHECK-ENCODING: [0xa0,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0faa0 <unknown>

sqdecw  x0, #22
// CHECK-INST: sqdecw  x0, #22
// CHECK-ENCODING: [0xc0,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fac0 <unknown>

sqdecw  x0, #23
// CHECK-INST: sqdecw  x0, #23
// CHECK-ENCODING: [0xe0,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fae0 <unknown>

sqdecw  x0, #24
// CHECK-INST: sqdecw  x0, #24
// CHECK-ENCODING: [0x00,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fb00 <unknown>

sqdecw  x0, #25
// CHECK-INST: sqdecw  x0, #25
// CHECK-ENCODING: [0x20,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fb20 <unknown>

sqdecw  x0, #26
// CHECK-INST: sqdecw  x0, #26
// CHECK-ENCODING: [0x40,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fb40 <unknown>

sqdecw  x0, #27
// CHECK-INST: sqdecw  x0, #27
// CHECK-ENCODING: [0x60,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fb60 <unknown>

sqdecw  x0, #28
// CHECK-INST: sqdecw  x0, #28
// CHECK-ENCODING: [0x80,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0fb80 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

sqdecw  z0.s
// CHECK-INST: sqdecw	z0.s
// CHECK-ENCODING: [0xe0,0xcb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0cbe0 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

sqdecw  z0.s, pow2, mul #16
// CHECK-INST: sqdecw	z0.s, pow2, mul #16
// CHECK-ENCODING: [0x00,0xc8,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04afc800 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

sqdecw  z0.s, pow2
// CHECK-INST: sqdecw	z0.s, pow2
// CHECK-ENCODING: [0x00,0xc8,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a0c800 <unknown>
