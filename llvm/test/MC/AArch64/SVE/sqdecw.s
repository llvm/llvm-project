// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// ---------------------------------------------------------------------------//
// Test 64-bit form (x0) and its aliases
// ---------------------------------------------------------------------------//

sqdecw  x0
// CHECK-INST: sqdecw  x0
// CHECK-ENCODING: [0xe0,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 fb b0 04 <unknown>

sqdecw  x0, all
// CHECK-INST: sqdecw  x0
// CHECK-ENCODING: [0xe0,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 fb b0 04 <unknown>

sqdecw  x0, all, mul #1
// CHECK-INST: sqdecw  x0
// CHECK-ENCODING: [0xe0,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 fb b0 04 <unknown>

sqdecw  x0, all, mul #16
// CHECK-INST: sqdecw  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xfb,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 fb bf 04 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (x0, w0) and its aliases
// ---------------------------------------------------------------------------//

sqdecw  x0, w0
// CHECK-INST: sqdecw  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 fb a0 04 <unknown>

sqdecw  x0, w0, all
// CHECK-INST: sqdecw  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 fb a0 04 <unknown>

sqdecw  x0, w0, all, mul #1
// CHECK-INST: sqdecw  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 fb a0 04 <unknown>

sqdecw  x0, w0, all, mul #16
// CHECK-INST: sqdecw  x0, w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xfb,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 fb af 04 <unknown>

sqdecw  x0, w0, pow2
// CHECK-INST: sqdecw  x0, w0, pow2
// CHECK-ENCODING: [0x00,0xf8,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f8 a0 04 <unknown>

sqdecw  x0, w0, pow2, mul #16
// CHECK-INST: sqdecw  x0, w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xf8,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f8 af 04 <unknown>


// ---------------------------------------------------------------------------//
// Test vector form and aliases.
// ---------------------------------------------------------------------------//
sqdecw  z0.s
// CHECK-INST: sqdecw  z0.s
// CHECK-ENCODING: [0xe0,0xcb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 cb a0 04 <unknown>

sqdecw  z0.s, all
// CHECK-INST: sqdecw  z0.s
// CHECK-ENCODING: [0xe0,0xcb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 cb a0 04 <unknown>

sqdecw  z0.s, all, mul #1
// CHECK-INST: sqdecw  z0.s
// CHECK-ENCODING: [0xe0,0xcb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 cb a0 04 <unknown>

sqdecw  z0.s, all, mul #16
// CHECK-INST: sqdecw  z0.s, all, mul #16
// CHECK-ENCODING: [0xe0,0xcb,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 cb af 04 <unknown>

sqdecw  z0.s, pow2
// CHECK-INST: sqdecw  z0.s, pow2
// CHECK-ENCODING: [0x00,0xc8,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c8 a0 04 <unknown>

sqdecw  z0.s, pow2, mul #16
// CHECK-INST: sqdecw  z0.s, pow2, mul #16
// CHECK-ENCODING: [0x00,0xc8,0xaf,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c8 af 04 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

sqdecw  x0, pow2
// CHECK-INST: sqdecw  x0, pow2
// CHECK-ENCODING: [0x00,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f8 b0 04 <unknown>

sqdecw  x0, vl1
// CHECK-INST: sqdecw  x0, vl1
// CHECK-ENCODING: [0x20,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f8 b0 04 <unknown>

sqdecw  x0, vl2
// CHECK-INST: sqdecw  x0, vl2
// CHECK-ENCODING: [0x40,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 f8 b0 04 <unknown>

sqdecw  x0, vl3
// CHECK-INST: sqdecw  x0, vl3
// CHECK-ENCODING: [0x60,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 f8 b0 04 <unknown>

sqdecw  x0, vl4
// CHECK-INST: sqdecw  x0, vl4
// CHECK-ENCODING: [0x80,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 f8 b0 04 <unknown>

sqdecw  x0, vl5
// CHECK-INST: sqdecw  x0, vl5
// CHECK-ENCODING: [0xa0,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 f8 b0 04 <unknown>

sqdecw  x0, vl6
// CHECK-INST: sqdecw  x0, vl6
// CHECK-ENCODING: [0xc0,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 f8 b0 04 <unknown>

sqdecw  x0, vl7
// CHECK-INST: sqdecw  x0, vl7
// CHECK-ENCODING: [0xe0,0xf8,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f8 b0 04 <unknown>

sqdecw  x0, vl8
// CHECK-INST: sqdecw  x0, vl8
// CHECK-ENCODING: [0x00,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f9 b0 04 <unknown>

sqdecw  x0, vl16
// CHECK-INST: sqdecw  x0, vl16
// CHECK-ENCODING: [0x20,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f9 b0 04 <unknown>

sqdecw  x0, vl32
// CHECK-INST: sqdecw  x0, vl32
// CHECK-ENCODING: [0x40,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 f9 b0 04 <unknown>

sqdecw  x0, vl64
// CHECK-INST: sqdecw  x0, vl64
// CHECK-ENCODING: [0x60,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 f9 b0 04 <unknown>

sqdecw  x0, vl128
// CHECK-INST: sqdecw  x0, vl128
// CHECK-ENCODING: [0x80,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 f9 b0 04 <unknown>

sqdecw  x0, vl256
// CHECK-INST: sqdecw  x0, vl256
// CHECK-ENCODING: [0xa0,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 f9 b0 04 <unknown>

sqdecw  x0, #14
// CHECK-INST: sqdecw  x0, #14
// CHECK-ENCODING: [0xc0,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 f9 b0 04 <unknown>

sqdecw  x0, #15
// CHECK-INST: sqdecw  x0, #15
// CHECK-ENCODING: [0xe0,0xf9,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f9 b0 04 <unknown>

sqdecw  x0, #16
// CHECK-INST: sqdecw  x0, #16
// CHECK-ENCODING: [0x00,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 fa b0 04 <unknown>

sqdecw  x0, #17
// CHECK-INST: sqdecw  x0, #17
// CHECK-ENCODING: [0x20,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 fa b0 04 <unknown>

sqdecw  x0, #18
// CHECK-INST: sqdecw  x0, #18
// CHECK-ENCODING: [0x40,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 fa b0 04 <unknown>

sqdecw  x0, #19
// CHECK-INST: sqdecw  x0, #19
// CHECK-ENCODING: [0x60,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 fa b0 04 <unknown>

sqdecw  x0, #20
// CHECK-INST: sqdecw  x0, #20
// CHECK-ENCODING: [0x80,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 fa b0 04 <unknown>

sqdecw  x0, #21
// CHECK-INST: sqdecw  x0, #21
// CHECK-ENCODING: [0xa0,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 fa b0 04 <unknown>

sqdecw  x0, #22
// CHECK-INST: sqdecw  x0, #22
// CHECK-ENCODING: [0xc0,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 fa b0 04 <unknown>

sqdecw  x0, #23
// CHECK-INST: sqdecw  x0, #23
// CHECK-ENCODING: [0xe0,0xfa,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 fa b0 04 <unknown>

sqdecw  x0, #24
// CHECK-INST: sqdecw  x0, #24
// CHECK-ENCODING: [0x00,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 fb b0 04 <unknown>

sqdecw  x0, #25
// CHECK-INST: sqdecw  x0, #25
// CHECK-ENCODING: [0x20,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 fb b0 04 <unknown>

sqdecw  x0, #26
// CHECK-INST: sqdecw  x0, #26
// CHECK-ENCODING: [0x40,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 fb b0 04 <unknown>

sqdecw  x0, #27
// CHECK-INST: sqdecw  x0, #27
// CHECK-ENCODING: [0x60,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 fb b0 04 <unknown>

sqdecw  x0, #28
// CHECK-INST: sqdecw  x0, #28
// CHECK-ENCODING: [0x80,0xfb,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 fb b0 04 <unknown>
