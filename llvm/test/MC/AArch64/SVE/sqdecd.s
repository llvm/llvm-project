// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// ---------------------------------------------------------------------------//
// Test 64-bit form (x0) and its aliases
// ---------------------------------------------------------------------------//

sqdecd  x0
// CHECK-INST: sqdecd  x0
// CHECK-ENCODING: [0xe0,0xfb,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fb f0 04 <unknown>

sqdecd  x0, all
// CHECK-INST: sqdecd  x0
// CHECK-ENCODING: [0xe0,0xfb,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fb f0 04 <unknown>

sqdecd  x0, all, mul #1
// CHECK-INST: sqdecd  x0
// CHECK-ENCODING: [0xe0,0xfb,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fb f0 04 <unknown>

sqdecd  x0, all, mul #16
// CHECK-INST: sqdecd  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xfb,0xff,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fb ff 04 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (x0, w0) and its aliases
// ---------------------------------------------------------------------------//

sqdecd  x0, w0
// CHECK-INST: sqdecd  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fb e0 04 <unknown>

sqdecd  x0, w0, all
// CHECK-INST: sqdecd  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fb e0 04 <unknown>

sqdecd  x0, w0, all, mul #1
// CHECK-INST: sqdecd  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fb e0 04 <unknown>

sqdecd  x0, w0, all, mul #16
// CHECK-INST: sqdecd  x0, w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xfb,0xef,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fb ef 04 <unknown>

sqdecd  x0, w0, pow2
// CHECK-INST: sqdecd  x0, w0, pow2
// CHECK-ENCODING: [0x00,0xf8,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 f8 e0 04 <unknown>

sqdecd  x0, w0, pow2, mul #16
// CHECK-INST: sqdecd  x0, w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xf8,0xef,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 f8 ef 04 <unknown>


// ---------------------------------------------------------------------------//
// Test vector form and aliases.
// ---------------------------------------------------------------------------//
sqdecd  z0.d
// CHECK-INST: sqdecd  z0.d
// CHECK-ENCODING: [0xe0,0xcb,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 cb e0 04 <unknown>

sqdecd  z0.d, all
// CHECK-INST: sqdecd  z0.d
// CHECK-ENCODING: [0xe0,0xcb,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 cb e0 04 <unknown>

sqdecd  z0.d, all, mul #1
// CHECK-INST: sqdecd  z0.d
// CHECK-ENCODING: [0xe0,0xcb,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 cb e0 04 <unknown>

sqdecd  z0.d, all, mul #16
// CHECK-INST: sqdecd  z0.d, all, mul #16
// CHECK-ENCODING: [0xe0,0xcb,0xef,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 cb ef 04 <unknown>

sqdecd  z0.d, pow2
// CHECK-INST: sqdecd  z0.d, pow2
// CHECK-ENCODING: [0x00,0xc8,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 c8 e0 04 <unknown>

sqdecd  z0.d, pow2, mul #16
// CHECK-INST: sqdecd  z0.d, pow2, mul #16
// CHECK-ENCODING: [0x00,0xc8,0xef,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 c8 ef 04 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

sqdecd  x0, pow2
// CHECK-INST: sqdecd  x0, pow2
// CHECK-ENCODING: [0x00,0xf8,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 f8 f0 04 <unknown>

sqdecd  x0, vl1
// CHECK-INST: sqdecd  x0, vl1
// CHECK-ENCODING: [0x20,0xf8,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 f8 f0 04 <unknown>

sqdecd  x0, vl2
// CHECK-INST: sqdecd  x0, vl2
// CHECK-ENCODING: [0x40,0xf8,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 f8 f0 04 <unknown>

sqdecd  x0, vl3
// CHECK-INST: sqdecd  x0, vl3
// CHECK-ENCODING: [0x60,0xf8,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 f8 f0 04 <unknown>

sqdecd  x0, vl4
// CHECK-INST: sqdecd  x0, vl4
// CHECK-ENCODING: [0x80,0xf8,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 f8 f0 04 <unknown>

sqdecd  x0, vl5
// CHECK-INST: sqdecd  x0, vl5
// CHECK-ENCODING: [0xa0,0xf8,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: a0 f8 f0 04 <unknown>

sqdecd  x0, vl6
// CHECK-INST: sqdecd  x0, vl6
// CHECK-ENCODING: [0xc0,0xf8,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c0 f8 f0 04 <unknown>

sqdecd  x0, vl7
// CHECK-INST: sqdecd  x0, vl7
// CHECK-ENCODING: [0xe0,0xf8,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 f8 f0 04 <unknown>

sqdecd  x0, vl8
// CHECK-INST: sqdecd  x0, vl8
// CHECK-ENCODING: [0x00,0xf9,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 f9 f0 04 <unknown>

sqdecd  x0, vl16
// CHECK-INST: sqdecd  x0, vl16
// CHECK-ENCODING: [0x20,0xf9,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 f9 f0 04 <unknown>

sqdecd  x0, vl32
// CHECK-INST: sqdecd  x0, vl32
// CHECK-ENCODING: [0x40,0xf9,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 f9 f0 04 <unknown>

sqdecd  x0, vl64
// CHECK-INST: sqdecd  x0, vl64
// CHECK-ENCODING: [0x60,0xf9,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 f9 f0 04 <unknown>

sqdecd  x0, vl128
// CHECK-INST: sqdecd  x0, vl128
// CHECK-ENCODING: [0x80,0xf9,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 f9 f0 04 <unknown>

sqdecd  x0, vl256
// CHECK-INST: sqdecd  x0, vl256
// CHECK-ENCODING: [0xa0,0xf9,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: a0 f9 f0 04 <unknown>

sqdecd  x0, #14
// CHECK-INST: sqdecd  x0, #14
// CHECK-ENCODING: [0xc0,0xf9,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c0 f9 f0 04 <unknown>

sqdecd  x0, #15
// CHECK-INST: sqdecd  x0, #15
// CHECK-ENCODING: [0xe0,0xf9,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 f9 f0 04 <unknown>

sqdecd  x0, #16
// CHECK-INST: sqdecd  x0, #16
// CHECK-ENCODING: [0x00,0xfa,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 fa f0 04 <unknown>

sqdecd  x0, #17
// CHECK-INST: sqdecd  x0, #17
// CHECK-ENCODING: [0x20,0xfa,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 fa f0 04 <unknown>

sqdecd  x0, #18
// CHECK-INST: sqdecd  x0, #18
// CHECK-ENCODING: [0x40,0xfa,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 fa f0 04 <unknown>

sqdecd  x0, #19
// CHECK-INST: sqdecd  x0, #19
// CHECK-ENCODING: [0x60,0xfa,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 fa f0 04 <unknown>

sqdecd  x0, #20
// CHECK-INST: sqdecd  x0, #20
// CHECK-ENCODING: [0x80,0xfa,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 fa f0 04 <unknown>

sqdecd  x0, #21
// CHECK-INST: sqdecd  x0, #21
// CHECK-ENCODING: [0xa0,0xfa,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: a0 fa f0 04 <unknown>

sqdecd  x0, #22
// CHECK-INST: sqdecd  x0, #22
// CHECK-ENCODING: [0xc0,0xfa,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c0 fa f0 04 <unknown>

sqdecd  x0, #23
// CHECK-INST: sqdecd  x0, #23
// CHECK-ENCODING: [0xe0,0xfa,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fa f0 04 <unknown>

sqdecd  x0, #24
// CHECK-INST: sqdecd  x0, #24
// CHECK-ENCODING: [0x00,0xfb,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 fb f0 04 <unknown>

sqdecd  x0, #25
// CHECK-INST: sqdecd  x0, #25
// CHECK-ENCODING: [0x20,0xfb,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 fb f0 04 <unknown>

sqdecd  x0, #26
// CHECK-INST: sqdecd  x0, #26
// CHECK-ENCODING: [0x40,0xfb,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 fb f0 04 <unknown>

sqdecd  x0, #27
// CHECK-INST: sqdecd  x0, #27
// CHECK-ENCODING: [0x60,0xfb,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 fb f0 04 <unknown>

sqdecd  x0, #28
// CHECK-INST: sqdecd  x0, #28
// CHECK-ENCODING: [0x80,0xfb,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 fb f0 04 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

sqdecd  z0.d
// CHECK-INST: sqdecd	z0.d
// CHECK-ENCODING: [0xe0,0xcb,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 cb e0 04 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

sqdecd  z0.d, pow2, mul #16
// CHECK-INST: sqdecd	z0.d, pow2, mul #16
// CHECK-ENCODING: [0x00,0xc8,0xef,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 c8 ef 04 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

sqdecd  z0.d, pow2
// CHECK-INST: sqdecd	z0.d, pow2
// CHECK-ENCODING: [0x00,0xc8,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 c8 e0 04 <unknown>
