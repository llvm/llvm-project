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

sqincd  x0
// CHECK-INST: sqincd  x0
// CHECK-ENCODING: [0xe0,0xf3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f3 f0 04 <unknown>

sqincd  x0, all
// CHECK-INST: sqincd  x0
// CHECK-ENCODING: [0xe0,0xf3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f3 f0 04 <unknown>

sqincd  x0, all, mul #1
// CHECK-INST: sqincd  x0
// CHECK-ENCODING: [0xe0,0xf3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f3 f0 04 <unknown>

sqincd  x0, all, mul #16
// CHECK-INST: sqincd  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xf3,0xff,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f3 ff 04 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (x0, w0) and its aliases
// ---------------------------------------------------------------------------//

sqincd  x0, w0
// CHECK-INST: sqincd  x0, w0
// CHECK-ENCODING: [0xe0,0xf3,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f3 e0 04 <unknown>

sqincd  x0, w0, all
// CHECK-INST: sqincd  x0, w0
// CHECK-ENCODING: [0xe0,0xf3,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f3 e0 04 <unknown>

sqincd  x0, w0, all, mul #1
// CHECK-INST: sqincd  x0, w0
// CHECK-ENCODING: [0xe0,0xf3,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f3 e0 04 <unknown>

sqincd  x0, w0, all, mul #16
// CHECK-INST: sqincd  x0, w0, all
// CHECK-ENCODING: [0xe0,0xf3,0xef,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f3 ef 04 <unknown>

sqincd  x0, w0, pow2
// CHECK-INST: sqincd  x0, w0, pow2
// CHECK-ENCODING: [0x00,0xf0,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f0 e0 04 <unknown>

sqincd  x0, w0, pow2, mul #16
// CHECK-INST: sqincd  x0, w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xf0,0xef,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f0 ef 04 <unknown>


// ---------------------------------------------------------------------------//
// Test vector form and aliases.
// ---------------------------------------------------------------------------//
sqincd  z0.d
// CHECK-INST: sqincd  z0.d
// CHECK-ENCODING: [0xe0,0xc3,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 c3 e0 04 <unknown>

sqincd  z0.d, all
// CHECK-INST: sqincd  z0.d
// CHECK-ENCODING: [0xe0,0xc3,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 c3 e0 04 <unknown>

sqincd  z0.d, all, mul #1
// CHECK-INST: sqincd  z0.d
// CHECK-ENCODING: [0xe0,0xc3,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 c3 e0 04 <unknown>

sqincd  z0.d, all, mul #16
// CHECK-INST: sqincd  z0.d, all, mul #16
// CHECK-ENCODING: [0xe0,0xc3,0xef,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 c3 ef 04 <unknown>

sqincd  z0.d, pow2
// CHECK-INST: sqincd  z0.d, pow2
// CHECK-ENCODING: [0x00,0xc0,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 e0 04 <unknown>

sqincd  z0.d, pow2, mul #16
// CHECK-INST: sqincd  z0.d, pow2, mul #16
// CHECK-ENCODING: [0x00,0xc0,0xef,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 ef 04 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

sqincd  x0, pow2
// CHECK-INST: sqincd  x0, pow2
// CHECK-ENCODING: [0x00,0xf0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f0 f0 04 <unknown>

sqincd  x0, vl1
// CHECK-INST: sqincd  x0, vl1
// CHECK-ENCODING: [0x20,0xf0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f0 f0 04 <unknown>

sqincd  x0, vl2
// CHECK-INST: sqincd  x0, vl2
// CHECK-ENCODING: [0x40,0xf0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 f0 f0 04 <unknown>

sqincd  x0, vl3
// CHECK-INST: sqincd  x0, vl3
// CHECK-ENCODING: [0x60,0xf0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 f0 f0 04 <unknown>

sqincd  x0, vl4
// CHECK-INST: sqincd  x0, vl4
// CHECK-ENCODING: [0x80,0xf0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 f0 f0 04 <unknown>

sqincd  x0, vl5
// CHECK-INST: sqincd  x0, vl5
// CHECK-ENCODING: [0xa0,0xf0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 f0 f0 04 <unknown>

sqincd  x0, vl6
// CHECK-INST: sqincd  x0, vl6
// CHECK-ENCODING: [0xc0,0xf0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 f0 f0 04 <unknown>

sqincd  x0, vl7
// CHECK-INST: sqincd  x0, vl7
// CHECK-ENCODING: [0xe0,0xf0,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f0 f0 04 <unknown>

sqincd  x0, vl8
// CHECK-INST: sqincd  x0, vl8
// CHECK-ENCODING: [0x00,0xf1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f1 f0 04 <unknown>

sqincd  x0, vl16
// CHECK-INST: sqincd  x0, vl16
// CHECK-ENCODING: [0x20,0xf1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f1 f0 04 <unknown>

sqincd  x0, vl32
// CHECK-INST: sqincd  x0, vl32
// CHECK-ENCODING: [0x40,0xf1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 f1 f0 04 <unknown>

sqincd  x0, vl64
// CHECK-INST: sqincd  x0, vl64
// CHECK-ENCODING: [0x60,0xf1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 f1 f0 04 <unknown>

sqincd  x0, vl128
// CHECK-INST: sqincd  x0, vl128
// CHECK-ENCODING: [0x80,0xf1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 f1 f0 04 <unknown>

sqincd  x0, vl256
// CHECK-INST: sqincd  x0, vl256
// CHECK-ENCODING: [0xa0,0xf1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 f1 f0 04 <unknown>

sqincd  x0, #14
// CHECK-INST: sqincd  x0, #14
// CHECK-ENCODING: [0xc0,0xf1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 f1 f0 04 <unknown>

sqincd  x0, #15
// CHECK-INST: sqincd  x0, #15
// CHECK-ENCODING: [0xe0,0xf1,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f1 f0 04 <unknown>

sqincd  x0, #16
// CHECK-INST: sqincd  x0, #16
// CHECK-ENCODING: [0x00,0xf2,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f2 f0 04 <unknown>

sqincd  x0, #17
// CHECK-INST: sqincd  x0, #17
// CHECK-ENCODING: [0x20,0xf2,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f2 f0 04 <unknown>

sqincd  x0, #18
// CHECK-INST: sqincd  x0, #18
// CHECK-ENCODING: [0x40,0xf2,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 f2 f0 04 <unknown>

sqincd  x0, #19
// CHECK-INST: sqincd  x0, #19
// CHECK-ENCODING: [0x60,0xf2,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 f2 f0 04 <unknown>

sqincd  x0, #20
// CHECK-INST: sqincd  x0, #20
// CHECK-ENCODING: [0x80,0xf2,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 f2 f0 04 <unknown>

sqincd  x0, #21
// CHECK-INST: sqincd  x0, #21
// CHECK-ENCODING: [0xa0,0xf2,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 f2 f0 04 <unknown>

sqincd  x0, #22
// CHECK-INST: sqincd  x0, #22
// CHECK-ENCODING: [0xc0,0xf2,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 f2 f0 04 <unknown>

sqincd  x0, #23
// CHECK-INST: sqincd  x0, #23
// CHECK-ENCODING: [0xe0,0xf2,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f2 f0 04 <unknown>

sqincd  x0, #24
// CHECK-INST: sqincd  x0, #24
// CHECK-ENCODING: [0x00,0xf3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f3 f0 04 <unknown>

sqincd  x0, #25
// CHECK-INST: sqincd  x0, #25
// CHECK-ENCODING: [0x20,0xf3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f3 f0 04 <unknown>

sqincd  x0, #26
// CHECK-INST: sqincd  x0, #26
// CHECK-ENCODING: [0x40,0xf3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 f3 f0 04 <unknown>

sqincd  x0, #27
// CHECK-INST: sqincd  x0, #27
// CHECK-ENCODING: [0x60,0xf3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 f3 f0 04 <unknown>

sqincd  x0, #28
// CHECK-INST: sqincd  x0, #28
// CHECK-ENCODING: [0x80,0xf3,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 f3 f0 04 <unknown>
