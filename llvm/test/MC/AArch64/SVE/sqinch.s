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

sqinch  x0
// CHECK-INST: sqinch  x0
// CHECK-ENCODING: [0xe0,0xf3,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f3 70 04 <unknown>

sqinch  x0, all
// CHECK-INST: sqinch  x0
// CHECK-ENCODING: [0xe0,0xf3,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f3 70 04 <unknown>

sqinch  x0, all, mul #1
// CHECK-INST: sqinch  x0
// CHECK-ENCODING: [0xe0,0xf3,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f3 70 04 <unknown>

sqinch  x0, all, mul #16
// CHECK-INST: sqinch  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xf3,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f3 7f 04 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

sqinch  x0, pow2
// CHECK-INST: sqinch  x0, pow2
// CHECK-ENCODING: [0x00,0xf0,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f0 70 04 <unknown>

sqinch  x0, vl1
// CHECK-INST: sqinch  x0, vl1
// CHECK-ENCODING: [0x20,0xf0,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f0 70 04 <unknown>

sqinch  x0, vl2
// CHECK-INST: sqinch  x0, vl2
// CHECK-ENCODING: [0x40,0xf0,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 f0 70 04 <unknown>

sqinch  x0, vl3
// CHECK-INST: sqinch  x0, vl3
// CHECK-ENCODING: [0x60,0xf0,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 f0 70 04 <unknown>

sqinch  x0, vl4
// CHECK-INST: sqinch  x0, vl4
// CHECK-ENCODING: [0x80,0xf0,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 f0 70 04 <unknown>

sqinch  x0, vl5
// CHECK-INST: sqinch  x0, vl5
// CHECK-ENCODING: [0xa0,0xf0,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 f0 70 04 <unknown>

sqinch  x0, vl6
// CHECK-INST: sqinch  x0, vl6
// CHECK-ENCODING: [0xc0,0xf0,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 f0 70 04 <unknown>

sqinch  x0, vl7
// CHECK-INST: sqinch  x0, vl7
// CHECK-ENCODING: [0xe0,0xf0,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f0 70 04 <unknown>

sqinch  x0, vl8
// CHECK-INST: sqinch  x0, vl8
// CHECK-ENCODING: [0x00,0xf1,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f1 70 04 <unknown>

sqinch  x0, vl16
// CHECK-INST: sqinch  x0, vl16
// CHECK-ENCODING: [0x20,0xf1,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f1 70 04 <unknown>

sqinch  x0, vl32
// CHECK-INST: sqinch  x0, vl32
// CHECK-ENCODING: [0x40,0xf1,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 f1 70 04 <unknown>

sqinch  x0, vl64
// CHECK-INST: sqinch  x0, vl64
// CHECK-ENCODING: [0x60,0xf1,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 f1 70 04 <unknown>

sqinch  x0, vl128
// CHECK-INST: sqinch  x0, vl128
// CHECK-ENCODING: [0x80,0xf1,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 f1 70 04 <unknown>

sqinch  x0, vl256
// CHECK-INST: sqinch  x0, vl256
// CHECK-ENCODING: [0xa0,0xf1,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 f1 70 04 <unknown>

sqinch  x0, #14
// CHECK-INST: sqinch  x0, #14
// CHECK-ENCODING: [0xc0,0xf1,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 f1 70 04 <unknown>

sqinch  x0, #15
// CHECK-INST: sqinch  x0, #15
// CHECK-ENCODING: [0xe0,0xf1,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f1 70 04 <unknown>

sqinch  x0, #16
// CHECK-INST: sqinch  x0, #16
// CHECK-ENCODING: [0x00,0xf2,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f2 70 04 <unknown>

sqinch  x0, #17
// CHECK-INST: sqinch  x0, #17
// CHECK-ENCODING: [0x20,0xf2,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f2 70 04 <unknown>

sqinch  x0, #18
// CHECK-INST: sqinch  x0, #18
// CHECK-ENCODING: [0x40,0xf2,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 f2 70 04 <unknown>

sqinch  x0, #19
// CHECK-INST: sqinch  x0, #19
// CHECK-ENCODING: [0x60,0xf2,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 f2 70 04 <unknown>

sqinch  x0, #20
// CHECK-INST: sqinch  x0, #20
// CHECK-ENCODING: [0x80,0xf2,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 f2 70 04 <unknown>

sqinch  x0, #21
// CHECK-INST: sqinch  x0, #21
// CHECK-ENCODING: [0xa0,0xf2,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 f2 70 04 <unknown>

sqinch  x0, #22
// CHECK-INST: sqinch  x0, #22
// CHECK-ENCODING: [0xc0,0xf2,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c0 f2 70 04 <unknown>

sqinch  x0, #23
// CHECK-INST: sqinch  x0, #23
// CHECK-ENCODING: [0xe0,0xf2,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 f2 70 04 <unknown>

sqinch  x0, #24
// CHECK-INST: sqinch  x0, #24
// CHECK-ENCODING: [0x00,0xf3,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 f3 70 04 <unknown>

sqinch  x0, #25
// CHECK-INST: sqinch  x0, #25
// CHECK-ENCODING: [0x20,0xf3,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 f3 70 04 <unknown>

sqinch  x0, #26
// CHECK-INST: sqinch  x0, #26
// CHECK-ENCODING: [0x40,0xf3,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 40 f3 70 04 <unknown>

sqinch  x0, #27
// CHECK-INST: sqinch  x0, #27
// CHECK-ENCODING: [0x60,0xf3,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 60 f3 70 04 <unknown>

sqinch  x0, #28
// CHECK-INST: sqinch  x0, #28
// CHECK-ENCODING: [0x80,0xf3,0x70,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 80 f3 70 04 <unknown>
