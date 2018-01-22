// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// ---------------------------------------------------------------------------//
// Test all predicate sizes for pow2 pattern
// ---------------------------------------------------------------------------//

ptrues   p0.b, pow2
// CHECK-INST: ptrues   p0.b, pow2
// CHECK-ENCODING: [0x00,0xe0,0x19,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	00 e0 19 25  <unknown>

ptrues   p0.h, pow2
// CHECK-INST: ptrues   p0.h, pow2
// CHECK-ENCODING: [0x00,0xe0,0x59,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	00 e0 59 25  <unknown>

ptrues   p0.s, pow2
// CHECK-INST: ptrues   p0.s, pow2
// CHECK-ENCODING: [0x00,0xe0,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	00 e0 99 25  <unknown>

ptrues   p0.d, pow2
// CHECK-INST: ptrues   p0.d, pow2
// CHECK-ENCODING: [0x00,0xe0,0xd9,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	00 e0 d9 25  <unknown>

// ---------------------------------------------------------------------------//
// Test all predicate sizes without explicit pattern
// ---------------------------------------------------------------------------//

ptrues   p15.b
// CHECK-INST: ptrues   p15.b
// CHECK-ENCODING: [0xef,0xe3,0x19,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	ef e3 19 25  <unknown>

ptrues   p15.h
// CHECK-INST: ptrues   p15.h
// CHECK-ENCODING: [0xef,0xe3,0x59,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	ef e3 59 25  <unknown>

ptrues   p15.s
// CHECK-INST: ptrues   p15.s
// CHECK-ENCODING: [0xef,0xe3,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	ef e3 99 25  <unknown>

ptrues   p15.d
// CHECK-INST: ptrues   p15.d
// CHECK-ENCODING: [0xef,0xe3,0xd9,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	ef e3 d9 25  <unknown>

// ---------------------------------------------------------------------------//
// Test available patterns
// ---------------------------------------------------------------------------//

ptrues   p7.s, #1
// CHECK-INST: ptrues   p7.s, vl1
// CHECK-ENCODING: [0x27,0xe0,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	27 e0 99 25  <unknown>

ptrues   p7.s, vl1
// CHECK-INST: ptrues   p7.s, vl1
// CHECK-ENCODING: [0x27,0xe0,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	27 e0 99 25  <unknown>

ptrues   p7.s, vl2
// CHECK-INST: ptrues   p7.s, vl2
// CHECK-ENCODING: [0x47,0xe0,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	47 e0 99 25  <unknown>

ptrues   p7.s, vl3
// CHECK-INST: ptrues   p7.s, vl3
// CHECK-ENCODING: [0x67,0xe0,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	67 e0 99 25  <unknown>

ptrues   p7.s, vl4
// CHECK-INST: ptrues   p7.s, vl4
// CHECK-ENCODING: [0x87,0xe0,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	87 e0 99 25  <unknown>

ptrues   p7.s, vl5
// CHECK-INST: ptrues   p7.s, vl5
// CHECK-ENCODING: [0xa7,0xe0,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	a7 e0 99 25  <unknown>

ptrues   p7.s, vl6
// CHECK-INST: ptrues   p7.s, vl6
// CHECK-ENCODING: [0xc7,0xe0,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	c7 e0 99 25  <unknown>

ptrues   p7.s, vl7
// CHECK-INST: ptrues   p7.s, vl7
// CHECK-ENCODING: [0xe7,0xe0,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	e7 e0 99 25  <unknown>

ptrues   p7.s, vl8
// CHECK-INST: ptrues   p7.s, vl8
// CHECK-ENCODING: [0x07,0xe1,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	07 e1 99 25  <unknown>

ptrues   p7.s, vl16
// CHECK-INST: ptrues   p7.s, vl16
// CHECK-ENCODING: [0x27,0xe1,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	27 e1 99 25  <unknown>

ptrues   p7.s, vl32
// CHECK-INST: ptrues   p7.s, vl32
// CHECK-ENCODING: [0x47,0xe1,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	47 e1 99 25  <unknown>

ptrues   p7.s, vl64
// CHECK-INST: ptrues   p7.s, vl64
// CHECK-ENCODING: [0x67,0xe1,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	67 e1 99 25  <unknown>

ptrues   p7.s, vl128
// CHECK-INST: ptrues   p7.s, vl128
// CHECK-ENCODING: [0x87,0xe1,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	87 e1 99 25  <unknown>

ptrues   p7.s, vl256
// CHECK-INST: ptrues   p7.s, vl256
// CHECK-ENCODING: [0xa7,0xe1,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	a7 e1 99 25  <unknown>

ptrues   p7.s, mul4
// CHECK-INST: ptrues   p7.s, mul4
// CHECK-ENCODING: [0xa7,0xe3,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	a7 e3 99 25  <unknown>

ptrues   p7.s, mul3
// CHECK-INST: ptrues   p7.s, mul3
// CHECK-ENCODING: [0xc7,0xe3,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	c7 e3 99 25  <unknown>

ptrues   p7.s, all
// CHECK-INST: ptrues   p7.s
// CHECK-ENCODING: [0xe7,0xe3,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN:	e7 e3 99 25  <unknown>

// ---------------------------------------------------------------------------//
// Test immediate values not corresponding to a named pattern
// ---------------------------------------------------------------------------//

ptrues   p7.s, #14
// CHECK-INST: ptrues   p7.s, #14
// CHECK-ENCODING: [0xc7,0xe1,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c7 e1 99 25 <unknown>

ptrues   p7.s, #15
// CHECK-INST: ptrues   p7.s, #15
// CHECK-ENCODING: [0xe7,0xe1,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e7 e1 99 25 <unknown>

ptrues   p7.s, #16
// CHECK-INST: ptrues   p7.s, #16
// CHECK-ENCODING: [0x07,0xe2,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 07 e2 99 25 <unknown>

ptrues   p7.s, #17
// CHECK-INST: ptrues   p7.s, #17
// CHECK-ENCODING: [0x27,0xe2,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 27 e2 99 25 <unknown>

ptrues   p7.s, #18
// CHECK-INST: ptrues   p7.s, #18
// CHECK-ENCODING: [0x47,0xe2,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 47 e2 99 25 <unknown>

ptrues   p7.s, #19
// CHECK-INST: ptrues   p7.s, #19
// CHECK-ENCODING: [0x67,0xe2,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 67 e2 99 25 <unknown>

ptrues   p7.s, #20
// CHECK-INST: ptrues   p7.s, #20
// CHECK-ENCODING: [0x87,0xe2,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 87 e2 99 25 <unknown>

ptrues   p7.s, #21
// CHECK-INST: ptrues   p7.s, #21
// CHECK-ENCODING: [0xa7,0xe2,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a7 e2 99 25 <unknown>

ptrues   p7.s, #22
// CHECK-INST: ptrues   p7.s, #22
// CHECK-ENCODING: [0xc7,0xe2,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c7 e2 99 25 <unknown>

ptrues   p7.s, #23
// CHECK-INST: ptrues   p7.s, #23
// CHECK-ENCODING: [0xe7,0xe2,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e7 e2 99 25 <unknown>

ptrues   p7.s, #24
// CHECK-INST: ptrues   p7.s, #24
// CHECK-ENCODING: [0x07,0xe3,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 07 e3 99 25 <unknown>

ptrues   p7.s, #25
// CHECK-INST: ptrues   p7.s, #25
// CHECK-ENCODING: [0x27,0xe3,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 27 e3 99 25 <unknown>

ptrues   p7.s, #26
// CHECK-INST: ptrues   p7.s, #26
// CHECK-ENCODING: [0x47,0xe3,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 47 e3 99 25 <unknown>

ptrues   p7.s, #27
// CHECK-INST: ptrues   p7.s, #27
// CHECK-ENCODING: [0x67,0xe3,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 67 e3 99 25 <unknown>

ptrues   p7.s, #28
// CHECK-INST: ptrues   p7.s, #28
// CHECK-ENCODING: [0x87,0xe3,0x99,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 87 e3 99 25 <unknown>
