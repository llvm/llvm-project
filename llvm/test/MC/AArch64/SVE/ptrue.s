// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// ---------------------------------------------------------------------------//
// Test all predicate sizes for pow2 pattern
// ---------------------------------------------------------------------------//

ptrue   p0.b, pow2
// CHECK-INST: ptrue   p0.b, pow2
// CHECK-ENCODING: [0x00,0xe0,0x18,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 18 25 <unknown>

ptrue   p0.h, pow2
// CHECK-INST: ptrue   p0.h, pow2
// CHECK-ENCODING: [0x00,0xe0,0x58,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 58 25 <unknown>

ptrue   p0.s, pow2
// CHECK-INST: ptrue   p0.s, pow2
// CHECK-ENCODING: [0x00,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 98 25 <unknown>

ptrue   p0.d, pow2
// CHECK-INST: ptrue   p0.d, pow2
// CHECK-ENCODING: [0x00,0xe0,0xd8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 d8 25 <unknown>

// ---------------------------------------------------------------------------//
// Test all predicate sizes without explicit pattern
// ---------------------------------------------------------------------------//

ptrue   p15.b
// CHECK-INST: ptrue   p15.b
// CHECK-ENCODING: [0xef,0xe3,0x18,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef e3 18 25 <unknown>

ptrue   p15.h
// CHECK-INST: ptrue   p15.h
// CHECK-ENCODING: [0xef,0xe3,0x58,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef e3 58 25 <unknown>

ptrue   p15.s
// CHECK-INST: ptrue   p15.s
// CHECK-ENCODING: [0xef,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef e3 98 25 <unknown>

ptrue   p15.d
// CHECK-INST: ptrue   p15.d
// CHECK-ENCODING: [0xef,0xe3,0xd8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef e3 d8 25 <unknown>

// ---------------------------------------------------------------------------//
// Test available patterns
// ---------------------------------------------------------------------------//

ptrue   p7.s, #1
// CHECK-INST: ptrue   p7.s, vl1
// CHECK-ENCODING: [0x27,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 27 e0 98 25 <unknown>

ptrue   p7.s, vl1
// CHECK-INST: ptrue   p7.s, vl1
// CHECK-ENCODING: [0x27,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 27 e0 98 25 <unknown>

ptrue   p7.s, vl2
// CHECK-INST: ptrue   p7.s, vl2
// CHECK-ENCODING: [0x47,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 47 e0 98 25 <unknown>

ptrue   p7.s, vl3
// CHECK-INST: ptrue   p7.s, vl3
// CHECK-ENCODING: [0x67,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 67 e0 98 25 <unknown>

ptrue   p7.s, vl4
// CHECK-INST: ptrue   p7.s, vl4
// CHECK-ENCODING: [0x87,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 87 e0 98 25 <unknown>

ptrue   p7.s, vl5
// CHECK-INST: ptrue   p7.s, vl5
// CHECK-ENCODING: [0xa7,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a7 e0 98 25 <unknown>

ptrue   p7.s, vl6
// CHECK-INST: ptrue   p7.s, vl6
// CHECK-ENCODING: [0xc7,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c7 e0 98 25 <unknown>

ptrue   p7.s, vl7
// CHECK-INST: ptrue   p7.s, vl7
// CHECK-ENCODING: [0xe7,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e7 e0 98 25 <unknown>

ptrue   p7.s, vl8
// CHECK-INST: ptrue   p7.s, vl8
// CHECK-ENCODING: [0x07,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 07 e1 98 25 <unknown>

ptrue   p7.s, vl16
// CHECK-INST: ptrue   p7.s, vl16
// CHECK-ENCODING: [0x27,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 27 e1 98 25 <unknown>

ptrue   p7.s, vl32
// CHECK-INST: ptrue   p7.s, vl32
// CHECK-ENCODING: [0x47,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 47 e1 98 25 <unknown>

ptrue   p7.s, vl64
// CHECK-INST: ptrue   p7.s, vl64
// CHECK-ENCODING: [0x67,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 67 e1 98 25 <unknown>

ptrue   p7.s, vl128
// CHECK-INST: ptrue   p7.s, vl128
// CHECK-ENCODING: [0x87,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 87 e1 98 25 <unknown>

ptrue   p7.s, vl256
// CHECK-INST: ptrue   p7.s, vl256
// CHECK-ENCODING: [0xa7,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a7 e1 98 25 <unknown>

ptrue   p7.s, mul4
// CHECK-INST: ptrue   p7.s, mul4
// CHECK-ENCODING: [0xa7,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a7 e3 98 25 <unknown>

ptrue   p7.s, mul3
// CHECK-INST: ptrue   p7.s, mul3
// CHECK-ENCODING: [0xc7,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c7 e3 98 25 <unknown>

ptrue   p7.s, all
// CHECK-INST: ptrue   p7.s
// CHECK-ENCODING: [0xe7,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e7 e3 98 25 <unknown>

// ---------------------------------------------------------------------------//
// Test immediate values not corresponding to a named pattern
// ---------------------------------------------------------------------------//

ptrue   p7.s, #14
// CHECK-INST: ptrue   p7.s, #14
// CHECK-ENCODING: [0xc7,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c7 e1 98 25 <unknown>

ptrue   p7.s, #15
// CHECK-INST: ptrue   p7.s, #15
// CHECK-ENCODING: [0xe7,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e7 e1 98 25 <unknown>

ptrue   p7.s, #16
// CHECK-INST: ptrue   p7.s, #16
// CHECK-ENCODING: [0x07,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 07 e2 98 25 <unknown>

ptrue   p7.s, #17
// CHECK-INST: ptrue   p7.s, #17
// CHECK-ENCODING: [0x27,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 27 e2 98 25 <unknown>

ptrue   p7.s, #18
// CHECK-INST: ptrue   p7.s, #18
// CHECK-ENCODING: [0x47,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 47 e2 98 25 <unknown>

ptrue   p7.s, #19
// CHECK-INST: ptrue   p7.s, #19
// CHECK-ENCODING: [0x67,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 67 e2 98 25 <unknown>

ptrue   p7.s, #20
// CHECK-INST: ptrue   p7.s, #20
// CHECK-ENCODING: [0x87,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 87 e2 98 25 <unknown>

ptrue   p7.s, #21
// CHECK-INST: ptrue   p7.s, #21
// CHECK-ENCODING: [0xa7,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a7 e2 98 25 <unknown>

ptrue   p7.s, #22
// CHECK-INST: ptrue   p7.s, #22
// CHECK-ENCODING: [0xc7,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c7 e2 98 25 <unknown>

ptrue   p7.s, #23
// CHECK-INST: ptrue   p7.s, #23
// CHECK-ENCODING: [0xe7,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e7 e2 98 25 <unknown>

ptrue   p7.s, #24
// CHECK-INST: ptrue   p7.s, #24
// CHECK-ENCODING: [0x07,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 07 e3 98 25 <unknown>

ptrue   p7.s, #25
// CHECK-INST: ptrue   p7.s, #25
// CHECK-ENCODING: [0x27,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 27 e3 98 25 <unknown>

ptrue   p7.s, #26
// CHECK-INST: ptrue   p7.s, #26
// CHECK-ENCODING: [0x47,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 47 e3 98 25 <unknown>

ptrue   p7.s, #27
// CHECK-INST: ptrue   p7.s, #27
// CHECK-ENCODING: [0x67,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 67 e3 98 25 <unknown>

ptrue   p7.s, #28
// CHECK-INST: ptrue   p7.s, #28
// CHECK-ENCODING: [0x87,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 87 e3 98 25 <unknown>
