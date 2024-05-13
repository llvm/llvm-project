// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// ---------------------------------------------------------------------------//
// Test all predicate sizes for pow2 pattern
// ---------------------------------------------------------------------------//

ptrue   p0.b, pow2
// CHECK-INST: ptrue   p0.b, pow2
// CHECK-ENCODING: [0x00,0xe0,0x18,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2518e000 <unknown>

ptrue   p0.h, pow2
// CHECK-INST: ptrue   p0.h, pow2
// CHECK-ENCODING: [0x00,0xe0,0x58,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2558e000 <unknown>

ptrue   p0.s, pow2
// CHECK-INST: ptrue   p0.s, pow2
// CHECK-ENCODING: [0x00,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e000 <unknown>

ptrue   p0.d, pow2
// CHECK-INST: ptrue   p0.d, pow2
// CHECK-ENCODING: [0x00,0xe0,0xd8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25d8e000 <unknown>

// ---------------------------------------------------------------------------//
// Test all predicate sizes without explicit pattern
// ---------------------------------------------------------------------------//

ptrue   p15.b
// CHECK-INST: ptrue   p15.b
// CHECK-ENCODING: [0xef,0xe3,0x18,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2518e3ef <unknown>

ptrue   p15.h
// CHECK-INST: ptrue   p15.h
// CHECK-ENCODING: [0xef,0xe3,0x58,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2558e3ef <unknown>

ptrue   p15.s
// CHECK-INST: ptrue   p15.s
// CHECK-ENCODING: [0xef,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e3ef <unknown>

ptrue   p15.d
// CHECK-INST: ptrue   p15.d
// CHECK-ENCODING: [0xef,0xe3,0xd8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25d8e3ef <unknown>

// ---------------------------------------------------------------------------//
// Test available patterns
// ---------------------------------------------------------------------------//

ptrue   p7.s, #1
// CHECK-INST: ptrue   p7.s, vl1
// CHECK-ENCODING: [0x27,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e027 <unknown>

ptrue   p7.s, vl1
// CHECK-INST: ptrue   p7.s, vl1
// CHECK-ENCODING: [0x27,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e027 <unknown>

ptrue   p7.s, vl2
// CHECK-INST: ptrue   p7.s, vl2
// CHECK-ENCODING: [0x47,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e047 <unknown>

ptrue   p7.s, vl3
// CHECK-INST: ptrue   p7.s, vl3
// CHECK-ENCODING: [0x67,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e067 <unknown>

ptrue   p7.s, vl4
// CHECK-INST: ptrue   p7.s, vl4
// CHECK-ENCODING: [0x87,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e087 <unknown>

ptrue   p7.s, vl5
// CHECK-INST: ptrue   p7.s, vl5
// CHECK-ENCODING: [0xa7,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e0a7 <unknown>

ptrue   p7.s, vl6
// CHECK-INST: ptrue   p7.s, vl6
// CHECK-ENCODING: [0xc7,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e0c7 <unknown>

ptrue   p7.s, vl7
// CHECK-INST: ptrue   p7.s, vl7
// CHECK-ENCODING: [0xe7,0xe0,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e0e7 <unknown>

ptrue   p7.s, vl8
// CHECK-INST: ptrue   p7.s, vl8
// CHECK-ENCODING: [0x07,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e107 <unknown>

ptrue   p7.s, vl16
// CHECK-INST: ptrue   p7.s, vl16
// CHECK-ENCODING: [0x27,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e127 <unknown>

ptrue   p7.s, vl32
// CHECK-INST: ptrue   p7.s, vl32
// CHECK-ENCODING: [0x47,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e147 <unknown>

ptrue   p7.s, vl64
// CHECK-INST: ptrue   p7.s, vl64
// CHECK-ENCODING: [0x67,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e167 <unknown>

ptrue   p7.s, vl128
// CHECK-INST: ptrue   p7.s, vl128
// CHECK-ENCODING: [0x87,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e187 <unknown>

ptrue   p7.s, vl256
// CHECK-INST: ptrue   p7.s, vl256
// CHECK-ENCODING: [0xa7,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e1a7 <unknown>

ptrue   p7.s, mul4
// CHECK-INST: ptrue   p7.s, mul4
// CHECK-ENCODING: [0xa7,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e3a7 <unknown>

ptrue   p7.s, mul3
// CHECK-INST: ptrue   p7.s, mul3
// CHECK-ENCODING: [0xc7,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e3c7 <unknown>

ptrue   p7.s, all
// CHECK-INST: ptrue   p7.s
// CHECK-ENCODING: [0xe7,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e3e7 <unknown>

// ---------------------------------------------------------------------------//
// Test immediate values not corresponding to a named pattern
// ---------------------------------------------------------------------------//

ptrue   p7.s, #14
// CHECK-INST: ptrue   p7.s, #14
// CHECK-ENCODING: [0xc7,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e1c7 <unknown>

ptrue   p7.s, #15
// CHECK-INST: ptrue   p7.s, #15
// CHECK-ENCODING: [0xe7,0xe1,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e1e7 <unknown>

ptrue   p7.s, #16
// CHECK-INST: ptrue   p7.s, #16
// CHECK-ENCODING: [0x07,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e207 <unknown>

ptrue   p7.s, #17
// CHECK-INST: ptrue   p7.s, #17
// CHECK-ENCODING: [0x27,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e227 <unknown>

ptrue   p7.s, #18
// CHECK-INST: ptrue   p7.s, #18
// CHECK-ENCODING: [0x47,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e247 <unknown>

ptrue   p7.s, #19
// CHECK-INST: ptrue   p7.s, #19
// CHECK-ENCODING: [0x67,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e267 <unknown>

ptrue   p7.s, #20
// CHECK-INST: ptrue   p7.s, #20
// CHECK-ENCODING: [0x87,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e287 <unknown>

ptrue   p7.s, #21
// CHECK-INST: ptrue   p7.s, #21
// CHECK-ENCODING: [0xa7,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e2a7 <unknown>

ptrue   p7.s, #22
// CHECK-INST: ptrue   p7.s, #22
// CHECK-ENCODING: [0xc7,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e2c7 <unknown>

ptrue   p7.s, #23
// CHECK-INST: ptrue   p7.s, #23
// CHECK-ENCODING: [0xe7,0xe2,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e2e7 <unknown>

ptrue   p7.s, #24
// CHECK-INST: ptrue   p7.s, #24
// CHECK-ENCODING: [0x07,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e307 <unknown>

ptrue   p7.s, #25
// CHECK-INST: ptrue   p7.s, #25
// CHECK-ENCODING: [0x27,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e327 <unknown>

ptrue   p7.s, #26
// CHECK-INST: ptrue   p7.s, #26
// CHECK-ENCODING: [0x47,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e347 <unknown>

ptrue   p7.s, #27
// CHECK-INST: ptrue   p7.s, #27
// CHECK-ENCODING: [0x67,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e367 <unknown>

ptrue   p7.s, #28
// CHECK-INST: ptrue   p7.s, #28
// CHECK-ENCODING: [0x87,0xe3,0x98,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2598e387 <unknown>
