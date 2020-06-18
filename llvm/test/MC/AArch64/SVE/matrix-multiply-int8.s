// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+i8mm < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+i8mm < %s \
// RUN:        | llvm-objdump -d --mattr=+sve,+i8mm - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+i8mm < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


// --------------------------------------------------------------------------//
// SMMLA, UMMLA, USMMLA (SVE)

ummla z0.s, z1.b, z2.b
// CHECK-INST: ummla z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x98,0xc2,0x45]
// CHECK-ERROR: instruction requires: i8mm
// CHECK-UNKNOWN: 20 98 c2 45 <unknown>

smmla z0.s, z1.b, z2.b
// CHECK-INST: smmla z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x98,0x02,0x45]
// CHECK-ERROR: instruction requires: i8mm
// CHECK-UNKNOWN: 20 98 02 45 <unknown>

usmmla z0.s, z1.b, z2.b
// CHECK-INST: usmmla z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x98,0x82,0x45]
// CHECK-ERROR: instruction requires: i8mm
// CHECK-UNKNOWN: 20 98 82 45 <unknown>


// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

ummla z0.s, z1.b, z2.b
// CHECK-INST: ummla z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x98,0xc2,0x45]
// CHECK-ERROR: instruction requires: i8mm
// CHECK-UNKNOWN: 20 98 c2 45 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

smmla z0.s, z1.b, z2.b
// CHECK-INST: smmla z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x98,0x02,0x45]
// CHECK-ERROR: instruction requires: i8mm
// CHECK-UNKNOWN: 20 98 02 45 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

usmmla z0.s, z1.b, z2.b
// CHECK-INST: usmmla z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x98,0x82,0x45]
// CHECK-ERROR: instruction requires: i8mm
// CHECK-UNKNOWN: 20 98 82 45 <unknown>


// --------------------------------------------------------------------------//
// USDOT (SVE, vectors)

usdot z0.s, z1.b, z2.b
// CHECK-INST: usdot z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x78,0x82,0x44]
// CHECK-ERROR: instruction requires: i8mm
// CHECK-UNKNOWN: 20 78 82 44 <unknown>

// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

usdot z0.s, z1.b, z2.b
// CHECK-INST: usdot z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x78,0x82,0x44]
// CHECK-ERROR: instruction requires: i8mm
// CHECK-UNKNOWN: 20 78 82 44 <unknown>


// --------------------------------------------------------------------------//
// USDOT, SUDOT (SVE, indexed)

usdot z0.s, z1.b, z2.b[0]
// CHECK-INST: usdot z0.s, z1.b, z2.b[0]
// CHECK-ENCODING: [0x20,0x18,0xa2,0x44]
// CHECK-ERROR: instruction requires: i8mm
// CHECK-UNKNOWN: 20 18 a2 44 <unknown>

sudot z0.s, z1.b, z2.b[3]
// CHECK-INST: sudot z0.s, z1.b, z2.b[3]
// CHECK-ENCODING: [0x20,0x1c,0xba,0x44]
// CHECK-ERROR: instruction requires: i8mm
// CHECK-UNKNOWN: 20 1c ba 44 <unknown>

// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

usdot z0.s, z1.b, z2.b[0]
// CHECK-INST: usdot z0.s, z1.b, z2.b[0]
// CHECK-ENCODING: [0x20,0x18,0xa2,0x44]
// CHECK-ERROR: instruction requires: i8mm
// CHECK-UNKNOWN: 20 18 a2 44 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

sudot z0.s, z1.b, z2.b[0]
// CHECK-INST: sudot z0.s, z1.b, z2.b[0]
// CHECK-ENCODING: [0x20,0x1c,0xa2,0x44]
// CHECK-ERROR: instruction requires: i8mm
// CHECK-UNKNOWN: 20 1c a2 44 <unknown>
