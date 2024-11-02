// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+i8mm < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+i8mm < %s \
// RUN:        | llvm-objdump -d --mattr=+sve,+i8mm - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+i8mm < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN


// --------------------------------------------------------------------------//
// SMMLA, UMMLA, USMMLA (SVE)

ummla z0.s, z1.b, z2.b
// CHECK-INST: ummla z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x98,0xc2,0x45]
// CHECK-ERROR: instruction requires: i8mm sve
// CHECK-UNKNOWN: 45c29820 <unknown>

smmla z0.s, z1.b, z2.b
// CHECK-INST: smmla z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x98,0x02,0x45]
// CHECK-ERROR: instruction requires: i8mm sve
// CHECK-UNKNOWN: 45029820 <unknown>

usmmla z0.s, z1.b, z2.b
// CHECK-INST: usmmla z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x98,0x82,0x45]
// CHECK-ERROR: instruction requires: i8mm sve
// CHECK-UNKNOWN: 45829820 <unknown>


// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-UNKNOWN: 0420bce0 <unknown>

ummla z0.s, z1.b, z2.b
// CHECK-INST: ummla z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x98,0xc2,0x45]
// CHECK-ERROR: instruction requires: i8mm sve
// CHECK-UNKNOWN: 45c29820 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-UNKNOWN: 0420bce0 <unknown>

smmla z0.s, z1.b, z2.b
// CHECK-INST: smmla z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x98,0x02,0x45]
// CHECK-ERROR: instruction requires: i8mm sve
// CHECK-UNKNOWN: 45029820 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-UNKNOWN: 0420bce0 <unknown>

usmmla z0.s, z1.b, z2.b
// CHECK-INST: usmmla z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x98,0x82,0x45]
// CHECK-ERROR: instruction requires: i8mm sve
// CHECK-UNKNOWN: 45829820 <unknown>


// --------------------------------------------------------------------------//
// USDOT (SVE, vectors)

usdot z0.s, z1.b, z2.b
// CHECK-INST: usdot z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x78,0x82,0x44]
// CHECK-ERROR: instruction requires: i8mm sve or sme
// CHECK-UNKNOWN: 44827820 <unknown>

// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-UNKNOWN: 0420bce0 <unknown>

usdot z0.s, z1.b, z2.b
// CHECK-INST: usdot z0.s, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x78,0x82,0x44]
// CHECK-ERROR: instruction requires: i8mm sve or sme
// CHECK-UNKNOWN: 44827820 <unknown>


// --------------------------------------------------------------------------//
// USDOT, SUDOT (SVE, indexed)

usdot z0.s, z1.b, z2.b[0]
// CHECK-INST: usdot z0.s, z1.b, z2.b[0]
// CHECK-ENCODING: [0x20,0x18,0xa2,0x44]
// CHECK-ERROR: instruction requires: i8mm sve or sme
// CHECK-UNKNOWN: 44a21820 <unknown>

sudot z0.s, z1.b, z2.b[3]
// CHECK-INST: sudot z0.s, z1.b, z2.b[3]
// CHECK-ENCODING: [0x20,0x1c,0xba,0x44]
// CHECK-ERROR: instruction requires: i8mm sve or sme
// CHECK-UNKNOWN: 44ba1c20 <unknown>

// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-UNKNOWN: 0420bce0 <unknown>

usdot z0.s, z1.b, z2.b[0]
// CHECK-INST: usdot z0.s, z1.b, z2.b[0]
// CHECK-ENCODING: [0x20,0x18,0xa2,0x44]
// CHECK-ERROR: instruction requires: i8mm sve or sme
// CHECK-UNKNOWN: 44a21820 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-UNKNOWN: 0420bce0 <unknown>

sudot z0.s, z1.b, z2.b[0]
// CHECK-INST: sudot z0.s, z1.b, z2.b[0]
// CHECK-ENCODING: [0x20,0x1c,0xa2,0x44]
// CHECK-ERROR: instruction requires: i8mm sve or sme
// CHECK-UNKNOWN: 44a21c20 <unknown>
