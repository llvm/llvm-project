// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+i8mm  2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// SMMLA, UMMLA, USMMLA (SVE)

// Invalid element size

ummla z0.h, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
ummla z0.s, z1.h, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
ummla z0.s, z1.b, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width

// Negative tests for instructions that are incompatible with predicated movprfx

movprfx z0.d, p0/z, z7.d
ummla z0.s, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
movprfx z0.d, p0/z, z7.d
smmla z0.s, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
movprfx z0.d, p0/z, z7.d
usmmla z0.s, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx


// --------------------------------------------------------------------------//
// USDOT (SVE, vectors)

// Invalid element size

usdot z0.d, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
usdot z0.s, z1.s, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
usdot z0.s, z1.b, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z7.b

// Negative tests for instructions that are incompatible with predicated movprfx

movprfx z0.d, p0/z, z7.d
usdot z0.s, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx


// --------------------------------------------------------------------------//
// USDOT, SUDOT (SVE, indexed)

// Invalid element size

usdot z0.h, z1.b, z2.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
sudot z0.s, z1.h, z2.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
usdot z0.s, z1.b, z2.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z7.b

// Invalid restricted register for indexed vector.
usdot z0.s, z1.b, z9.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
sudot z0.s, z1.b, z9.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z7.b

// Invalid element index
usdot z0.s, z1.b, z2.b[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
sudot z0.s, z1.b, z2.b[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].

// Negative tests for instructions that are incompatible with predicated movprfx

movprfx z0.d, p0/z, z7.d
usdot z0.s, z1.b, z2.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
movprfx z0.d, p0/z, z7.d
sudot z0.s, z1.b, z2.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
