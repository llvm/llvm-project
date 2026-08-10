// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve-f16f32mm  2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// FMMLA (SVE)

// Invalid element size

fmmla z0.s, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: f8f32mm
fmmla z0.d, z1.h, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width

// Mis-matched element size

fmmla z0.s, z1.h, z2.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
fmmla z0.s, z1.d, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width