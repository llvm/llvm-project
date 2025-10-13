// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1,+sve-aes2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

pmull   {z0.q-z2.q}, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: pmull   {z0.q-z2.q}, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmull   {z0.q-z0.q}, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: pmull   {z0.q-z0.q}, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmull   {z1.q-z2.q}, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: pmull   {z1.q-z2.q}, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmull   {z0.d-z1.d}, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: pmull   {z0.d-z1.d}, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid single source vectors

pmull   {z0.q-z1.q}, z0.s, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: pmull   {z0.q-z1.q}, z0.s, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmull   {z0.q-z1.q}, z0.d, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: pmull   {z0.q-z1.q}, z0.d, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: