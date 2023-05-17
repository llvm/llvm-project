// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1,+b16b16 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid predicate register

bfmax z23.h, p8/m, z23.h, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: bfmax z23.h, p8/m, z23.h, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmax z23.h, p1/z, z23.h, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfmax z23.h, p1/z, z23.h, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

bfmax z23.h, p1/z, z23.s, z13.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfmax z23.h, p1/z, z23.s, z13.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmax z23.s, z23.h, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmax z23.s, z23.h, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
