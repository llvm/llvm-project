// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+lsfe 2>&1 < %s| FileCheck %s

//------------------------------------------------------------------------------
// LDFMAX
//------------------------------------------------------------------------------

ldfmax h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmax h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmax s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmax s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmax d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmax d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmax d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmax d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmax s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmax s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfmaxa

ldfmaxa h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxa h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxa s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxa s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxa d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxa d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxa d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxa d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxa s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxa s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfmaxal

ldfmaxal h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxal h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxal s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxal s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxal d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxal d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxal d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxal d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxal s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxal s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfmaxl

ldfmaxl h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxl h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxl s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxl s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxl d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxl d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxl d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxl d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxl s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxl s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

//------------------------------------------------------------------------------
// LDBFMAX
//------------------------------------------------------------------------------

ldbfmax s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmax s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmax h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmax h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmax s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmax s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmax d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmax d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmax h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmax h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmax h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmax h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfmaxa

ldbfmaxa s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxa s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxa h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxa h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxa s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxa s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxa d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxa d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxa h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxa h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxa h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxa h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfmaxal

ldbfmaxal s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxal s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxal h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxal h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxal s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxal s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxal d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxal d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxal h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxal h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxal h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxal h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfmaxl

ldbfmaxl s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxl s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxl h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxl h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxl s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxl s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxl d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxl d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxl h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxl h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxl h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxl h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: