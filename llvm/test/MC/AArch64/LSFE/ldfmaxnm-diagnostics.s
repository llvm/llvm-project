// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+lsfe 2>&1 < %s| FileCheck %s

//------------------------------------------------------------------------------
// LDFMAXNM
//------------------------------------------------------------------------------

ldfmaxnm h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnm h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnm s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnm s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnm d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnm d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnm d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnm d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnm s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnm s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfmaxnma

ldfmaxnma h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnma h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnma s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnma s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnma d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnma d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnma d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnma d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnma s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnma s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfmaxnmal

ldfmaxnmal h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnmal h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnmal s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnmal s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnmal d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnmal d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnmal d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnmal d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnmal s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnmal s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfmaxnml

ldfmaxnml h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnml h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnml s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnml s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnml d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnml d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnml d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnml d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmaxnml s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmaxnml s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

//------------------------------------------------------------------------------
// LDBFMAXNM
//------------------------------------------------------------------------------

ldbfmaxnm s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnm s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnm h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnm h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnm s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnm s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnm d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnm d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnm h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnm h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnm h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnm h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfmaxnma

ldbfmaxnma s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnma s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnma h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnma h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnma s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnma s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnma d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnma d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnma h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnma h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnma h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnma h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfmaxnmal

ldbfmaxnmal s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnmal s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnmal h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnmal h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnmal s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnmal s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnmal d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnmal d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnmal h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnmal h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnmal h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnmal h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfmaxnml

ldbfmaxnml s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnml s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnml h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnml h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnml s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnml s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnml d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnml d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmaxnml h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnml h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

ldbfmaxnml h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmaxnml h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: