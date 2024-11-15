// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+lsfe 2>&1 < %s| FileCheck %s

//------------------------------------------------------------------------------
// ldfmin
//------------------------------------------------------------------------------

ldfminnm h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnm h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnm s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnm s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnm d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnm d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnm d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnm d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnm s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnm s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfminnma

ldfminnma h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnma h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnma s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnma s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnma d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnma d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnma d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnma d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnma s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnma s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfminnmal

ldfminnmal h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnmal h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnmal s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnmal s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnmal d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnmal d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnmal d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnmal d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnmal s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnmal s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfminnml

ldfminnml h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnml h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnml s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnml s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnml d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnml d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnml d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnml d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminnml s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminnml s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

//------------------------------------------------------------------------------
// LDBFMINNM
//------------------------------------------------------------------------------

ldbfminnm s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnm s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnm h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnm h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnm s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnm s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnm d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnm d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnm h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnm h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnm h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnm h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfminnma

ldbfminnma s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnma s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnma h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnma h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnma s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnma s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnma d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnma d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnma h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnma h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnma h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnma h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfminnmal

ldbfminnmal s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnmal s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnmal h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnmal h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnmal s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnmal s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnmal d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnmal d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnmal h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnmal h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnmal h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnmal h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfminnml

ldbfminnml s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnml s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnml h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnml h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnml s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnml s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnml d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnml d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnml h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnml h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminnml h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminnml h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: