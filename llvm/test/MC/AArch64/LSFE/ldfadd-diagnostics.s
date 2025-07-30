// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+lsfe 2>&1 < %s| FileCheck %s

//------------------------------------------------------------------------------
// LDFADD
//------------------------------------------------------------------------------

ldfadd h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfadd h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfadd s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfadd s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfadd d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfadd d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfadd d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfadd d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfadd s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfadd s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfadda

ldfadda h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfadda h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfadda s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfadda s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfadda d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfadda d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfadda d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfadda d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfadda s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfadda s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfaddal

ldfaddal h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfaddal h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfaddal s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfaddal s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfaddal d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfaddal d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfaddal d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfaddal d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfaddal s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfaddal s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfaddl

ldfaddl h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfaddl h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfaddl s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfaddl s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfaddl d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfaddl d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfaddl d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfaddl d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfaddl s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfaddl s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

//------------------------------------------------------------------------------
// LDBFADD
//------------------------------------------------------------------------------

ldbfadd s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfadd s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfadd h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfadd h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfadd s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfadd s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfadd d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfadd d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfadd h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfadd h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfadd h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfadd h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfadda

ldbfadda s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfadda s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfadda h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfadda h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfadda s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfadda s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfadda d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfadda d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfadda h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfadda h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfadda h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfadda h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfaddal

ldbfaddal s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfaddal s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfaddal h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfaddal h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfaddal s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfaddal s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfaddal d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfaddal d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfaddal h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfaddal h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfaddal h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfaddal h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfaddl

ldbfaddl s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfaddl s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfaddl h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfaddl h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfaddl s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfaddl s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfaddl d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfaddl d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfaddl h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfaddl h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfaddl h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfaddl h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: