// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+lsfe 2>&1 < %s| FileCheck %s

//------------------------------------------------------------------------------
// LDFMIN
//------------------------------------------------------------------------------

ldfmin h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmin h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmin s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmin s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmin d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmin d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmin d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmin d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmin s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmin s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfmina

ldfmina h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmina h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmina s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmina s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmina d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmina d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmina d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmina d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfmina s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfmina s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfminal

ldfminal h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminal h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminal s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminal s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminal d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminal d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminal d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminal d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminal s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminal s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldfminl

ldfminl h0, s2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminl h0, s2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminl s0, d2, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminl s0, d2, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminl d0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminl d0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminl d0, d1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminl d0, d1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldfminl s0, s1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldfminl s0, s1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

//------------------------------------------------------------------------------
// LDBFMIN
//------------------------------------------------------------------------------

ldbfmin s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmin s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmin h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmin h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmin s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmin s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmin d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmin d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmin h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmin h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmin h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmin h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfmina

ldbfmina s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmina s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmina h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmina h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmina s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmina s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmina d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmina d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmina h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmina h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfmina h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfmina h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfminal

ldbfminal s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminal s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminal h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminal h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminal s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminal s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminal d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminal d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminal h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminal h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminal h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminal h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- ldbfminl

ldbfminl s0, h1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminl s0, h1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminl h0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminl h0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminl s0, s1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminl s0, s1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminl d0, d1, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminl d0, d1, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminl h0, h1, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminl h0, h1, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldbfminl h0, h1, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldbfminl h0, h1, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: