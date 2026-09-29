// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

uqincb wsp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqincb wsp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincb sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqincb sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincb z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqincb z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Operands not matching up (unsigned inc only has one register operand)

uqincb x0, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqincb x0, w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincb w0, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqincb w0, w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincb x0, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqincb x0, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Immediate not compatible with encode/decode function.

uqincb x0, all, mul #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: uqincb x0, all, mul #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincb x0, all, mul #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: uqincb x0, all, mul #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincb x0, all, mul #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: uqincb x0, all, mul #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate patterns

uqincb x0, vl512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqincb x0, vl512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincb x0, vl9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqincb x0, vl9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincb x0, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: uqincb x0, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincb x0, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: uqincb x0, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
