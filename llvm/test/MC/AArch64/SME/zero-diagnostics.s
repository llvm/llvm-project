// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Registers list not in ascending order

zero    {za1.s, za0.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: warning: tile list not in ascending order
// CHECK-NEXT: zero {za1.s, za0.s}
// CHECK-NEXT:              ^

zero    {za0.d, za1.d, za4.d, za3.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: warning: tile list not in ascending order
// CHECK-NEXT: zero {za0.d, za1.d, za4.d, za3.d}
// CHECK-NEXT:                            ^

// --------------------------------------------------------------------------//
// Duplicate tile

zero    {za0.s, za0.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: warning: duplicate tile in list
// CHECK-NEXT: zero {za0.s, za0.s}
// CHECK-NEXT:              ^

zero    {za0.d, za1.d, za2.d, za2.d, za3.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: warning: duplicate tile in list
// CHECK-NEXT: zero {za0.d, za1.d, za2.d, za2.d, za3.d}
// CHECK-NEXT:                            ^

// --------------------------------------------------------------------------//
// Mismatched register size suffix

zero    {za0.b, za5.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: zero {za0.b, za5.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Missing '}'

zero    {za
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: '}' expected
// CHECK-NEXT: zero {za
// CHECK-NEXT:         ^
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid matrix tile

zero    {za0.b, za1.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: zero    {za0.b, za1.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zero    {za2.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: zero    {za2.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zero    {za0.s, za1.s, za2.s, za3.s, za4.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: zero    {za0.s, za1.s, za2.s, za3.s, za4.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zero    {za0.d, za1.d, za2.d, za3.d, za4.d, za5.d, za6.d, za7.d, za8.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: zero {za0.d, za1.d, za2.d, za3.d, za4.d, za5.d, za6.d, za7.d, za8.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zero    {za0h.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: zero    {za0h.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zero    {za0.s, za1h.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: zero    {za0.s, za1h.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zero    {za15.q}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: zero {za15.q}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
