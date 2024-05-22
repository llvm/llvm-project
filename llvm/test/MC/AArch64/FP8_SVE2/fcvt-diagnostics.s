// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+fp8 2>&1 < %s | FileCheck %s
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+fp8 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//

f1cvt z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: f1cvt z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f1cvt z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: f1cvt z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f1cvt z32.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: f1cvt z32.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


f2cvt z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: f2cvt z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f2cvt z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: f2cvt z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f2cvt z32.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: f2cvt z32.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


bf1cvt z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bf1cvt z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf1cvt z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bf1cvt z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf1cvt z32.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bf1cvt z32.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


bf2cvt z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bf2cvt z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf2cvt z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bf2cvt z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf2cvt z32.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bf2cvt z32.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


f1cvtlt z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: f1cvtlt z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f1cvtlt z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: f1cvtlt z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f1cvtlt z32.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: f1cvtlt z32.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


f2cvtlt z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: f2cvtlt z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f2cvtlt z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: f2cvtlt z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f2cvtlt z32.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: f2cvtlt z32.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


bf1cvtlt z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bf1cvtlt z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf1cvtlt z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bf1cvtlt z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf1cvtlt z32.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bf1cvtlt z32.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


bf2cvtlt z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bf2cvtlt z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf2cvtlt z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bf2cvtlt z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf2cvtlt z32.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bf2cvtlt z32.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: