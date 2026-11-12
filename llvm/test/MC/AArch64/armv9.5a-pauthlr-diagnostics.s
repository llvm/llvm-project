// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+pauth-lr 2>&1 < %s | FileCheck %s

  autiasppc #2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: autiasppc #2
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

  autiasppc #1<<17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: autiasppc #1<<17
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

  autiasppc #-2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: autiasppc #-2
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

  autiasppc w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: autiasppc w0
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

  autiasppc sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: autiasppc sp
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

  retabsppc #2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: retabsppc #2
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

  retabsppc #(1<<17)
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: retabsppc #(1<<17)
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

  retabsppc #-2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: retabsppc #-2
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

  retaasppc w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: retaasppc w0
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

  retaasppc sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: retaasppc sp
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

  retaasppc xzr
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: retaasppc xzr
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

