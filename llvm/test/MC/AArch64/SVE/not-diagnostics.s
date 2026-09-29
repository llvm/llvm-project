// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid predicate

not     z31.b, p8/m, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: not     z31.b, p8/m, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element width

not     z31.b, p7/m, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: not     z31.b, p7/m, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
