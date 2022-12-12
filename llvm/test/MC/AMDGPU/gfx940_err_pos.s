// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx940 %s 2>&1 | FileCheck %s --implicit-check-not=error: --strict-whitespace

//==============================================================================
// instruction must not use sc0

global_atomic_or v[0:1], v2, off sc1 nt sc0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must not use sc0
// CHECK-NEXT:{{^}}global_atomic_or v[0:1], v2, off sc1 nt sc0
// CHECK-NEXT:{{^}}                                        ^

global_atomic_or v[0:1], v2, off sc0 sc1 nt
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must not use sc0
// CHECK-NEXT:{{^}}global_atomic_or v[0:1], v2, off sc0 sc1 nt
// CHECK-NEXT:{{^}}                                 ^
