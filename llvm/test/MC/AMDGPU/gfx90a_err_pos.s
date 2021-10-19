// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck %s --implicit-check-not=error: --strict-whitespace

//==============================================================================
// vgpr must be even aligned

ds_gws_init a1 offset:65535 gds
// CHECK: error: vgpr must be even aligned
// CHECK-NEXT:{{^}}ds_gws_init a1 offset:65535 gds
// CHECK-NEXT:{{^}}            ^
