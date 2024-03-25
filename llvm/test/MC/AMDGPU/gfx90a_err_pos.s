// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck %s --implicit-check-not=error: --strict-whitespace

//==============================================================================
// vgpr must be even aligned

ds_gws_init a1 offset:65535 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: vgpr must be even aligned
// CHECK-NEXT:{{^}}ds_gws_init a1 offset:65535 gds
// CHECK-NEXT:{{^}}            ^

v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], v[2:9]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: source 2 operand must not partially overlap with dst
// CHECK-NEXT:{{^}}v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], v[2:9]
// CHECK-NEXT:{{^}}                                              ^
