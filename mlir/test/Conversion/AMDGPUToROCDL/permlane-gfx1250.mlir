// RUN: mlir-opt --convert-amdgpu-to-rocdl=triple=amdgpu12.50-amd-amdhsa --canonicalize %s | FileCheck %s

// gfx1250 has FeaturePermlane16Swap. It does not have FeaturePermlane32Swap;
// see permlane-gfx1250-invalid.mlir.

// CHECK-LABEL: func @permlane16_i32
// CHECK-SAME: (%[[ARG0:.*]]: i32)
func.func @permlane16_i32(%arg0 : i32) -> i32 {
// CHECK:  %[[PERM:.*]] = rocdl.permlane16.swap %[[ARG0]], %[[ARG0]], false, false : (i32, i32) -> <(i32, i32)>
  %0 = amdgpu.permlane_swap %arg0 16 : i32
  return %0 : i32
}
