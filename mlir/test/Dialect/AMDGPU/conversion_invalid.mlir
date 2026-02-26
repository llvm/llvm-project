// RUN: mlir-opt %s -split-input-file --convert-amdgpu-to-rocdl 2>&1 | FileCheck %s

// CHECK: failed to legalize operation 'amdgpu.swizzle_bitmode'
func.func @swizzle_bitmode_non_multiple_of_32() {
  %5 = vector.constant_mask [42] : vector<42xi1>
  %6 = amdgpu.swizzle_bitmode %5 1 2 4 : vector<42xi1>
  return
}
