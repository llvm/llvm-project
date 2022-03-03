// RUN: mlir-opt -split-input-file -convert-func-to-spirv -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// func.return
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: spv.func @return_one_val
//  CHECK-SAME: (%[[ARG:.+]]: f32)
func @return_one_val(%arg0: f32) -> f32 {
  // CHECK: spv.ReturnValue %[[ARG]] : f32
  return %arg0: f32
}

// Check that multiple-return functions are not converted.
// CHECK-LABEL: func @return_multi_val
func @return_multi_val(%arg0: f32) -> (f32, f32) {
  // CHECK: return
  return %arg0, %arg0: f32, f32
}

}

// -----
