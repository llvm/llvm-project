// RUN: mlir-opt -split-input-file -convert-func-to-spirv -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// func.return
//===----------------------------------------------------------------------===//

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @return_none_val
func.func @return_none_val() {
  // CHECK: spirv.Return
  return
}

// CHECK-LABEL: spirv.func @return_one_val
//  CHECK-SAME: (%[[ARG:.+]]: f32)
func.func @return_one_val(%arg0: f32) -> f32 {
  // CHECK: spirv.ReturnValue %[[ARG]] : f32
  return %arg0: f32
}

// Check that multiple-return functions are not converted.
// CHECK-LABEL: func @return_multi_val
func.func @return_multi_val(%arg0: f32) -> (f32, f32) {
  // CHECK: return
  return %arg0, %arg0: f32, f32
}

// CHECK-LABEL: spirv.func @return_one_index
//  CHECK-SAME: (%[[ARG:.+]]: i32)
func.func @return_one_index(%arg0: index) -> index {
  // CHECK: spirv.ReturnValue %[[ARG]] : i32
  return %arg0: index
}

// CHECK-LABEL: spirv.func @call_functions
//  CHECK-SAME: (%[[ARG:.+]]: i32)
func.func @call_functions(%arg0: index) -> index {
  // CHECK: spirv.FunctionCall @return_none_val() : () -> ()
  call @return_none_val(): () -> ()
  // CHECK: {{%.*}} = spirv.FunctionCall @return_one_index(%[[ARG]]) : (i32) -> i32
  %0 = call @return_one_index(%arg0): (index) -> index
  // CHECK: spirv.ReturnValue {{%.*}} : i32
  return %0: index
}

}

// -----
