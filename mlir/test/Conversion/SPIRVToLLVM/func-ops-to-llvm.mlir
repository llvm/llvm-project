// RUN: mlir-opt -convert-spirv-to-llvm='use-opaque-pointers=1' %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.Return
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @return
spirv.func @return() "None" {
  // CHECK: llvm.return
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.ReturnValue
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @return_value
spirv.func @return_value(%arg: i32) -> i32 "None" {
  // CHECK: llvm.return %{{.*}} : i32
  spirv.ReturnValue %arg : i32
}

//===----------------------------------------------------------------------===//
// spirv.func
//===----------------------------------------------------------------------===//

// CHECK-LABEL: llvm.func @none()
spirv.func @none() "None" {
  spirv.Return
}

// CHECK-LABEL: llvm.func @inline() attributes {passthrough = ["alwaysinline"]}
spirv.func @inline() "Inline" {
  spirv.Return
}

// CHECK-LABEL: llvm.func @dont_inline() attributes {passthrough = ["noinline"]}
spirv.func @dont_inline() "DontInline" {
  spirv.Return
}

// CHECK-LABEL: llvm.func @pure() attributes {passthrough = ["readonly"]}
spirv.func @pure() "Pure" {
  spirv.Return
}

// CHECK-LABEL: llvm.func @const() attributes {passthrough = ["readnone"]}
spirv.func @const() "Const" {
  spirv.Return
}

// CHECK-LABEL: llvm.func @scalar_types(%arg0: i32, %arg1: i1, %arg2: f64, %arg3: f32)
spirv.func @scalar_types(%arg0: i32, %arg1: i1, %arg2: f64, %arg3: f32) "None" {
  spirv.Return
}

// CHECK-LABEL: llvm.func @vector_types(%arg0: vector<2xi64>, %arg1: vector<2xi64>) -> vector<2xi64>
spirv.func @vector_types(%arg0: vector<2xi64>, %arg1: vector<2xi64>) -> vector<2xi64> "None" {
  %0 = spirv.IAdd %arg0, %arg1 : vector<2xi64>
  spirv.ReturnValue %0 : vector<2xi64>
}

//===----------------------------------------------------------------------===//
// spirv.FunctionCall
//===----------------------------------------------------------------------===//

// CHECK-LABEL: llvm.func @function_calls
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i1, %[[ARG2:.*]]: f64, %[[ARG3:.*]]: vector<2xi64>, %[[ARG4:.*]]: vector<2xf32>
spirv.func @function_calls(%arg0: i32, %arg1: i1, %arg2: f64, %arg3: vector<2xi64>, %arg4: vector<2xf32>) "None" {
  // CHECK: llvm.call @void_1() : () -> ()
  // CHECK: llvm.call @void_2(%[[ARG3]]) : (vector<2xi64>) -> ()
  // CHECK: llvm.call @value_scalar(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (i32, i1, f64) -> i32
  // CHECK: llvm.call @value_vector(%[[ARG3]], %[[ARG4]]) : (vector<2xi64>, vector<2xf32>) -> vector<2xf32>
  spirv.FunctionCall @void_1() : () -> ()
  spirv.FunctionCall @void_2(%arg3) : (vector<2xi64>) -> ()
  %0 = spirv.FunctionCall @value_scalar(%arg0, %arg1, %arg2) : (i32, i1, f64) -> i32
  %1 = spirv.FunctionCall @value_vector(%arg3, %arg4) : (vector<2xi64>, vector<2xf32>) -> vector<2xf32>
  spirv.Return
}

spirv.func @void_1() "None" {
  spirv.Return
}

spirv.func @void_2(%arg0: vector<2xi64>) "None" {
  spirv.Return
}

spirv.func @value_scalar(%arg0: i32, %arg1: i1, %arg2: f64) -> i32 "None" {
  spirv.ReturnValue %arg0: i32
}

spirv.func @value_vector(%arg0: vector<2xi64>, %arg1: vector<2xf32>) -> vector<2xf32> "None" {
  spirv.ReturnValue %arg1: vector<2xf32>
}
