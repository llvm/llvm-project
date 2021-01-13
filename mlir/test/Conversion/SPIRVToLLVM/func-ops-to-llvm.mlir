// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Return
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @return
spv.func @return() "None" {
  // CHECK: llvm.return
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ReturnValue
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @return_value
spv.func @return_value(%arg: i32) -> i32 "None" {
  // CHECK: llvm.return %{{.*}} : i32
  spv.ReturnValue %arg : i32
}

//===----------------------------------------------------------------------===//
// spv.func
//===----------------------------------------------------------------------===//

// CHECK-LABEL: llvm.func @none()
spv.func @none() "None" {
  spv.Return
}

// CHECK-LABEL: llvm.func @inline() attributes {passthrough = ["alwaysinline"]}
spv.func @inline() "Inline" {
  spv.Return
}

// CHECK-LABEL: llvm.func @dont_inline() attributes {passthrough = ["noinline"]}
spv.func @dont_inline() "DontInline" {
  spv.Return
}

// CHECK-LABEL: llvm.func @pure() attributes {passthrough = ["readonly"]}
spv.func @pure() "Pure" {
  spv.Return
}

// CHECK-LABEL: llvm.func @const() attributes {passthrough = ["readnone"]}
spv.func @const() "Const" {
  spv.Return
}

// CHECK-LABEL: llvm.func @scalar_types(%arg0: i32, %arg1: i1, %arg2: f64, %arg3: f32)
spv.func @scalar_types(%arg0: i32, %arg1: i1, %arg2: f64, %arg3: f32) "None" {
  spv.Return
}

// CHECK-LABEL: llvm.func @vector_types(%arg0: vector<2xi64>, %arg1: vector<2xi64>) -> vector<2xi64>
spv.func @vector_types(%arg0: vector<2xi64>, %arg1: vector<2xi64>) -> vector<2xi64> "None" {
  %0 = spv.IAdd %arg0, %arg1 : vector<2xi64>
  spv.ReturnValue %0 : vector<2xi64>
}

//===----------------------------------------------------------------------===//
// spv.FunctionCall
//===----------------------------------------------------------------------===//

// CHECK-LABEL: llvm.func @function_calls
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i1, %[[ARG2:.*]]: f64, %[[ARG3:.*]]: vector<2xi64>, %[[ARG4:.*]]: vector<2xf32>
spv.func @function_calls(%arg0: i32, %arg1: i1, %arg2: f64, %arg3: vector<2xi64>, %arg4: vector<2xf32>) "None" {
  // CHECK: llvm.call @void_1() : () -> ()
  // CHECK: llvm.call @void_2(%[[ARG3]]) : (vector<2xi64>) -> ()
  // CHECK: llvm.call @value_scalar(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (i32, i1, f64) -> i32
  // CHECK: llvm.call @value_vector(%[[ARG3]], %[[ARG4]]) : (vector<2xi64>, vector<2xf32>) -> vector<2xf32>
  spv.FunctionCall @void_1() : () -> ()
  spv.FunctionCall @void_2(%arg3) : (vector<2xi64>) -> ()
  %0 = spv.FunctionCall @value_scalar(%arg0, %arg1, %arg2) : (i32, i1, f64) -> i32
  %1 = spv.FunctionCall @value_vector(%arg3, %arg4) : (vector<2xi64>, vector<2xf32>) -> vector<2xf32>
  spv.Return
}

spv.func @void_1() "None" {
  spv.Return
}

spv.func @void_2(%arg0: vector<2xi64>) "None" {
  spv.Return
}

spv.func @value_scalar(%arg0: i32, %arg1: i1, %arg2: f64) -> i32 "None" {
  spv.ReturnValue %arg0: i32
}

spv.func @value_vector(%arg0: vector<2xi64>, %arg1: vector<2xf32>) -> vector<2xf32> "None" {
  spv.ReturnValue %arg1: vector<2xf32>
}
