// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.LogicalEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_equal_scalar
spirv.func @logical_equal_scalar(%arg0: i1, %arg1: i1) "None" {
  // CHECK: llvm.icmp "eq" %{{.*}}, %{{.*}} : i1
  %0 = spirv.LogicalEqual %arg0, %arg1 : i1
  spirv.Return
}

// CHECK-LABEL: @logical_equal_vector
spirv.func @logical_equal_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) "None" {
  // CHECK: llvm.icmp "eq" %{{.*}}, %{{.*}} : vector<4xi1>
  %0 = spirv.LogicalEqual %arg0, %arg1 : vector<4xi1>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.LogicalNotEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_not_equal_scalar
spirv.func @logical_not_equal_scalar(%arg0: i1, %arg1: i1) "None" {
  // CHECK: llvm.icmp "ne" %{{.*}}, %{{.*}} : i1
  %0 = spirv.LogicalNotEqual %arg0, %arg1 : i1
  spirv.Return
}

// CHECK-LABEL: @logical_not_equal_vector
spirv.func @logical_not_equal_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) "None" {
  // CHECK: llvm.icmp "ne" %{{.*}}, %{{.*}} : vector<4xi1>
  %0 = spirv.LogicalNotEqual %arg0, %arg1 : vector<4xi1>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.LogicalNot
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_not_scalar
spirv.func @logical_not_scalar(%arg0: i1) "None" {
  // CHECK: %[[CONST:.*]] = llvm.mlir.constant(true) : i1
  // CHECK: llvm.xor %{{.*}}, %[[CONST]] : i1
  %0 = spirv.LogicalNot %arg0 : i1
  spirv.Return
}

// CHECK-LABEL: @logical_not_vector
spirv.func @logical_not_vector(%arg0: vector<4xi1>) "None" {
  // CHECK: %[[CONST:.*]] = llvm.mlir.constant(dense<true> : vector<4xi1>) : vector<4xi1>
  // CHECK: llvm.xor %{{.*}}, %[[CONST]] : vector<4xi1>
  %0 = spirv.LogicalNot %arg0 : vector<4xi1>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.LogicalAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_and_scalar
spirv.func @logical_and_scalar(%arg0: i1, %arg1: i1) "None" {
  // CHECK: llvm.and %{{.*}}, %{{.*}} : i1
  %0 = spirv.LogicalAnd %arg0, %arg1 : i1
  spirv.Return
}

// CHECK-LABEL: @logical_and_vector
spirv.func @logical_and_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) "None" {
  // CHECK: llvm.and %{{.*}}, %{{.*}} : vector<4xi1>
  %0 = spirv.LogicalAnd %arg0, %arg1 : vector<4xi1>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.LogicalOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_or_scalar
spirv.func @logical_or_scalar(%arg0: i1, %arg1: i1) "None" {
  // CHECK: llvm.or %{{.*}}, %{{.*}} : i1
  %0 = spirv.LogicalOr %arg0, %arg1 : i1
  spirv.Return
}

// CHECK-LABEL: @logical_or_vector
spirv.func @logical_or_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) "None" {
  // CHECK: llvm.or %{{.*}}, %{{.*}} : vector<4xi1>
  %0 = spirv.LogicalOr %arg0, %arg1 : vector<4xi1>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.Ordered
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @ordered_scalar
spirv.func @ordered_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fcmp "ord" %{{.*}}, %{{.*}} : f32
  %0 = spirv.Ordered %arg0, %arg1 : f32
  spirv.Return
}

// CHECK-LABEL: @ordered_vector
spirv.func @ordered_vector(%arg0: vector<4xf32>, %arg1: vector<4xf32>) "None" {
  // CHECK: llvm.fcmp "ord" %{{.*}}, %{{.*}} : vector<4xf32>
  %0 = spirv.Ordered %arg0, %arg1 : vector<4xf32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.Unordered
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @unordered_scalar
spirv.func @unordered_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fcmp "uno" %{{.*}}, %{{.*}} : f32
  %0 = spirv.Unordered %arg0, %arg1 : f32
  spirv.Return
}

// CHECK-LABEL: @unordered_vector
spirv.func @unordered_vector(%arg0: vector<4xf32>, %arg1: vector<4xf32>) "None" {
  // CHECK: llvm.fcmp "uno" %{{.*}}, %{{.*}} : vector<4xf32>
  %0 = spirv.Unordered %arg0, %arg1 : vector<4xf32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.IsInf
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @is_inf_scalar
spirv.func @is_inf_scalar(%arg0: f32) "None" {
  // CHECK: "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 516 : i32}> : (f32) -> i1
  %0 = spirv.IsInf %arg0 : f32
  spirv.Return
}

// CHECK-LABEL: @is_inf_vector
spirv.func @is_inf_vector(%arg0: vector<4xf64>) "None" {
  // CHECK: "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 516 : i32}> : (vector<4xf64>) -> vector<4xi1>
  %0 = spirv.IsInf %arg0 : vector<4xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.IsNan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @is_nan_scalar
spirv.func @is_nan_scalar(%arg0: f32) "None" {
  // CHECK: "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (f32) -> i1
  %0 = spirv.IsNan %arg0 : f32
  spirv.Return
}

// CHECK-LABEL: @is_nan_vector
spirv.func @is_nan_vector(%arg0: vector<2xf16>) "None" {
  // CHECK: "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (vector<2xf16>) -> vector<2xi1>
  %0 = spirv.IsNan %arg0 : vector<2xf16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.IsFinite
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @is_finite_scalar
spirv.func @is_finite_scalar(%arg0: f64) "None" {
  // CHECK: "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 504 : i32}> : (f64) -> i1
  %0 = spirv.IsFinite %arg0 : f64
  spirv.Return
}

// CHECK-LABEL: @is_finite_vector
spirv.func @is_finite_vector(%arg0: vector<3xf32>) "None" {
  // CHECK: "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 504 : i32}> : (vector<3xf32>) -> vector<3xi1>
  %0 = spirv.IsFinite %arg0 : vector<3xf32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.IsNormal
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @is_normal_scalar
spirv.func @is_normal_scalar(%arg0: f32) "None" {
  // CHECK: "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 264 : i32}> : (f32) -> i1
  %0 = spirv.IsNormal %arg0 : f32
  spirv.Return
}

// CHECK-LABEL: @is_normal_vector
spirv.func @is_normal_vector(%arg0: vector<4xf32>) "None" {
  // CHECK: "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 264 : i32}> : (vector<4xf32>) -> vector<4xi1>
  %0 = spirv.IsNormal %arg0 : vector<4xf32>
  spirv.Return
}
