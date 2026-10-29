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
