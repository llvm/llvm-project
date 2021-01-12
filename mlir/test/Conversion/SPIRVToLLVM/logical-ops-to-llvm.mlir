// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.LogicalEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_equal_scalar
spv.func @logical_equal_scalar(%arg0: i1, %arg1: i1) "None" {
  // CHECK: llvm.icmp "eq" %{{.*}}, %{{.*}} : i1
  %0 = spv.LogicalEqual %arg0, %arg0 : i1
  spv.Return
}

// CHECK-LABEL: @logical_equal_vector
spv.func @logical_equal_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) "None" {
  // CHECK: llvm.icmp "eq" %{{.*}}, %{{.*}} : vector<4xi1>
  %0 = spv.LogicalEqual %arg0, %arg0 : vector<4xi1>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.LogicalNotEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_not_equal_scalar
spv.func @logical_not_equal_scalar(%arg0: i1, %arg1: i1) "None" {
  // CHECK: llvm.icmp "ne" %{{.*}}, %{{.*}} : i1
  %0 = spv.LogicalNotEqual %arg0, %arg0 : i1
  spv.Return
}

// CHECK-LABEL: @logical_not_equal_vector
spv.func @logical_not_equal_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) "None" {
  // CHECK: llvm.icmp "ne" %{{.*}}, %{{.*}} : vector<4xi1>
  %0 = spv.LogicalNotEqual %arg0, %arg0 : vector<4xi1>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.LogicalNot
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_not_scalar
spv.func @logical_not_scalar(%arg0: i1) "None" {
  // CHECK: %[[CONST:.*]] = llvm.mlir.constant(true) : i1
  // CHECK: llvm.xor %{{.*}}, %[[CONST]] : i1
  %0 = spv.LogicalNot %arg0 : i1
  spv.Return
}

// CHECK-LABEL: @logical_not_vector
spv.func @logical_not_vector(%arg0: vector<4xi1>) "None" {
  // CHECK: %[[CONST:.*]] = llvm.mlir.constant(dense<true> : vector<4xi1>) : vector<4xi1>
  // CHECK: llvm.xor %{{.*}}, %[[CONST]] : vector<4xi1>
  %0 = spv.LogicalNot %arg0 : vector<4xi1>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.LogicalAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_and_scalar
spv.func @logical_and_scalar(%arg0: i1, %arg1: i1) "None" {
  // CHECK: llvm.and %{{.*}}, %{{.*}} : i1
  %0 = spv.LogicalAnd %arg0, %arg0 : i1
  spv.Return
}

// CHECK-LABEL: @logical_and_vector
spv.func @logical_and_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) "None" {
  // CHECK: llvm.and %{{.*}}, %{{.*}} : vector<4xi1>
  %0 = spv.LogicalAnd %arg0, %arg0 : vector<4xi1>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.LogicalOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_or_scalar
spv.func @logical_or_scalar(%arg0: i1, %arg1: i1) "None" {
  // CHECK: llvm.or %{{.*}}, %{{.*}} : i1
  %0 = spv.LogicalOr %arg0, %arg0 : i1
  spv.Return
}

// CHECK-LABEL: @logical_or_vector
spv.func @logical_or_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) "None" {
  // CHECK: llvm.or %{{.*}}, %{{.*}} : vector<4xi1>
  %0 = spv.LogicalOr %arg0, %arg0 : vector<4xi1>
  spv.Return
}
