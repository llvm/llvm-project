// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.LogicalEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_equal_scalar
func @logical_equal_scalar(%arg0: i1, %arg1: i1) {
  // CHECK: llvm.icmp "eq" %{{.*}}, %{{.*}} : !llvm.i1
  %0 = spv.LogicalEqual %arg0, %arg0 : i1
  return
}

// CHECK-LABEL: @logical_equal_vector
func @logical_equal_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) {
  // CHECK: llvm.icmp "eq" %{{.*}}, %{{.*}} : !llvm.vec<4 x i1>
  %0 = spv.LogicalEqual %arg0, %arg0 : vector<4xi1>
  return
}

//===----------------------------------------------------------------------===//
// spv.LogicalNotEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_not_equal_scalar
func @logical_not_equal_scalar(%arg0: i1, %arg1: i1) {
  // CHECK: llvm.icmp "ne" %{{.*}}, %{{.*}} : !llvm.i1
  %0 = spv.LogicalNotEqual %arg0, %arg0 : i1
  return
}

// CHECK-LABEL: @logical_not_equal_vector
func @logical_not_equal_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) {
  // CHECK: llvm.icmp "ne" %{{.*}}, %{{.*}} : !llvm.vec<4 x i1>
  %0 = spv.LogicalNotEqual %arg0, %arg0 : vector<4xi1>
  return
}

//===----------------------------------------------------------------------===//
// spv.LogicalNot
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_not_scalar
func @logical_not_scalar(%arg0: i1) {
  // CHECK: %[[CONST:.*]] = llvm.mlir.constant(true) : !llvm.i1
  // CHECK: llvm.xor %{{.*}}, %[[CONST]] : !llvm.i1
  %0 = spv.LogicalNot %arg0 : i1
  return
}

// CHECK-LABEL: @logical_not_vector
func @logical_not_vector(%arg0: vector<4xi1>) {
  // CHECK: %[[CONST:.*]] = llvm.mlir.constant(dense<true> : vector<4xi1>) : !llvm.vec<4 x i1>
  // CHECK: llvm.xor %{{.*}}, %[[CONST]] : !llvm.vec<4 x i1>
  %0 = spv.LogicalNot %arg0 : vector<4xi1>
  return
}

//===----------------------------------------------------------------------===//
// spv.LogicalAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_and_scalar
func @logical_and_scalar(%arg0: i1, %arg1: i1) {
  // CHECK: llvm.and %{{.*}}, %{{.*}} : !llvm.i1
  %0 = spv.LogicalAnd %arg0, %arg0 : i1
  return
}

// CHECK-LABEL: @logical_and_vector
func @logical_and_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) {
  // CHECK: llvm.and %{{.*}}, %{{.*}} : !llvm.vec<4 x i1>
  %0 = spv.LogicalAnd %arg0, %arg0 : vector<4xi1>
  return
}

//===----------------------------------------------------------------------===//
// spv.LogicalOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_or_scalar
func @logical_or_scalar(%arg0: i1, %arg1: i1) {
  // CHECK: llvm.or %{{.*}}, %{{.*}} : !llvm.i1
  %0 = spv.LogicalOr %arg0, %arg0 : i1
  return
}

// CHECK-LABEL: @logical_or_vector
func @logical_or_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) {
  // CHECK: llvm.or %{{.*}}, %{{.*}} : !llvm.vec<4 x i1>
  %0 = spv.LogicalOr %arg0, %arg0 : vector<4xi1>
  return
}
