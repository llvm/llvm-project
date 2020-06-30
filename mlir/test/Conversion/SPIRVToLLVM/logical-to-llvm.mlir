// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.LogicalEqual
//===----------------------------------------------------------------------===//

func @logical_equal_scalar(%arg0: i1, %arg1: i1) {
  // CHECK: %{{.*}} = llvm.icmp "eq" %{{.*}}, %{{.*}} : !llvm.i1
	%0 = spv.LogicalEqual %arg0, %arg0 : i1
  return
}

func @logical_equal_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) {
  // CHECK: %{{.*}} = llvm.icmp "eq" %{{.*}}, %{{.*}} : !llvm<"<4 x i1>">
	%0 = spv.LogicalEqual %arg0, %arg0 : vector<4xi1>
  return
}

//===----------------------------------------------------------------------===//
// spv.LogicalNotEqual
//===----------------------------------------------------------------------===//

func @logical_not_equal_scalar(%arg0: i1, %arg1: i1) {
  // CHECK: %{{.*}} = llvm.icmp "ne" %{{.*}}, %{{.*}} : !llvm.i1
	%0 = spv.LogicalNotEqual %arg0, %arg0 : i1
  return
}

func @logical_not_equal_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) {
  // CHECK: %{{.*}} = llvm.icmp "ne" %{{.*}}, %{{.*}} : !llvm<"<4 x i1>">
	%0 = spv.LogicalNotEqual %arg0, %arg0 : vector<4xi1>
  return
}

//===----------------------------------------------------------------------===//
// spv.LogicalNot
//===----------------------------------------------------------------------===//

func @logical_not__scalar(%arg0: i1) {
  // CHECK: %[[CONST:.*]] = llvm.mlir.constant(true) : !llvm.i1
  // CHECK: %{{.*}} = llvm.xor %{{.*}}, %[[CONST]] : !llvm.i1
	%0 = spv.LogicalNot %arg0 : i1
  return
}

func @logical_not_vector(%arg0: vector<4xi1>) {
  // CHECK: %[[CONST:.*]] = llvm.mlir.constant(dense<true> : vector<4xi1>) : !llvm<"<4 x i1>">
  // CHECK: %{{.*}} = llvm.xor %{{.*}}, %[[CONST]] : !llvm<"<4 x i1>">
	%0 = spv.LogicalNot %arg0 : vector<4xi1>
  return
}

//===----------------------------------------------------------------------===//
// spv.LogicalAnd
//===----------------------------------------------------------------------===//

func @logical_and_scalar(%arg0: i1, %arg1: i1) {
  // CHECK: %{{.*}} = llvm.and %{{.*}}, %{{.*}} : !llvm.i1
	%0 = spv.LogicalAnd %arg0, %arg0 : i1
  return
}

func @logical_and_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) {
  // CHECK: %{{.*}} = llvm.and %{{.*}}, %{{.*}} : !llvm<"<4 x i1>">
	%0 = spv.LogicalAnd %arg0, %arg0 : vector<4xi1>
  return
}

//===----------------------------------------------------------------------===//
// spv.LogicalOr
//===----------------------------------------------------------------------===//

func @logical_or_scalar(%arg0: i1, %arg1: i1) {
  // CHECK: %{{.*}} = llvm.or %{{.*}}, %{{.*}} : !llvm.i1
	%0 = spv.LogicalOr %arg0, %arg0 : i1
  return
}

func @logical_or_vector(%arg0: vector<4xi1>, %arg1: vector<4xi1>) {
  // CHECK: %{{.*}} = llvm.or %{{.*}}, %{{.*}} : !llvm<"<4 x i1>">
	%0 = spv.LogicalOr %arg0, %arg0 : vector<4xi1>
  return
}
