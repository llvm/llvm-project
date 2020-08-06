// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Select
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @select_scalar
func @select_scalar(%arg0: i1, %arg1: vector<3xi32>, %arg2: f32) {
  // CHECK: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.vec<3 x i32>
  %0 = spv.Select %arg0, %arg1, %arg1 : i1, vector<3xi32>
  // CHECK: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.float
  %1 = spv.Select %arg0, %arg2, %arg2 : i1, f32
  return
}

// CHECK-LABEL: @select_vector
func @select_vector(%arg0: vector<2xi1>, %arg1: vector<2xi32>) {
  // CHECK: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.vec<2 x i1>, !llvm.vec<2 x i32>
  %0 = spv.Select %arg0, %arg1, %arg1 : vector<2xi1>, vector<2xi32>
  return
}

//===----------------------------------------------------------------------===//
// spv.Undef
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @undef_scalar
func @undef_scalar() {
  // CHECK: llvm.mlir.undef : !llvm.float
  %0 = spv.undef : f32
  return
}

// CHECK-LABEL: @undef_vector
func @undef_vector() {
  // CHECK: llvm.mlir.undef : !llvm.vec<2 x i32>
  %0 = spv.undef : vector<2xi32>
  return
}
