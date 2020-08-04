// RUN: mlir-opt -allow-unregistered-dialect %s -convert-std-to-llvm -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @address_space(
// CHECK-SAME:    !llvm.ptr<float, 7>
func @address_space(%arg0 : memref<32xf32, affine_map<(d0) -> (d0)>, 7>) {
  %0 = alloc() : memref<32xf32, affine_map<(d0) -> (d0)>, 5>
  %1 = constant 7 : index
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<float, 5>
  %2 = load %0[%1] : memref<32xf32, affine_map<(d0) -> (d0)>, 5>
  std.return
}

// CHECK-LABEL: func @strided_memref(
func @strided_memref(%ind: index) {
  %0 = alloc()[%ind] : memref<32x64xf32, affine_map<(i, j)[M] -> (32 + M * i + j)>>
  std.return
}

// -----

// CHECK-LABEL: func @rsqrt(
// CHECK-SAME: !llvm.float
func @rsqrt(%arg0 : f32) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
  // CHECK: %[[SQRT:.*]] = "llvm.intr.sqrt"(%arg0) : (!llvm.float) -> !llvm.float
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] : !llvm.float
  %0 = rsqrt %arg0 : f32
  std.return
}

// -----

// CHECK-LABEL: func @sine(
// CHECK-SAME: !llvm.float
func @sine(%arg0 : f32) {
  // CHECK: "llvm.intr.sin"(%arg0) : (!llvm.float) -> !llvm.float
  %0 = sin %arg0 : f32
  std.return
}

// -----

// CHECK-LABEL: func @rsqrt_double(
// CHECK-SAME: !llvm.double
func @rsqrt_double(%arg0 : f64) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f64) : !llvm.double
  // CHECK: %[[SQRT:.*]] = "llvm.intr.sqrt"(%arg0) : (!llvm.double) -> !llvm.double
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] : !llvm.double
  %0 = rsqrt %arg0 : f64
  std.return
}

// -----

// CHECK-LABEL: func @rsqrt_vector(
// CHECK-SAME: !llvm.vec<4 x float>
func @rsqrt_vector(%arg0 : vector<4xf32>) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : !llvm.vec<4 x float>
  // CHECK: %[[SQRT:.*]] = "llvm.intr.sqrt"(%arg0) : (!llvm.vec<4 x float>) -> !llvm.vec<4 x float>
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] : !llvm.vec<4 x float>
  %0 = rsqrt %arg0 : vector<4xf32>
  std.return
}

// -----

// CHECK-LABEL: func @rsqrt_multidim_vector(
// CHECK-SAME: !llvm.array<4 x vec<3 x float>>
func @rsqrt_multidim_vector(%arg0 : vector<4x3xf32>) {
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %arg0[0] : !llvm.array<4 x vec<3 x float>>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<3xf32>) : !llvm.vec<3 x float>
  // CHECK: %[[SQRT:.*]] = "llvm.intr.sqrt"(%[[EXTRACT]]) : (!llvm.vec<3 x float>) -> !llvm.vec<3 x float>
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] : !llvm.vec<3 x float>
  // CHECK: %[[INSERT:.*]] = llvm.insertvalue %[[DIV]], %0[0] : !llvm.array<4 x vec<3 x float>>
  %0 = rsqrt %arg0 : vector<4x3xf32>
  std.return
}

// -----

// Lowers `assert` to a function call to `abort` if the assertion is violated.
// CHECK: llvm.func @abort()
// CHECK-LABEL: @assert_test_function
// CHECK-SAME:  (%[[ARG:.*]]: !llvm.i1)
func @assert_test_function(%arg : i1) {
  // CHECK: llvm.cond_br %[[ARG]], ^[[CONTINUATION_BLOCK:.*]], ^[[FAILURE_BLOCK:.*]]
  // CHECK: ^[[CONTINUATION_BLOCK]]:
  // CHECK: llvm.return
  // CHECK: ^[[FAILURE_BLOCK]]:
  // CHECK: llvm.call @abort() : () -> ()
  // CHECK: llvm.unreachable
  assert %arg, "Computer says no"
  return
}

