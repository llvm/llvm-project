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

// CHECK-LABEL: func @ceilf(
// CHECK-SAME: !llvm.float
func @ceilf(%arg0 : f32) {
  // CHECK: "llvm.intr.ceil"(%arg0) : (!llvm.float) -> !llvm.float
  %0 = ceilf %arg0 : f32
  std.return
}

// -----

// CHECK-LABEL: func @floorf(
// CHECK-SAME: !llvm.float
func @floorf(%arg0 : f32) {
  // CHECK: "llvm.intr.floor"(%arg0) : (!llvm.float) -> !llvm.float
  %0 = floorf %arg0 : f32
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

// -----

// CHECK-LABEL: func @transpose
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.insertvalue {{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[3, 2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[3, 2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
func @transpose(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  %0 = transpose %arg0 (i, j, k) -> (k, i, j) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2 * s1 + s0 + d0 * s2 + d1)>>
  return
}

// -----

// CHECK: llvm.mlir.global external @gv0() : !llvm.array<2 x float>
global_memref @gv0 : memref<2xf32> = uninitialized

// CHECK: llvm.mlir.global private @gv1() : !llvm.array<2 x float>
global_memref "private" @gv1 : memref<2xf32>

// CHECK: llvm.mlir.global external @gv2(dense<{{\[\[}}0.000000e+00, 1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00, 5.000000e+00]]> : tensor<2x3xf32>) : !llvm.array<2 x array<3 x float>>
global_memref @gv2 : memref<2x3xf32> = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]>

// Test 1D memref.
// CHECK-LABEL: func @get_gv0_memref
func @get_gv0_memref() {
  %0 = get_global_memref @gv0 : memref<2xf32>
  // CHECK: %[[DIM:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
  // CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @gv0 : !llvm.ptr<array<2 x float>>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDR]][%[[ZERO]], %[[ZERO]]] : (!llvm.ptr<array<2 x float>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<float>
  // CHECK: %[[DEADBEEF:.*]] = llvm.mlir.constant(3735928559 : index) : !llvm.i64
  // CHECK: %[[DEADBEEFPTR:.*]] = llvm.inttoptr %[[DEADBEEF]] : !llvm.i64 to !llvm.ptr<float>
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[DEADBEEFPTR]], {{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[GEP]], {{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM]], {{.*}}[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[STRIDE]], {{.*}}[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  return
}

// Test 2D memref.
// CHECK-LABEL: func @get_gv2_memref
func @get_gv2_memref() {
  // CHECK: %[[DIM0:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
  // CHECK: %[[DIM1:.*]] = llvm.mlir.constant(3 : index) : !llvm.i64
  // CHECK: %[[STRIDE1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @gv2 : !llvm.ptr<array<2 x array<3 x float>>>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDR]][%[[ZERO]], %[[ZERO]], %[[ZERO]]] : (!llvm.ptr<array<2 x array<3 x float>>>, !llvm.i64, !llvm.i64, !llvm.i64) -> !llvm.ptr<float>
  // CHECK: %[[DEADBEEF:.*]] = llvm.mlir.constant(3735928559 : index) : !llvm.i64
  // CHECK: %[[DEADBEEFPTR:.*]] = llvm.inttoptr %[[DEADBEEF]] : !llvm.i64 to !llvm.ptr<float>
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DEADBEEFPTR]], {{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[GEP]], {{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM0]], {{.*}}[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM1]], {{.*}}[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM1]], {{.*}}[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[STRIDE1]], {{.*}}[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>

  %0 = get_global_memref @gv2 : memref<2x3xf32>
  return
}

// Test scalar memref.
// CHECK: llvm.mlir.global external @gv3(1.000000e+00 : f32) : !llvm.float
global_memref @gv3 : memref<f32> = dense<1.0>

// CHECK-LABEL: func @get_gv3_memref
func @get_gv3_memref() {
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @gv3 : !llvm.ptr<float>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDR]][%[[ZERO]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  // CHECK: %[[DEADBEEF:.*]] = llvm.mlir.constant(3735928559 : index) : !llvm.i64
  // CHECK: %[[DEADBEEFPTR:.*]] = llvm.inttoptr %[[DEADBEEF]] : !llvm.i64 to !llvm.ptr<float>
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64)>
  // CHECK: llvm.insertvalue %[[DEADBEEFPTR]], {{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
  // CHECK: llvm.insertvalue %[[GEP]], {{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
  %0 = get_global_memref @gv3 : memref<f32>
  return
}

// This should not trigger an assertion by creating an LLVM::CallOp with a
// nullptr result type.

// CHECK-LABEL: @call_zero_result_func
func @call_zero_result_func() {
  // CHECK: call @zero_result_func
  call @zero_result_func() : () -> ()
  return
}
func private @zero_result_func()
