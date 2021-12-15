// RUN: mlir-opt %s -tensor-constant-bufferize -split-input-file | FileCheck %s
// RUN: mlir-opt %s -tensor-constant-bufferize=alignment=64 -split-input-file | FileCheck --check-prefix=ALIGNED %s

// CHECK-LABEL: module {

// We check the debug name too since we put some effort into making that readable.
// The name isn't load-bearing though.

// CHECK: memref.global "private" constant @__constant_3x4xf32 : memref<3x4xf32> = dense<7.000000e+00>
// CHECK-NOT: alignment

// ALIGNED: memref.global "private" constant @__constant_3x4xf32 : memref<3x4xf32> = dense<7.000000e+00>
// ALIGNED-SAME: {alignment = 64 : i64}

// CHECK: @basic
func @basic() -> tensor<3x4xf32> {
  // CHECK: %[[MEMREF:.*]] = memref.get_global @__constant_3x4xf32 : memref<3x4xf32>
  // CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]
  %0 = arith.constant dense<7.0> : tensor<3x4xf32>
  // CHECK: return %[[TENSOR]]
  return %0 : tensor<3x4xf32>
}

// CHECK: }

// -----

// CHECK-LABEL: module {

// Only one global is created.
// CHECK: memref.global
// CHECK-NOT: memref.global
func @duplicate_constants() -> (tensor<3x4xf32>, tensor<3x4xf32>) {
  %0 = arith.constant dense<7.0> : tensor<3x4xf32>
  %1 = arith.constant dense<7.0> : tensor<3x4xf32>
  return %0, %1 : tensor<3x4xf32>, tensor<3x4xf32>
}

// CHECK: }

// -----

// CHECK-LABEL: module {

// Two globals are created.
// CHECK: memref.global
// CHECK: memref.global
// CHECK-NOT: memref.global
func @multiple_constants() -> (tensor<3x4xf32>, tensor<3x4xf32>) {
  %0 = arith.constant dense<7.0> : tensor<3x4xf32>
  %1 = arith.constant dense<8.0> : tensor<3x4xf32>
  return %0, %1 : tensor<3x4xf32>, tensor<3x4xf32>
}

// CHECK: }

// -----

// CHECK-LABEL: module {
// We don't convert non-tensor globals.
// CHECK-NOT: memref.global
func @non_tensor() {
    %0 = arith.constant 7 : i32
    return
}

// CHECK: }
