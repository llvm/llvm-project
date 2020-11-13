// RUN: mlir-opt %s -tensor-constant-bufferize -split-input-file

// CHECK-LABEL: module {
// We check the debug name too since we put some effort into making that readable.
// The name isn't load-bearing though.
// CHECK: global_memref "private" constant @__constant_3x4xf32 : memref<3x4xf32> = dense<7.000000e+00>
// CHECK: @basic
func @basic() -> tensor<3x4xf32> {
  // CHECK: %[[MEMREF:.*]] = get_global_memref @__constant_3x4xf32 : memref<3x4xf32>
  // CHECK: %[[TENSOR:.*]] = tensor_load %[[MEMREF]]
  %0 = constant dense<7.0> : tensor<3x4xf32>
  // CHECK: return %[[TENSOR]]
  return %0 : tensor<3x4xf32>
}

// CHECK: }

// -----

// CHECK-LABEL: module {

// Only one global is created.
// CHECK: global_memref
// CHECK-NOT: global_memref
func @duplicate_constants() -> (tensor<3x4xf32>, tensor<3x4xf32>) {
  %0 = constant dense<7.0> : tensor<3x4xf32>
  %1 = constant dense<7.0> : tensor<3x4xf32>
  return %0, %1 : tensor<3x4xf32>, tensor<3x4xf32>
}

// CHECK: }

// -----

// CHECK-LABEL: module {

// Two globals are created.
// CHECK: global_memref
// CHECK: global_memref
// CHECK-NOT: global_memref
func @multiple_constants() -> (tensor<3x4xf32>, tensor<3x4xf32>) {
  %0 = constant dense<7.0> : tensor<3x4xf32>
  %1 = constant dense<8.0> : tensor<3x4xf32>
  return %0, %1 : tensor<3x4xf32>, tensor<3x4xf32>
}

// CHECK: }

// -----

// CHECK-LABEL: module {
// We don't convert non-tensor globals.
// CHECK-NOT: global_memref
func @non_tensor() {
    %0 = constant 7 : i32
    return
}

// CHECK: }
