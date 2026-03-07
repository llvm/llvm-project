// RUN: mlir-opt -one-shot-bufferize="bufferize-function-boundaries" %s | FileCheck %s


// Test that foldMemRefCasts does not drop a precision-gaining return cast from
// unranked to ranked shape.
func.func private @callee() -> tensor<*xf32>

func.func @test_keep_ranked_return() -> tensor<8xf32> {
  %0 = func.call @callee() : () -> tensor<*xf32>
  %1 = tensor.cast %0 : tensor<*xf32> to tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-LABEL: func.func @test_keep_ranked_return()
// CHECK-SAME: -> memref<8xf32{{.*}}>
// CHECK: %[[SRC:.*]] = call @callee() : () -> memref<*xf32>
// CHECK: %[[CAST:.*]] = memref.cast %[[SRC]] : memref<*xf32> to memref<8xf32{{.*}}>
// CHECK: return %[[CAST]] : memref<8xf32{{.*}}>
// CHECK-NOT: return %[[SRC]] : memref<*xf32>

// Test that foldMemRefCasts does not drop a precision-gaining return cast from
// dynamic to static shape.
func.func private @callee_dynamic_shape() -> tensor<?xf32>

func.func @test_keep_static_shape_return() -> tensor<8xf32> {
  %0 = func.call @callee_dynamic_shape() : () -> tensor<?xf32>
  %1 = tensor.cast %0 : tensor<?xf32> to tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-LABEL: func.func @test_keep_static_shape_return()
// CHECK-SAME: -> memref<8xf32{{.*}}>
// CHECK: %[[SRC:.*]] = call @callee_dynamic_shape() : () -> memref<?xf32{{.*}}>
// CHECK: %[[CAST:.*]] = memref.cast %[[SRC]] : memref<?xf32{{.*}}> to memref<8xf32{{.*}}>
// CHECK: return %[[CAST]] : memref<8xf32{{.*}}>
// CHECK-NOT: return %[[SRC]] : memref<?xf32{{.*}}>

// Test current bufferization behavior for a tensor with layout annotation.
// This input does not currently produce a return memref.cast; the result stays
// in the strided memref form.

#src_layout = affine_map<(d0) -> (d0)>

func.func private @callee_layout() -> tensor<8xf32, #src_layout>

func.func @test_layout_return() -> tensor<8xf32> {
  %0 = func.call @callee_layout() : () -> tensor<8xf32, #src_layout>
  %1 = tensor.cast %0 : tensor<8xf32, #src_layout> to tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-LABEL: func.func @test_layout_return()
// CHECK-SAME: -> memref<8xf32, strided<[?], offset: ?>>
/// CHECK: %[[SRC:.*]] = call @callee_layout() : () -> memref<8xf32, strided<[?], offset: ?>>
/// CHECK-NOT: memref.cast
/// CHECK: return %[[SRC]] : memref<8xf32, strided<[?], offset: ?>>

