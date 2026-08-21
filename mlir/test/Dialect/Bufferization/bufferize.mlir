// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries" -split-input-file | FileCheck %s

// CHECK-LABEL: @alloc_tensor_static
func.func @alloc_tensor_static() -> tensor<8x16xf32> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x16xf32>
  // CHECK: return %[[ALLOC]]
  %0 = bufferization.alloc_tensor() : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

// The dynamic extents become the memref.alloc operands.

// CHECK-LABEL: @alloc_tensor_dynamic
// CHECK-SAME:    %[[D0:.*]]: index
func.func @alloc_tensor_dynamic(%d0: index) -> tensor<?x16xf32> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[D0]]) {alignment = 64 : i64} : memref<?x16xf32>
  // CHECK: return %[[ALLOC]]
  %0 = bufferization.alloc_tensor(%d0) : tensor<?x16xf32>
  return %0 : tensor<?x16xf32>
}

// -----

// The `memory_space` attribute wins over `defaultMemorySpaceFn`. The result
// stays inside the function so the function-boundary type conversion does not
// drive the allocated type.

// CHECK-LABEL: @alloc_tensor_memory_space
func.func @alloc_tensor_memory_space(%i: index) -> f32 {
  // CHECK: memref.alloc() {alignment = 64 : i64} : memref<8x16xf32, 1>
  %0 = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<8x16xf32>
  %1 = tensor.extract %0[%i, %i] : tensor<8x16xf32>
  return %1 : f32
}

// -----

// A `copy` operand allocates a fresh buffer and copies into it.

// CHECK-LABEL: @alloc_tensor_copy
// CHECK-SAME:    %[[ARG:.*]]: memref<8x16xf32
func.func @alloc_tensor_copy(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x16xf32>
  // CHECK: memref.copy %[[ARG]], %[[ALLOC]]
  // CHECK: return %[[ALLOC]]
  %0 = bufferization.alloc_tensor() copy(%arg0) : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

// An unused alloc_tensor is erased rather than allocated.

// CHECK-LABEL: @alloc_tensor_dead
// CHECK-NOT:     memref.alloc
func.func @alloc_tensor_dead() {
  %0 = bufferization.alloc_tensor() : tensor<8x16xf32>
  return
}

// -----

// CHECK-LABEL: @dealloc_tensor
func.func @dealloc_tensor() {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x16xf32>
  // CHECK: memref.dealloc %[[ALLOC]]
  %0 = bufferization.alloc_tensor() : tensor<8x16xf32>
  bufferization.dealloc_tensor %0 : tensor<8x16xf32>
  return
}

// -----

// A tensor destination is written in place and returned.

// CHECK-LABEL: @materialize_in_destination_tensor
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9_]*]]: memref<5xf32,
// CHECK-SAME:    %[[DST:[a-zA-Z0-9_]*]]: memref<5xf32,
func.func @materialize_in_destination_tensor(%src: tensor<5xf32>, %dst: tensor<5xf32>) -> tensor<5xf32> {
  // CHECK: memref.copy %[[SRC]], %[[DST]]
  // CHECK: return %[[DST]]
  %0 = bufferization.materialize_in_destination %src in %dst : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

// A memref destination is copied into directly and the op has no result.

// CHECK-LABEL: @materialize_in_destination_memref
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9_]*]]: memref<5xf32,
// CHECK-SAME:    %[[DST:[a-zA-Z0-9_]*]]: memref<5xf32>
func.func @materialize_in_destination_memref(%src: tensor<5xf32>, %dst: memref<5xf32>) {
  // CHECK: memref.copy %[[SRC]], %[[DST]]
  bufferization.materialize_in_destination %src in restrict writable %dst
      : (tensor<5xf32>, memref<5xf32>) -> ()
  return
}

// -----

// Without `writable`, the buffer of a to_tensor must not be written, so the
// insert bufferizes out of place.

// CHECK-LABEL: @to_tensor_not_writable
// CHECK-SAME:    %[[M:[a-zA-Z0-9_]*]]: memref<5xf32>
func.func @to_tensor_not_writable(%m: memref<5xf32>, %f: f32, %idx: index) -> tensor<5xf32> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK: memref.copy %[[M]], %[[ALLOC]]
  // CHECK: memref.store %{{.*}}, %[[ALLOC]]
  %t = bufferization.to_tensor %m restrict : memref<5xf32> to tensor<5xf32>
  %r = tensor.insert %f into %t[%idx] : tensor<5xf32>
  return %r : tensor<5xf32>
}

// -----

// With `writable`, the insert bufferizes in place.

// CHECK-LABEL: @to_tensor_writable
// CHECK-SAME:    %[[M:[a-zA-Z0-9_]*]]: memref<5xf32>
// CHECK-NOT:     memref.alloc
func.func @to_tensor_writable(%m: memref<5xf32>, %f: f32, %idx: index) -> tensor<5xf32> {
  // CHECK: memref.store %{{.*}}, %[[M]]
  %t = bufferization.to_tensor %m restrict writable : memref<5xf32> to tensor<5xf32>
  %r = tensor.insert %f into %t[%idx] : tensor<5xf32>
  return %r : tensor<5xf32>
}

// -----

// to_buffer/to_tensor pairs fold away.

// CHECK-LABEL: @to_buffer_of_to_tensor
// CHECK-SAME:    %[[M:[a-zA-Z0-9_]*]]: memref<5xf32>
// CHECK-NOT:     bufferization.to_tensor
// CHECK-NOT:     bufferization.to_buffer
func.func @to_buffer_of_to_tensor(%m: memref<5xf32>, %f: f32, %idx: index) {
  // CHECK: memref.store %{{.*}}, %[[M]]
  %t = bufferization.to_tensor %m restrict writable : memref<5xf32> to tensor<5xf32>
  %r = bufferization.to_buffer %t : tensor<5xf32> to memref<5xf32>
  memref.store %f, %r[%idx] : memref<5xf32>
  return
}

// -----

// A `read_only` to_buffer does not write, so no copy of the source buffer is
// needed.

// CHECK-LABEL: @to_buffer_read_only
// CHECK-SAME:    %[[M:[a-zA-Z0-9_]*]]: memref<5xf32>
// CHECK-NOT:     memref.alloc
func.func @to_buffer_read_only(%m: memref<5xf32>, %idx: index) -> f32 {
  // CHECK: memref.load %[[M]]
  %t = bufferization.to_tensor %m restrict : memref<5xf32> to tensor<5xf32>
  %r = bufferization.to_buffer %t read_only : tensor<5xf32> to memref<5xf32>
  %v = memref.load %r[%idx] : memref<5xf32>
  return %v : f32
}
