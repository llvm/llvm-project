// RUN: mlir-opt %s -canonicalize --split-input-file | FileCheck %s

// Test case: Basic folding of memref.tensor_load(memref.buffer_cast(t)) -> t
// CHECK-LABEL: func @tensor_load_of_buffer_cast(
//  CHECK-SAME:   %[[TENSOR:.*]]: tensor<?xf32>) -> tensor<?xf32> {
//       CHECK: return %[[TENSOR]]
func @tensor_load_of_buffer_cast(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = memref.buffer_cast %arg0 : memref<?xf32>
  %1 = memref.tensor_load %0 : memref<?xf32>
  return %1 : tensor<?xf32>
}

// -----

// Test case: Basic folding of memref.buffer_cast(memref.tensor_load(m)) -> m
// CHECK-LABEL: func @buffer_cast_of_tensor_load(
//  CHECK-SAME:   %[[MEMREF:.*]]: memref<?xf32>) -> memref<?xf32> {
//       CHECK: return %[[MEMREF]]
func @buffer_cast_of_tensor_load(%arg0: memref<?xf32>) -> memref<?xf32> {
  %0 = memref.tensor_load %arg0 : memref<?xf32>
  %1 = memref.buffer_cast %0 : memref<?xf32>
  return %1 : memref<?xf32>
}

// -----

// Test case: If the memrefs are not the same type, don't fold them.
// Test case: If the memrefs are not cast-compatible (e.g. different address space),
// don't canonicalize them either.
// CHECK-LABEL: func @no_fold_buffer_cast_of_tensor_load(
//  CHECK-SAME:   %[[MEMREF_ADDRSPACE2:.*]]: memref<?xf32, 2>)
//  CHECK-SAME:     -> memref<?xf32, 7> {
//       CHECK: %[[TENSOR:.*]] = memref.tensor_load
//  CHECK_SAME:   %[[MEMREF_ADDRSPACE2]] : memref<?xf32, 2>
//       CHECK: %[[MEMREF_ADDRSPACE7:.*]] = memref.buffer_cast
//  CHECK_SAME:   %[[TENSOR]] : memref<?xf32, 7>
//       CHECK: return %[[MEMREF_ADDRSPACE7]]
func @no_fold_buffer_cast_of_tensor_load(%arg0: memref<?xf32, 2>) -> memref<?xf32, 7> {
  %0 = memref.tensor_load %arg0 : memref<?xf32, 2>
  %1 = memref.buffer_cast %0 : memref<?xf32, 7>
  return %1 : memref<?xf32, 7>
}

// -----

// CHECK-DAG: #[[$OFF_3:[a-z0-9]+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-DAG: #[[$OFF_UNK:[a-z0-9]+]] = affine_map<(d0)[s0] -> (d0 + s0)>

// Test case: If the memrefs are cast-compatible, canonicalize.
// CHECK-LABEL: func @canonicalize_buffer_cast_of_tensor_load(
//  CHECK-SAME:   %[[M:.*]]: memref<?xf32, #[[$OFF_3]]>)
//  CHEKC-SAME:     -> memref<?xf32, #[[$OFF_UNK]]> {
//   CHECK-NOT: memref.tensor_load
//   CHECK-NOT: memref.buffer_cast
//       CHECK: %[[R:.*]] = memref.cast %[[M]]
//  CHECK-SAME:   memref<?xf32, #[[$OFF_3]]> to memref<?xf32, #[[$OFF_UNK]]>
//       CHECK: return %[[R]]
func @canonicalize_buffer_cast_of_tensor_load(%arg0: memref<?xf32, offset: 3, strides: [1]>)
  -> memref<?xf32, offset: ?, strides: [1]>
{
  %0 = memref.tensor_load %arg0 : memref<?xf32, offset: 3, strides: [1]>
  %1 = memref.buffer_cast %0 : memref<?xf32, offset: ?, strides: [1]>
  return %1 : memref<?xf32, offset: ?, strides: [1]>
}
