// RUN: mlir-opt %s -canonicalize --split-input-file -allow-unregistered-dialect | FileCheck %s

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

// -----

// CHECK-LABEL: func @subview_of_memcast
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: memref<4x6x16x32xi8>
//       CHECK:   %[[S:.+]] = memref.subview %arg0[0, 1, 0] [1, 1, 16] [1, 1, 1] : memref<4x6x16x32xi8> to memref<16x32xi8, #{{.*}}>
//       CHECK:   %[[M:.+]] = memref.cast %[[S]] : memref<16x32xi8, #{{.*}}> to memref<16x32xi8, #{{.*}}>
//       CHECK:   return %[[M]] : memref<16x32xi8, #{{.*}}>
func @subview_of_memcast(%arg : memref<4x6x16x32xi8>) ->
  memref<16x32xi8, affine_map<(d0, d1)[s0] -> (d0 * 32 + d1 + s0)>>{
  %0 = memref.cast %arg : memref<4x6x16x32xi8> to memref<?x?x16x32xi8>
  %1 = memref.subview %0[0, 1, 0] [1, 1, 16] [1, 1, 1] :
    memref<?x?x16x32xi8> to
    memref<16x32xi8, affine_map<(d0, d1)[s0] -> (d0 * 32 + d1 + s0)>>
  return %1 : memref<16x32xi8, affine_map<(d0, d1)[s0] -> (d0 * 32 + d1 + s0)>>
}

// -----

// CHECK-LABEL: func @subview_of_static_full_size
// CHECK-SAME: %[[ARG0:.+]]: memref<4x6x16x32xi8>
// CHECK-NOT: memref.subview
// CHECK: return %[[ARG0]] : memref<4x6x16x32xi8>
func @subview_of_static_full_size(%arg0 : memref<4x6x16x32xi8>) -> memref<4x6x16x32xi8> {
  %0 = memref.subview %arg0[0, 0, 0, 0] [4, 6, 16, 32] [1, 1, 1, 1] : memref<4x6x16x32xi8> to memref<4x6x16x32xi8>
  return %0 : memref<4x6x16x32xi8>
}

// -----

#map0 = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>
func @subview_canonicalize(%arg0 : memref<?x?x?xf32>, %arg1 : index,
    %arg2 : index) -> memref<?x?x?xf32, #map0>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %0 = memref.subview %arg0[%c0, %arg1, %c1] [%c4, %c1, %arg2] [%c1, %c1, %c1] : memref<?x?x?xf32> to memref<?x?x?xf32, #map0>
  return %0 : memref<?x?x?xf32, #map0>
}
// CHECK-LABEL: func @subview_canonicalize
//  CHECK-SAME:   %[[ARG0:.+]]: memref<?x?x?xf32>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][0, %{{[a-zA-Z0-9_]+}}, 1]
//  CHECK-SAME:      [4, 1, %{{[a-zA-Z0-9_]+}}] [1, 1, 1]
//  CHECK-SAME:      : memref<?x?x?xf32> to memref<4x1x?xf32
//       CHECK:   %[[RESULT:.+]] = memref.cast %[[SUBVIEW]]
//       CHEKC:   return %[[RESULT]]

// -----

#map0 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
func @rank_reducing_subview_canonicalize(%arg0 : memref<?x?x?xf32>, %arg1 : index,
    %arg2 : index) -> memref<?x?xf32, #map0>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %0 = memref.subview %arg0[%c0, %arg1, %c1] [%c4, 1, %arg2] [%c1, %c1, %c1] : memref<?x?x?xf32> to memref<?x?xf32, #map0>
  return %0 : memref<?x?xf32, #map0>
}
// CHECK-LABEL: func @rank_reducing_subview_canonicalize
//  CHECK-SAME:   %[[ARG0:.+]]: memref<?x?x?xf32>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][0, %{{[a-zA-Z0-9_]+}}, 1]
//  CHECK-SAME:      [4, 1, %{{[a-zA-Z0-9_]+}}] [1, 1, 1]
//  CHECK-SAME:      : memref<?x?x?xf32> to memref<4x?xf32
//       CHECK:   %[[RESULT:.+]] = memref.cast %[[SUBVIEW]]
//       CHECK:   return %[[RESULT]]

// -----

// CHECK-LABEL: @clone_before_dealloc
// CHECK-SAME: %[[ARG:.*]]: memref<?xf32>
func @clone_before_dealloc(%arg0: memref<?xf32>) -> memref<?xf32> {
  // CHECK-NEXT: return %[[ARG]]
  %0 = memref.clone %arg0 : memref<?xf32> to memref<?xf32>
  memref.dealloc %arg0 : memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

// CHECK-LABEL: @clone_before_dealloc
// CHECK-SAME: %[[ARG:.*]]: memref<?xf32>
func @clone_before_dealloc(%arg0: memref<?xf32>) -> memref<?xf32> {
  // CHECK-NEXT: "use"(%arg0)
  // CHECK-NEXT: return %[[ARG]]
  %0 = memref.clone %arg0 : memref<?xf32> to memref<?xf32>
  "use"(%0) : (memref<?xf32>) -> ()
  memref.dealloc %0 : memref<?xf32>
  return %arg0 : memref<?xf32>
}

// -----

// CHECK-LABEL: @clone_after_cast
// CHECK-SAME: %[[ARG:.*]]: memref<?xf32>
func @clone_after_cast(%arg0: memref<?xf32>) -> memref<32xf32> {
  // CHECK-NEXT: memref.clone %[[ARG]] : memref<?xf32> to memref<32xf32>
  // CHECK-NOT: memref.cast
  %0 = memref.cast %arg0 : memref<?xf32> to memref<32xf32>
  %1 = memref.clone %0 : memref<32xf32> to memref<32xf32>
  return %1 : memref<32xf32>
}

// -----

// CHECK-LABEL: @clone_and_cast
// CHECK-SAME: %[[ARG:.*]]: memref<?xf32>
func @clone_and_cast(%arg0: memref<?xf32>) -> memref<32xf32> {
  // CHECK-NEXT: %[[RES:.*]] = memref.cast %[[ARG]] : memref<?xf32> to memref<32xf32>
  %0 = memref.clone %arg0 : memref<?xf32> to memref<32xf32>
  // CHECK-NEXT: return %[[RES]]
  memref.dealloc %arg0 : memref<?xf32>
  return %0 : memref<32xf32>
}

// -----

// CHECK-LABEL: @alias_is_freed
func @alias_is_freed(%arg0 : memref<?xf32>) {
  // CHECK: memref.clone
  // CHECK: memref.dealloc
  // CHECK: memref.dealloc
  %0 = memref.cast %arg0 : memref<?xf32> to memref<32xf32>
  %1 = memref.clone %0 : memref<32xf32> to memref<32xf32>
  memref.dealloc %arg0 : memref<?xf32>
  "use"(%1) : (memref<32xf32>) -> ()
  memref.dealloc %1 : memref<32xf32>
  return
}

// -----

// CHECK-LABEL: func @dim_of_sized_view
//  CHECK-SAME:   %{{[a-z0-9A-Z_]+}}: memref<?xi8>
//  CHECK-SAME:   %[[SIZE:.[a-z0-9A-Z_]+]]: index
//       CHECK:   return %[[SIZE]] : index
func @dim_of_sized_view(%arg : memref<?xi8>, %size: index) -> index {
  %c0 = constant 0 : index
  %0 = memref.reinterpret_cast %arg to offset: [0], sizes: [%size], strides: [0] : memref<?xi8> to memref<?xi8>
  %1 = memref.dim %0, %c0 : memref<?xi8>
  return %1 : index
}

// -----

// CHECK-LABEL: func @no_fold_of_store
//  CHECK:   %[[cst:.+]] = memref.cast %arg
//  CHECK:   memref.store %[[cst]]
func @no_fold_of_store(%arg : memref<32xi8>, %holder: memref<memref<?xi8>>) {
  %0 = memref.cast %arg : memref<32xi8> to memref<?xi8>
  memref.store %0, %holder[] : memref<memref<?xi8>>
  return
}

