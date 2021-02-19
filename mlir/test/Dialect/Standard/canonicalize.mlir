// RUN: mlir-opt %s -canonicalize --split-input-file | FileCheck %s

// -----

// Test case: Basic folding of tensor_load(tensor_to_memref(t)) -> t
// CHECK-LABEL:   func @tensor_load_of_tensor_to_memref(
// CHECK-SAME:                                          %[[TENSOR:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           return %[[TENSOR]]
func @tensor_load_of_tensor_to_memref(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tensor_to_memref %arg0 : memref<?xf32>
  %1 = tensor_load %0 : memref<?xf32>
  return %1 : tensor<?xf32>
}

// -----

// Test case: Basic folding of tensor_to_memref(tensor_load(m)) -> m
// CHECK-LABEL:   func @tensor_to_memref_of_tensor_load(
// CHECK-SAME:                                          %[[MEMREF:.*]]: memref<?xf32>) -> memref<?xf32> {
// CHECK:           return %[[MEMREF]]
func @tensor_to_memref_of_tensor_load(%arg0: memref<?xf32>) -> memref<?xf32> {
  %0 = tensor_load %arg0 : memref<?xf32>
  %1 = tensor_to_memref %0 : memref<?xf32>
  return %1 : memref<?xf32>
}

// -----

// Test case: If the memrefs are not the same type, don't fold them.
// Test case: If the memrefs are not cast-compatible (e.g. different address space),
// don't canonicalize them either.
// CHECK-LABEL:   func @no_fold_tensor_to_memref_of_tensor_load(
// CHECK-SAME:                                                  %[[MEMREF_ADDRSPACE2:.*]]: memref<?xf32, 2>) -> memref<?xf32, 7> {
// CHECK:           %[[TENSOR:.*]] = tensor_load %[[MEMREF_ADDRSPACE2]] : memref<?xf32, 2>
// CHECK:           %[[MEMREF_ADDRSPACE7:.*]] = tensor_to_memref %[[TENSOR]] : memref<?xf32, 7>
// CHECK:           return %[[MEMREF_ADDRSPACE7]]
func @no_fold_tensor_to_memref_of_tensor_load(%arg0: memref<?xf32, 2>) -> memref<?xf32, 7> {
  %0 = tensor_load %arg0 : memref<?xf32, 2>
  %1 = tensor_to_memref %0 : memref<?xf32, 7>
  return %1 : memref<?xf32, 7>
}

// -----

// CHECK-DAG: #[[$OFF_3:[a-z0-9]+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-DAG: #[[$OFF_UNK:[a-z0-9]+]] = affine_map<(d0)[s0] -> (d0 + s0)>

// Test case: If the memrefs are cast-compatible, canonicalize.
// CHECK-LABEL: func @canonicalize_tensor_to_memref_of_tensor_load(
//  CHECK-SAME:   %[[M:.*]]: memref<?xf32, #[[$OFF_3]]>) -> memref<?xf32, #[[$OFF_UNK]]> {
//   CHECK-NOT:   tensor_load
//   CHECK-NOT:   tensor_to_memref
//       CHECK:   %[[R:.*]] = memref_cast %[[M]] : memref<?xf32, #[[$OFF_3]]> to memref<?xf32, #[[$OFF_UNK]]>
//       CHECK:   return %[[R]]
func @canonicalize_tensor_to_memref_of_tensor_load(%arg0: memref<?xf32, offset: 3, strides: [1]>)
  -> memref<?xf32, offset: ?, strides: [1]>
{
  %0 = tensor_load %arg0 : memref<?xf32, offset: 3, strides: [1]>
  %1 = tensor_to_memref %0 : memref<?xf32, offset: ?, strides: [1]>
  return %1 : memref<?xf32, offset: ?, strides: [1]>
}

// -----

// Test case: Basic folding of dim(tensor_load(m)) -> dim(m).
// CHECK-LABEL: func @dim_of_tensor_load(
//  CHECK-SAME:     %[[MEMREF:[0-9a-z]*]]: memref<?xf32>
//       CHECK:   %[[C0:.*]] = constant 0
//       CHECK:   %[[D:.*]] = dim %[[MEMREF]], %[[C0]]
//       CHECK:   return %[[D]] : index
func @dim_of_tensor_load(%arg0: memref<?xf32>) -> index {
  %c0 = constant 0 : index
  %0 = tensor_load %arg0 : memref<?xf32>
  %1 = dim %0, %c0 : tensor<?xf32>
  return %1 : index
}

// -----

// Test case: Folding of load(tensor_to_memref(%v, %idxs))
//            -> tensor.extract(%v, %idx)
// CHECK-LABEL: func @load_from_tensor_to_memref(
//  CHECK-SAME:     %[[IDX0:[0-9a-z]+]]: index, %[[IDX1:[0-9a-z]+]]: index
//  CHECK-SAME:     %[[TENSOR:[0-9a-z]+]]: tensor<?x?xf32>
//       CHECK:   %[[RES:.*]] = tensor.extract %[[TENSOR]][%[[IDX0]], %[[IDX1]]]
//   CHECK-NOT:   load
//       CHECK:   return %[[RES]] : f32
func @load_from_tensor_to_memref(%arg0: index, %arg1: index, %arg2: tensor<?x?xf32>) -> f32 {
  %0 = tensor_to_memref %arg2 : memref<?x?xf32>
  %1 = load %0[%arg0, %arg1] : memref<?x?xf32>
  return %1 : f32
}

// -----

// Test case: Folding of dim(tensor.generate %idx) -> %idx
// CHECK-LABEL: func @dim_of_tensor.generate(
//  CHECK-SAME:     %[[IDX0:[0-9a-z]+]]: index, %[[IDX1:[0-9a-z]+]]: index
//   CHECK-NOT:   dim
//       CHECK:   return %[[IDX1]] : index
func @dim_of_tensor.generate(%arg0: index, %arg1: index) -> index {
  %c3 = constant 3 : index
  %0 = tensor.generate %arg0, %arg1 {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %c3 : index
  } : tensor<2x?x4x?x5xindex>
  %1 = dim %0, %c3 : tensor<2x?x4x?x5xindex>
  return %1 : index
}

// -----

// Test case: Folding of comparisons with equal operands.
// CHECK-LABEL: @cmpi_equal_operands
//   CHECK-DAG:   %[[T:.*]] = constant true
//   CHECK-DAG:   %[[F:.*]] = constant false
//       CHECK:   return %[[T]], %[[T]], %[[T]], %[[T]], %[[T]],
//  CHECK-SAME:          %[[F]], %[[F]], %[[F]], %[[F]], %[[F]]
func @cmpi_equal_operands(%arg0: i64)
    -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %0 = cmpi eq, %arg0, %arg0 : i64
  %1 = cmpi sle, %arg0, %arg0 : i64
  %2 = cmpi sge, %arg0, %arg0 : i64
  %3 = cmpi ule, %arg0, %arg0 : i64
  %4 = cmpi uge, %arg0, %arg0 : i64
  %5 = cmpi ne, %arg0, %arg0 : i64
  %6 = cmpi slt, %arg0, %arg0 : i64
  %7 = cmpi sgt, %arg0, %arg0 : i64
  %8 = cmpi ult, %arg0, %arg0 : i64
  %9 = cmpi ugt, %arg0, %arg0 : i64
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9
      : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// -----

// Test case: Folding of dim(memref_reshape %v %shp, %idx) -> load %shp[%idx]
// CHECK-LABEL: func @dim_of_memref_reshape(
//  CHECK-SAME:     %[[MEM:[0-9a-z]+]]: memref<*xf32>,
//  CHECK-SAME:     %[[SHP:[0-9a-z]+]]: memref<?xindex>
//  CHECK-NEXT:   %[[IDX:.*]] = constant 3
//  CHECK-NEXT:   %[[DIM:.*]] = load %[[SHP]][%[[IDX]]]
//  CHECK-NEXT:   store
//   CHECK-NOT:   dim
//       CHECK:   return %[[DIM]] : index
func @dim_of_memref_reshape(%arg0: memref<*xf32>, %arg1: memref<?xindex>)
    -> index {
  %c3 = constant 3 : index
  %0 = memref_reshape %arg0(%arg1)
      : (memref<*xf32>, memref<?xindex>) -> memref<*xf32>
  // Update the shape to test that he load ends up in the right place.
  store %c3, %arg1[%c3] : memref<?xindex>
  %1 = dim %0, %c3 : memref<*xf32>
  return %1 : index
}

// -----

// Test case: Folding dim(tensor.cast %0, %idx) -> dim %0, %idx
// CHECK-LABEL: func @fold_dim_of_tensor.cast
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: tensor<4x?xf32>
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//       CHECK:   %[[T0:.+]] = dim %[[ARG0]], %[[C1]]
//  CHECK-NEXT:   return %[[C4]], %[[T0]]
func @fold_dim_of_tensor.cast(%arg0 : tensor<4x?xf32>) -> (index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = tensor.cast %arg0 : tensor<4x?xf32> to tensor<?x?xf32>
  %1 = dim %0, %c0 : tensor<?x?xf32>
  %2 = dim %0, %c1 : tensor<?x?xf32>
  return %1, %2: index, index
}

// -----

// CHECK-LABEL: func @tensor_cast_to_memref
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<4x6x16x32xi8>
//       CHECK:   %[[M:.+]] = tensor_to_memref %[[ARG0]] : memref<4x6x16x32xi8>
//       CHECK:   %[[M1:.+]] = memref_cast %[[M]] : memref<4x6x16x32xi8> to memref<?x?x16x32xi8>
//       CHECK:   return %[[M1]] : memref<?x?x16x32xi8>
func @tensor_cast_to_memref(%arg0 : tensor<4x6x16x32xi8>) ->
  memref<?x?x16x32xi8> {
  %0 = tensor.cast %arg0 : tensor<4x6x16x32xi8> to tensor<?x?x16x32xi8>
  %1 = tensor_to_memref %0 : memref<?x?x16x32xi8>
  return %1 : memref<?x?x16x32xi8>
}

// -----

// CHECK-LABEL: func @subview_of_memcast
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: memref<4x6x16x32xi8>
//       CHECK:   %[[S:.+]] = subview %arg0[0, 1, 0] [1, 1, 16] [1, 1, 1] : memref<4x6x16x32xi8> to memref<16x32xi8, #{{.*}}>
//       CHECK:   %[[M:.+]] = memref_cast %[[S]] : memref<16x32xi8, #{{.*}}> to memref<16x32xi8, #{{.*}}>
//       CHECK:   return %[[M]] : memref<16x32xi8, #{{.*}}>
func @subview_of_memcast(%arg : memref<4x6x16x32xi8>) ->
  memref<16x32xi8, affine_map<(d0, d1)[s0] -> (d0 * 32 + d1 + s0)>>{
  %0 = memref_cast %arg : memref<4x6x16x32xi8> to memref<?x?x16x32xi8>
  %1 = subview %0[0, 1, 0] [1, 1, 16] [1, 1, 1] :
    memref<?x?x16x32xi8> to
    memref<16x32xi8, affine_map<(d0, d1)[s0] -> (d0 * 32 + d1 + s0)>>
  return %1 : memref<16x32xi8, affine_map<(d0, d1)[s0] -> (d0 * 32 + d1 + s0)>>
}

// -----

// CHECK-LABEL: func @trivial_subtensor
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: tensor<4x6x16x32xi8>
//   CHECK-NOT:   subtensor
//       CHECK:   return %[[ARG0]] :  tensor<4x6x16x32xi8>
func @trivial_subtensor(%arg0 : tensor<4x6x16x32xi8>) -> tensor<4x6x16x32xi8> {
  %0 = subtensor %arg0[0, 0, 0, 0] [4, 6, 16, 32] [1, 1, 1, 1] : tensor<4x6x16x32xi8> to tensor<4x6x16x32xi8>
  return %0 : tensor<4x6x16x32xi8>
}

// -----

// CHECK-LABEL: func @trivial_subtensor_insert
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: tensor<4x6x16x32xi8>
//   CHECK-NOT:   subtensor
//       CHECK:   return %[[ARG0]] :  tensor<4x6x16x32xi8>
func @trivial_subtensor_insert(%arg0 : tensor<4x6x16x32xi8>, %arg1 : tensor<4x6x16x32xi8>) -> tensor<4x6x16x32xi8> {
  %0 = subtensor_insert %arg0 into %arg1[0, 0, 0, 0] [4, 6, 16, 32] [1, 1, 1, 1] : tensor<4x6x16x32xi8> into tensor<4x6x16x32xi8>
  return %0 : tensor<4x6x16x32xi8>
}

// -----

// CHECK-LABEL: func @rank_reducing_tensor_of_cast
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: tensor<4x6x16x32xi8>
//       CHECK:   %[[S:.+]] = subtensor %arg0[0, 1, 0] [1, 1, 16] [1, 1, 1] : tensor<4x6x16x32xi8> to tensor<16x32xi8>
// Tensor cast is moved after subtensor and then gets canonicalized away.
//   CHECK-NOT:   tensor.cast
//       CHECK:   return %[[S]] : tensor<16x32xi8>
func @rank_reducing_tensor_of_cast(%arg : tensor<4x6x16x32xi8>) -> tensor<16x32xi8> {
  %0 = tensor.cast %arg : tensor<4x6x16x32xi8> to tensor<?x?x16x32xi8>
  %1 = subtensor %0[0, 1, 0] [1, 1, 16] [1, 1, 1] : tensor<?x?x16x32xi8> to tensor<16x32xi8>
  return %1 : tensor<16x32xi8>
}

// -----

// CHECK-LABEL: func @rank_reducing_subtensor_insert_of_cast
//  CHECK-SAME:   %[[A:.[a-z0-9A-Z_]+]]: tensor<16x32xi8>
//  CHECK-SAME:   %[[B:.[a-z0-9A-Z_]+]]: tensor<4x6x16x32xi8>
//       CHECK:   %[[S:.+]] = subtensor_insert %[[A]] into %[[B]][0, 1, 0] [1, 1, 16] [1, 1, 1] : tensor<16x32xi8> into tensor<4x6x16x32xi8>
// Tensor cast is folded away.
//   CHECK-NOT:   tensor.cast
//       CHECK:   return %[[S]] : tensor<4x6x16x32xi8>
func @rank_reducing_subtensor_insert_of_cast(%a : tensor<16x32xi8>, %b : tensor<4x6x16x32xi8>) -> tensor<4x6x16x32xi8> {
  %cast = tensor.cast %a : tensor<16x32xi8> to tensor<?x32xi8>
  %res = subtensor_insert %cast into %b[0, 1, 0] [1, 1, 16] [1, 1, 1] : tensor<?x32xi8> into tensor<4x6x16x32xi8>
  return %res : tensor<4x6x16x32xi8>
}
