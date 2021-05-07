// RUN: mlir-opt %s -canonicalize --split-input-file | FileCheck %s

// Test case: Basic folding of memref.dim(memref.tensor_load(m)) -> memref.dim(m).
// CHECK-LABEL: func @dim_of_tensor_load(
//  CHECK-SAME:     %[[MEMREF:[0-9a-z]*]]: memref<?xf32>
//       CHECK:   %[[C0:.*]] = constant 0
//       CHECK:   %[[D:.*]] = memref.dim %[[MEMREF]], %[[C0]]
//       CHECK:   return %[[D]] : index
func @dim_of_tensor_load(%arg0: memref<?xf32>) -> index {
  %c0 = constant 0 : index
  %0 = memref.tensor_load %arg0 : memref<?xf32>
  %1 = memref.dim %0, %c0 : tensor<?xf32>
  return %1 : index
}

// -----

// Test case: Folding of memref.load(memref.buffer_cast(%v, %idxs))
//            -> tensor.extract(%v, %idx)
// CHECK-LABEL: func @load_from_buffer_cast(
//  CHECK-SAME:     %[[IDX0:[0-9a-z]+]]: index, %[[IDX1:[0-9a-z]+]]: index
//  CHECK-SAME:     %[[TENSOR:[0-9a-z]+]]: tensor<?x?xf32>
//       CHECK:   %[[RES:.*]] = tensor.extract %[[TENSOR]][%[[IDX0]], %[[IDX1]]]
//   CHECK-NOT:   memref.load
//       CHECK:   return %[[RES]] : f32
func @load_from_buffer_cast(%arg0: index, %arg1: index, %arg2: tensor<?x?xf32>) -> f32 {
  %0 = memref.buffer_cast %arg2 : memref<?x?xf32>
  %1 = memref.load %0[%arg0, %arg1] : memref<?x?xf32>
  return %1 : f32
}

// -----

// Test case: Folding of memref.dim(tensor.generate %idx) -> %idx
// CHECK-LABEL: func @dim_of_tensor.generate(
//  CHECK-SAME:     %[[IDX0:[0-9a-z]+]]: index, %[[IDX1:[0-9a-z]+]]: index
//   CHECK-NOT:   memref.dim
//       CHECK:   return %[[IDX1]] : index
func @dim_of_tensor.generate(%arg0: index, %arg1: index) -> index {
  %c3 = constant 3 : index
  %0 = tensor.generate %arg0, %arg1 {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %c3 : index
  } : tensor<2x?x4x?x5xindex>
  %1 = memref.dim %0, %c3 : tensor<2x?x4x?x5xindex>
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

// Test case: Folding of memref.dim(memref.alloca(%size), %idx) -> %size
// CHECK-LABEL: func @dim_of_alloca(
//  CHECK-SAME:     %[[SIZE:[0-9a-z]+]]: index
//  CHECK-NEXT:   return %[[SIZE]] : index
func @dim_of_alloca(%size: index) -> index {
  %0 = memref.alloca(%size) : memref<?xindex>
  %c0 = constant 0 : index
  %1 = memref.dim %0, %c0 : memref<?xindex>
  return %1 : index
}

// -----

// Test case: Folding of memref.dim(memref.alloca(rank(%v)), %idx) -> rank(%v)
// CHECK-LABEL: func @dim_of_alloca_with_dynamic_size(
//  CHECK-SAME:     %[[MEM:[0-9a-z]+]]: memref<*xf32>
//  CHECK-NEXT:   %[[RANK:.*]] = rank %[[MEM]] : memref<*xf32>
//  CHECK-NEXT:   return %[[RANK]] : index
func @dim_of_alloca_with_dynamic_size(%arg0: memref<*xf32>) -> index {
  %0 = rank %arg0 : memref<*xf32>
  %1 = memref.alloca(%0) : memref<?xindex>
  %c0 = constant 0 : index
  %2 = memref.dim %1, %c0 : memref<?xindex>
  return %2 : index
}

// -----

// Test case: Folding of memref.dim(memref.reshape %v %shp, %idx) -> memref.load %shp[%idx]
// CHECK-LABEL: func @dim_of_memref_reshape(
//  CHECK-SAME:     %[[MEM:[0-9a-z]+]]: memref<*xf32>,
//  CHECK-SAME:     %[[SHP:[0-9a-z]+]]: memref<?xindex>
//  CHECK-NEXT:   %[[IDX:.*]] = constant 3
//  CHECK-NEXT:   %[[DIM:.*]] = memref.load %[[SHP]][%[[IDX]]]
//  CHECK-NEXT:   memref.store
//   CHECK-NOT:   memref.dim
//       CHECK:   return %[[DIM]] : index
func @dim_of_memref_reshape(%arg0: memref<*xf32>, %arg1: memref<?xindex>)
    -> index {
  %c3 = constant 3 : index
  %0 = memref.reshape %arg0(%arg1)
      : (memref<*xf32>, memref<?xindex>) -> memref<*xf32>
  // Update the shape to test that he load ends up in the right place.
  memref.store %c3, %arg1[%c3] : memref<?xindex>
  %1 = memref.dim %0, %c3 : memref<*xf32>
  return %1 : index
}

// -----

// Test case: Folding memref.dim(tensor.cast %0, %idx) -> memref.dim %0, %idx
// CHECK-LABEL: func @fold_dim_of_tensor.cast
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: tensor<4x?xf32>
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//       CHECK:   %[[T0:.+]] = memref.dim %[[ARG0]], %[[C1]]
//  CHECK-NEXT:   return %[[C4]], %[[T0]]
func @fold_dim_of_tensor.cast(%arg0 : tensor<4x?xf32>) -> (index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = tensor.cast %arg0 : tensor<4x?xf32> to tensor<?x?xf32>
  %1 = memref.dim %0, %c0 : tensor<?x?xf32>
  %2 = memref.dim %0, %c1 : tensor<?x?xf32>
  return %1, %2: index, index
}

// -----

// CHECK-LABEL: func @tensor_cast_to_memref
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<4x6x16x32xi8>
//       CHECK:   %[[M:.+]] = memref.buffer_cast %[[ARG0]] : memref<4x6x16x32xi8>
//       CHECK:   %[[M1:.+]] = memref.cast %[[M]] : memref<4x6x16x32xi8> to memref<?x?x16x32xi8>
//       CHECK:   return %[[M1]] : memref<?x?x16x32xi8>
func @tensor_cast_to_memref(%arg0 : tensor<4x6x16x32xi8>) ->
  memref<?x?x16x32xi8> {
  %0 = tensor.cast %arg0 : tensor<4x6x16x32xi8> to tensor<?x?x16x32xi8>
  %1 = memref.buffer_cast %0 : memref<?x?x16x32xi8>
  return %1 : memref<?x?x16x32xi8>
}

// -----

func @subtensor_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
    %arg2 : index) -> tensor<?x?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %0 = subtensor %arg0[%c0, %arg1, %c1] [%c4, %c1, %arg2] [%c1, %c1, %c1] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @subtensor_canonicalize
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?xf32>
//       CHECK:   %[[SUBTENSOR:.+]] = subtensor %[[ARG0]][0, %{{[a-zA-Z0-9_]+}}, 1]
//  CHECK-SAME:      [4, 1, %{{[a-zA-Z0-9_]+}}] [1, 1, 1]
//  CHECK-SAME:      : tensor<?x?x?xf32> to tensor<4x1x?xf32>
//       CHECK:   %[[RESULT:.+]] = tensor.cast %[[SUBTENSOR]]
//       CHEKC:   return %[[RESULT]]

// -----

func @rank_reducing_subtensor_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
    %arg2 : index) -> tensor<?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %0 = subtensor %arg0[%c0, %arg1, %c1] [%c4, 1, %arg2] [%c1, %c1, %c1] : tensor<?x?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @rank_reducing_subtensor_canonicalize
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?xf32>
//       CHECK:   %[[SUBTENSOR:.+]] = subtensor %[[ARG0]][0, %{{[a-zA-Z0-9_]+}}, 1]
//  CHECK-SAME:      [4, 1, %{{[a-zA-Z0-9_]+}}] [1, 1, 1]
//  CHECK-SAME:      : tensor<?x?x?xf32> to tensor<4x?xf32>
//       CHECK:   %[[RESULT:.+]] = tensor.cast %[[SUBTENSOR]]
//       CHEKC:   return %[[RESULT]]

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

// -----

func @subtensor_insert_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
    %arg2 : index, %arg3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %0 = subtensor_insert %arg0 into %arg3[%c0, %arg1, %c1] [%c4, %c1, %arg2] [%c1, %c1, %c1] : tensor<?x?x?xf32> into tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @subtensor_insert_canonicalize
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//       CHECK:   %[[RESULT:.+]] = subtensor_insert %[[ARG0]]
//  CHECK-SAME:      [0, %{{.+}}, 1] [4, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:      : tensor<?x?x?xf32> into tensor<?x?x?xf32>
//       CHEKC:   return %[[RESULT]]

// -----

func @subtensor_to_subtensor_insert_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
    %arg2 : index, %arg3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %0 = subtensor %arg0[%c0, %arg1, %c1] [%c4, %c1, %arg2] [%c1, %c1, %c1] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
  %1 = subtensor_insert %0 into %arg3[%c0, %arg1, %c1] [%c4, %c1, %arg2] [%c1, %c1, %c1] : tensor<?x?x?xf32> into tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @subtensor_to_subtensor_insert_canonicalize
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//       CHECK:   %[[SUBTENSOR:.+]] = subtensor %[[ARG0]]
//  CHECK-SAME:      [0, %{{.+}}, 1] [4, 1, %{{.+}} [1, 1, 1]
//  CHECK-SAME:      : tensor<?x?x?xf32> to tensor<4x1x?xf32>
//       CHECK:   %[[RESULT:.+]] = subtensor_insert %[[SUBTENSOR]]
//  CHECK-SAME:      [0, %{{.+}}, 1] [4, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:      : tensor<4x1x?xf32> into tensor<?x?x?xf32>
//       CHEKC:   return %[[RESULT]]

// -----

func @rank_reducing_subtensor_insert_canonicalize(%arg0 : tensor<?x?xf32>, %arg1 : index,
    %arg2 : index, %arg3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %0 = subtensor_insert %arg0 into %arg3[%c0, %arg1, %c1] [%c4, 1, %arg2] [%c1, %c1, %c1] : tensor<?x?xf32> into tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @rank_reducing_subtensor_insert_canonicalize
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]] = subtensor_insert %[[ARG0]]
//  CHECK-SAME:      [0, %{{.+}}, 1] [4, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:      : tensor<?x?xf32> into tensor<?x?x?xf32>
//       CHEKC:   return %[[RESULT]]

// -----

func @rank_reducing_subtensor_to_subtensor_insert_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
    %arg2 : index, %arg3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %0 = subtensor %arg0[%c0, %arg1, %c1] [%c4, 1, %arg2] [%c1, %c1, %c1] : tensor<?x?x?xf32> to tensor<?x?xf32>
  %1 = subtensor_insert %0 into %arg3[%c0, %arg1, %c1] [%c4, 1, %arg2] [%c1, %c1, %c1] : tensor<?x?xf32> into tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @rank_reducing_subtensor_to_subtensor_insert_canonicalize
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//       CHECK:   %[[SUBTENSOR:.+]] = subtensor %[[ARG0]]
//  CHECK-SAME:     [0, %{{.+}}, 1] [4, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:     : tensor<?x?x?xf32> to tensor<4x?xf32>
//       CHECK:   %[[RESULT:.+]] = subtensor_insert %[[SUBTENSOR]] into %[[ARG3]]
//  CHECK-SAME:      [0, %{{.+}}, 1] [4, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:      : tensor<4x?xf32> into tensor<?x?x?xf32>
//       CHEKC:   return %[[RESULT]]

// -----

func @subtensor_insert_propagate_dest_cast(%arg0 : tensor<2x?xi32>, %arg1 : tensor<i32>,
    %arg2 : index, %arg3 : index) -> tensor<?x?xi32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c8 = constant 8 : index
  %0 = memref.dim %arg0, %c1 : tensor<2x?xi32>
  %1 = tensor.extract %arg1[] : tensor<i32>
  %2 = tensor.generate %arg2, %c8 {
  ^bb0(%arg4: index, %arg5: index):
    tensor.yield %1 : i32
  } : tensor<?x?xi32>
  %3 = subtensor_insert %arg0 into %2[%c0, %arg3] [%c2, %0] [%c1, %c1] : tensor<2x?xi32> into tensor<?x?xi32>
  return %3 : tensor<?x?xi32>
}
// CHECK-LABEL: func @subtensor_insert_propagate_dest_cast
//       CHECK:   %[[UPDATED:.+]] = subtensor_insert %{{.+}} into %{{.+}}[0, %{{.+}}] [2, %{{.+}}] [1, 1]
//  CHECK-SAME:     tensor<2x?xi32> into tensor<?x8xi32>
//       CHECK:   %[[CAST:.+]] = tensor.cast %[[UPDATED]]
//       CHECK:   return %[[CAST]]

// -----

func @subtensor_insert_output_dest_canonicalize(%arg0 : tensor<2x3xi32>, %arg1 : tensor<i32>) -> tensor<3x9xi32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c9 = constant 9 : index
  %c3 = constant 3 : index
  %2 = tensor.extract %arg1[] : tensor<i32>
  %4 = tensor.generate %c3, %c9 {
  ^bb0(%arg2: index, %arg3: index):
    tensor.yield %2 : i32
  } : tensor<?x?xi32>
  %5 = subtensor_insert %arg0 into %4[%c0, %c1] [%c2, %c3] [1, 1] : tensor<2x3xi32> into tensor<?x?xi32>
  %6 = tensor.cast %5 : tensor<?x?xi32> to tensor<3x9xi32>
  return %6 : tensor<3x9xi32>
}
// CHECK-LABEL: func @subtensor_insert_output_dest_canonicalize
//  CHECK-SAME:   %[[ARG0:[a-zA-z0-9_]+]]: tensor<2x3xi32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<i32>
//       CHECK:   %[[PAD:.+]] = tensor.extract %[[ARG1]]
//       CHECK:   %[[GENERATE:.+]] = tensor.generate
//       CHECK:   %[[RESULT:.+]] = subtensor_insert %[[ARG0]] into %[[GENERATE]]
//       CHECK:   return %[[RESULT]]

// -----

// CHECK-LABEL: @select_same_val
//       CHECK:   return %arg1
func @select_same_val(%arg0: i1, %arg1: i64) -> i64 {
  %0 = select %arg0, %arg1, %arg1 : i64
  return %0 : i64
}

// -----

// CHECK-LABEL: @select_cmp_eq_select
//       CHECK:   return %arg1
func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
  %0 = cmpi eq, %arg0, %arg1 : i64
  %1 = select %0, %arg0, %arg1 : i64
  return %1 : i64
}

// -----

// CHECK-LABEL: @select_cmp_ne_select
//       CHECK:   return %arg0
func @select_cmp_ne_select(%arg0: i64, %arg1: i64) -> i64 {
  %0 = cmpi ne, %arg0, %arg1 : i64
  %1 = select %0, %arg0, %arg1 : i64
  return %1 : i64
}

// -----

// CHECK-LABEL: @indexCastOfSignExtend
//       CHECK:   %[[res:.+]] = index_cast %arg0 : i8 to index
//       CHECK:   return %[[res]]
func @indexCastOfSignExtend(%arg0: i8) -> index {
  %ext = sexti %arg0 : i8 to i16
  %idx = index_cast %ext : i16 to index
  return %idx : index
}

// CHECK-LABEL: @signExtendConstant
//       CHECK:   %[[cres:.+]] = constant -2 : i16
//       CHECK:   return %[[cres]]
func @signExtendConstant() -> i16 {
  %c-2 = constant -2 : i8
  %ext = sexti %c-2 : i8 to i16
  return %ext : i16
}

// CHECK-LABEL: @truncConstant
//       CHECK:   %[[cres:.+]] = constant -2 : i16
//       CHECK:   return %[[cres]]
func @truncConstant(%arg0: i8) -> i16 {
  %c-2 = constant -2 : i32
  %tr = trunci %c-2 : i32 to i16
  return %tr : i16
}

// -----

// CHECK-LABEL: @tripleAddAdd
//       CHECK:   %[[cres:.+]] = constant 59 : index 
//       CHECK:   %[[add:.+]] = addi %arg0, %[[cres]] : index 
//       CHECK:   return %[[add]]
func @tripleAddAdd(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = addi %c17, %arg0 : index
  %add2 = addi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleAddSub0
//       CHECK:   %[[cres:.+]] = constant 59 : index 
//       CHECK:   %[[add:.+]] = subi %[[cres]], %arg0 : index 
//       CHECK:   return %[[add]]
func @tripleAddSub0(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = subi %c17, %arg0 : index
  %add2 = addi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleAddSub1
//       CHECK:   %[[cres:.+]] = constant 25 : index 
//       CHECK:   %[[add:.+]] = addi %arg0, %[[cres]] : index 
//       CHECK:   return %[[add]]
func @tripleAddSub1(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = subi %arg0, %c17 : index
  %add2 = addi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubAdd0
//       CHECK:   %[[cres:.+]] = constant 25 : index 
//       CHECK:   %[[add:.+]] = subi %[[cres]], %arg0 : index 
//       CHECK:   return %[[add]]
func @tripleSubAdd0(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = addi %c17, %arg0 : index
  %add2 = subi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubAdd1
//       CHECK:   %[[cres:.+]] = constant -25 : index 
//       CHECK:   %[[add:.+]] = addi %arg0, %[[cres]] : index 
//       CHECK:   return %[[add]]
func @tripleSubAdd1(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = addi %c17, %arg0 : index
  %add2 = subi %add1, %c42 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubSub0
//       CHECK:   %[[cres:.+]] = constant 25 : index 
//       CHECK:   %[[add:.+]] = addi %arg0, %[[cres]] : index 
//       CHECK:   return %[[add]]
func @tripleSubSub0(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = subi %c17, %arg0 : index
  %add2 = subi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubSub1
//       CHECK:   %[[cres:.+]] = constant -25 : index 
//       CHECK:   %[[add:.+]] = subi %[[cres]], %arg0 : index 
//       CHECK:   return %[[add]]
func @tripleSubSub1(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = subi %c17, %arg0 : index
  %add2 = subi %add1, %c42 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubSub2
//       CHECK:   %[[cres:.+]] = constant 59 : index 
//       CHECK:   %[[add:.+]] = subi %[[cres]], %arg0 : index 
//       CHECK:   return %[[add]]
func @tripleSubSub2(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = subi %arg0, %c17 : index
  %add2 = subi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubSub3
//       CHECK:   %[[cres:.+]] = constant 59 : index 
//       CHECK:   %[[add:.+]] = subi %arg0, %[[cres]] : index 
//       CHECK:   return %[[add]]
func @tripleSubSub3(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = subi %arg0, %c17 : index
  %add2 = subi %add1, %c42 : index
  return %add2 : index
}

// CHECK-LABEL: @notCmpEQ
//       CHECK:   %[[cres:.+]] = cmpi ne, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpEQ(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "eq", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpEQ2
//       CHECK:   %[[cres:.+]] = cmpi ne, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpEQ2(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "eq", %arg0, %arg1 : i8
  %ncmp = xor %true, %cmp : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpNE
//       CHECK:   %[[cres:.+]] = cmpi eq, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpNE(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "ne", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSLT
//       CHECK:   %[[cres:.+]] = cmpi sge, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpSLT(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "slt", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSLE
//       CHECK:   %[[cres:.+]] = cmpi sgt, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpSLE(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "sle", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSGT
//       CHECK:   %[[cres:.+]] = cmpi sle, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpSGT(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "sgt", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSGE
//       CHECK:   %[[cres:.+]] = cmpi slt, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpSGE(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "sge", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpULT
//       CHECK:   %[[cres:.+]] = cmpi uge, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpULT(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "ult", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpULE
//       CHECK:   %[[cres:.+]] = cmpi ugt, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpULE(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "ule", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpUGT
//       CHECK:   %[[cres:.+]] = cmpi ule, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpUGT(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "ugt", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpUGE
//       CHECK:   %[[cres:.+]] = cmpi ult, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpUGE(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "uge", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}
