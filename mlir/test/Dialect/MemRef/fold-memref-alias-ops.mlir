// RUN: mlir-opt -fold-memref-alias-ops -split-input-file %s | FileCheck %s

func.func @fold_static_stride_subview_with_load(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) -> f32 {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] : memref<12x32xf32> to memref<4x4xf32, strided<[64, 3], offset: ?>>
  %1 = memref.load %0[%arg3, %arg4] : memref<4x4xf32, strided<[64, 3], offset: ?>>
  return %1 : f32
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 + s1 * 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 + s1 * 3)>
//      CHECK: func @fold_static_stride_subview_with_load
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]], %[[ARG3]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG4]]]
//      CHECK:   memref.load %[[ARG0]][%[[I1]], %[[I2]]]

// -----

func.func @fold_dynamic_stride_subview_with_load(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index) -> f32 {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] :
    memref<12x32xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
  %1 = memref.load %0[%arg3, %arg4] : memref<4x4xf32, strided<[?, ?], offset: ?>>
  return %1 : f32
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
//      CHECK: func @fold_dynamic_stride_subview_with_load
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG5:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG6:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG3]], %[[ARG5]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[ARG4]], %[[ARG6]]]
//      CHECK:   memref.load %[[ARG0]][%[[I1]], %[[I2]]]

// -----

func.func @fold_static_stride_subview_with_store(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : f32) {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] :
    memref<12x32xf32> to memref<4x4xf32, strided<[64, 3], offset: ?>>
  memref.store %arg5, %0[%arg3, %arg4] : memref<4x4xf32, strided<[64, 3], offset: ?>>
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 + s1 * 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 + s1 * 3)>
//      CHECK: func @fold_static_stride_subview_with_store
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]], %[[ARG3]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG4]]]
//      CHECK:   memref.store %{{.+}}, %[[ARG0]][%[[I1]], %[[I2]]]

// -----

func.func @fold_dynamic_stride_subview_with_store(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index, %arg7 : f32) {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] :
    memref<12x32xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
  memref.store %arg7, %0[%arg3, %arg4] : memref<4x4xf32, strided<[?, ?], offset: ?>>
  return
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
//      CHECK: func @fold_dynamic_stride_subview_with_store
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG5:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG6:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG3]], %[[ARG5]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[ARG4]], %[[ARG6]]]
//      CHECK:   memref.store %{{.+}}, %[[ARG0]][%[[I1]], %[[I2]]]

// -----

func.func @fold_subview_with_transfer_read_0d(
  %arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index)
    -> vector<f32> {
  %f1 = arith.constant 1.0 : f32
  %0 = memref.subview %arg0[%arg1, %arg2][1, 1][1, 1] : memref<12x32xf32> to memref<f32, strided<[], offset: ?>>
  %1 = vector.transfer_read %0[], %f1 : memref<f32, strided<[], offset: ?>>, vector<f32>
  return %1 : vector<f32>
}
//      CHECK: func @fold_subview_with_transfer_read_0d
// CHECK-SAME:   %[[MEM:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[SZ0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[SZ1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ST1:[a-zA-Z0-9_]+]]: index
//      CHECK:   vector.transfer_read %[[MEM]][%[[SZ0]], %[[SZ1]]]

// -----

func.func @fold_subview_with_transfer_read(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index) -> vector<4xf32> {
  %f1 = arith.constant 1.0 : f32

  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] : memref<12x32xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
  %1 = vector.transfer_read %0[%arg3, %arg4], %f1 {in_bounds = [true]} : memref<4x4xf32, strided<[?, ?], offset: ?>>, vector<4xf32>
  return %1 : vector<4xf32>
}
//      CHECK: func @fold_subview_with_transfer_read
// Can't fold this atm since we don't emit the proper vector.extract_strided_slice.
//   CHECK: memref.subview

// -----

func.func @fold_static_stride_subview_with_transfer_write_0d(
    %arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index,
    %v : vector<f32>) {
  %f1 = arith.constant 1.0 : f32
  %0 = memref.subview %arg0[%arg1, %arg2][1, 1][1, 1] : memref<12x32xf32> to memref<f32, strided<[], offset: ?>>
  vector.transfer_write %v, %0[] {in_bounds = []} : vector<f32>, memref<f32, strided<[], offset: ?>>
  return
}
//      CHECK: func @fold_static_stride_subview_with_transfer_write_0d
// CHECK-SAME:   %[[MEM:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[SZ0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[SZ1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ST1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[V:[a-zA-Z0-9_]+]]: vector<f32>
//      CHECK:   vector.transfer_write %[[V]], %[[MEM]][%[[SZ0]], %[[SZ1]]]

// -----

func.func @fold_static_stride_subview_with_transfer_write(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5: index, %arg6 : index, %arg7 : vector<4xf32>) {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] :
    memref<12x32xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
  vector.transfer_write %arg7, %0[%arg3, %arg4] {in_bounds = [true]} : vector<4xf32>, memref<4x4xf32, strided<[?, ?], offset: ?>>
  return
}
//      CHECK: func @fold_static_stride_subview_with_transfer_write
// Can't fold this atm since we don't emit the proper vector.extract_strided_slice.
//   CHECK: memref.subview

// -----

func.func @fold_rank_reducing_subview_with_load
    (%arg0 : memref<?x?x?x?x?x?xf32>, %arg1 : index, %arg2 : index,
     %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index,
     %arg7 : index, %arg8 : index, %arg9 : index, %arg10: index,
     %arg11 : index, %arg12 : index, %arg13 : index, %arg14: index,
     %arg15 : index, %arg16 : index) -> f32 {
  %0 = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4, %arg5, %arg6][4, 1, 1, 4, 1, 1][%arg7, %arg8, %arg9, %arg10, %arg11, %arg12] : memref<?x?x?x?x?x?xf32> to memref<4x1x4x1xf32, strided<[?, ?, ?, ?], offset: ?>>
  %1 = memref.load %0[%arg13, %arg14, %arg15, %arg16] : memref<4x1x4x1xf32, strided<[?, ?, ?, ?], offset: ?>>
  return %1 : f32
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
//      CHECK: func @fold_rank_reducing_subview_with_load
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?x?x?x?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG5:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG6:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG7:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG8:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG9:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG10:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG11:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG12:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG13:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG14:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG15:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG16:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[I0:.+]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG13]], %[[ARG7]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP]]()[%[[ARG3]], %[[ARG14]], %[[ARG9]]]
//  CHECK-DAG:   %[[I3:.+]] = affine.apply #[[MAP]]()[%[[ARG4]], %[[ARG15]], %[[ARG10]]]
//  CHECK-DAG:   %[[I4:.+]] = affine.apply #[[MAP]]()[%[[ARG5]], %[[ARG16]], %[[ARG11]]]
//      CHECK:   memref.load %[[ARG0]][%[[I0]], %[[ARG2]], %[[I2]], %[[I3]], %[[I4]], %[[ARG6]]]

// -----

func.func @fold_vector_transfer_read_with_rank_reduced_subview(
    %arg0 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>,
    %arg1: index, %arg2 : index, %arg3 : index, %arg4: index, %arg5 : index,
    %arg6 : index) -> vector<4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = memref.subview %arg0[0, %arg1, %arg2] [1, %arg3, %arg4] [1, 1, 1]
      : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to
        memref<?x?xf32, strided<[?, ?], offset: ?>>
  %1 = vector.transfer_read %0[%arg5, %arg6], %cst {in_bounds = [true]}
      : memref<?x?xf32, strided<[?, ?], offset: ?>>, vector<4xf32>
  return %1 : vector<4xf32>
}
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func @fold_vector_transfer_read_with_rank_reduced_subview
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG5:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG6:[a-zA-Z0-9]+]]: index
//   CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:    %[[IDX0:.+]] = affine.apply #[[MAP1]]()[%[[ARG1]], %[[ARG5]]]
//   CHECK-DAG:    %[[IDX1:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG6]]]
//       CHECK:    vector.transfer_read %[[ARG0]][%[[C0]], %[[IDX0]], %[[IDX1]]], %{{.*}} : memref<?x?x?xf32

// -----

func.func @fold_vector_transfer_write_with_rank_reduced_subview(
    %arg0 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>,
    %arg1 : vector<4xf32>, %arg2: index, %arg3 : index, %arg4 : index,
    %arg5: index, %arg6 : index, %arg7 : index) {
  %cst = arith.constant 0.0 : f32
  %0 = memref.subview %arg0[0, %arg2, %arg3] [1, %arg4, %arg5] [1, 1, 1]
      : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to
        memref<?x?xf32, strided<[?, ?], offset: ?>>
  vector.transfer_write %arg1, %0[%arg6, %arg7] {in_bounds = [true]}
      : vector<4xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
  return
}
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func @fold_vector_transfer_write_with_rank_reduced_subview
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: vector<4xf32>
//  CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG5:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG6:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG7:[a-zA-Z0-9]+]]: index
//   CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:    %[[IDX0:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG6]]]
//   CHECK-DAG:    %[[IDX1:.+]] = affine.apply #[[MAP1]]()[%[[ARG3]], %[[ARG7]]]
//   CHECK-DAG:    vector.transfer_write %[[ARG1]], %[[ARG0]][%[[C0]], %[[IDX0]], %[[IDX1]]] {in_bounds = [true]} : vector<4xf32>, memref<?x?x?xf32

// -----

func.func @fold_vector_transfer_write_with_inner_rank_reduced_subview(
    %arg0 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>,
    %arg1 : vector<4xf32>, %arg2: index, %arg3 : index, %arg4 : index,
    %arg5: index, %arg6 : index, %arg7 : index) {
  %cst = arith.constant 0.0 : f32
  %0 = memref.subview %arg0[%arg2, %arg3, 0] [%arg4, %arg5, 1] [1, 1, 1]
      : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to
        memref<?x?xf32, strided<[?, ?], offset: ?>>
  vector.transfer_write %arg1, %0[%arg6, %arg7] {in_bounds = [true]}
      : vector<4xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
  return
}
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1)>
//       CHECK: func @fold_vector_transfer_write_with_inner_rank_reduced_subview
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: vector<4xf32>
//  CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG5:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG6:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG7:[a-zA-Z0-9]+]]: index
//   CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:    %[[IDX0:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG6]]]
//   CHECK-DAG:    %[[IDX1:.+]] = affine.apply #[[MAP1]]()[%[[ARG3]], %[[ARG7]]]
//   CHECK-DAG:    vector.transfer_write %[[ARG1]], %[[ARG0]][%[[IDX0]], %[[IDX1]], %[[C0]]]
//  CHECK-SAME:    {in_bounds = [true], permutation_map = #[[MAP2]]} : vector<4xf32>, memref<?x?x?xf32

// -----

func.func @fold_masked_vector_transfer_read_with_subview(
    %arg0 : memref<?x?xf32, strided<[?, ?], offset: ?>>,
    %arg1: index, %arg2 : index, %arg3 : index, %arg4: index, %arg5 : index,
    %arg6 : index, %mask : vector<4xi1>) -> vector<4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = memref.subview %arg0[%arg1, %arg2] [%arg3, %arg4] [1, 1]
      : memref<?x?xf32, strided<[?, ?], offset: ?>> to
        memref<?x?xf32, strided<[?, ?], offset: ?>>
  %1 = vector.transfer_read %0[%arg5, %arg6], %cst, %mask {in_bounds = [true]}
      : memref<?x?xf32, strided<[?, ?], offset: ?>>, vector<4xf32>
  return %1 : vector<4xf32>
}
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func @fold_masked_vector_transfer_read_with_subview
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?xf32, strided<[?, ?], offset: ?>>
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG5:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG6:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[MASK:[a-zA-Z0-9]+]]: vector<4xi1>
//   CHECK-DAG:    %[[IDX0:.+]] = affine.apply #[[MAP1]]()[%[[ARG1]], %[[ARG5]]]
//   CHECK-DAG:    %[[IDX1:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG6]]]
//       CHECK:    vector.transfer_read %[[ARG0]][%[[IDX0]], %[[IDX1]]], %{{.*}}, %[[MASK]] {{.*}} : memref<?x?xf32

// -----

func.func @fold_masked_vector_transfer_read_with_rank_reducing_subview(
    %arg0 : memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>,
    %arg1: index, %arg2 : index, %arg3 : index, %arg4: index, %arg5 : index,
    %arg6 : index, %mask : vector<4x3xi1>) -> vector<3x4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = memref.subview %arg0[0, %arg1, 0, %arg2] [1, %arg3, 1, %arg4] [1, 1, 1, 1]
      : memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>> to
        memref<?x?xf32, strided<[?, ?], offset: ?>>
  %1 = vector.transfer_read %0[%arg5, %arg6], %cst, %mask {
         permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]}
      : memref<?x?xf32, strided<[?, ?], offset: ?>>, vector<3x4xf32>
  return %1 : vector<3x4xf32>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d1)>
//       CHECK: func @fold_masked_vector_transfer_read_with_rank_reducing_subview
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG5:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG6:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[MASK:[a-zA-Z0-9]+]]: vector<4x3xi1>
//   CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:    %[[PAD:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:    %[[IDX0:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]], %[[ARG5]]]
//   CHECK-DAG:    %[[IDX1:.+]] = affine.apply #[[MAP0]]()[%[[ARG2]], %[[ARG6]]]
//       CHECK:    vector.transfer_read %[[ARG0]][%[[C0]], %[[IDX0]], %[[C0]], %[[IDX1]]], %[[PAD]], %[[MASK]] {{.*}} permutation_map = #[[MAP1]]} : memref<?x?x?x?xf32

// -----

func.func @fold_masked_vector_transfer_write_with_subview(
    %arg0 : memref<?x?xf32, strided<[?, ?], offset: ?>>,
    %arg1 : vector<4xf32>, %arg2: index, %arg3 : index, %arg4 : index,
    %arg5: index, %arg6 : index, %arg7 : index, %mask : vector<4xi1>) {
  %cst = arith.constant 0.0 : f32
  %0 = memref.subview %arg0[%arg2, %arg3] [%arg4, %arg5] [1, 1]
      : memref<?x?xf32, strided<[?, ?], offset: ?>> to
        memref<?x?xf32, strided<[?, ?], offset: ?>>
  vector.transfer_write %arg1, %0[%arg6, %arg7], %mask {in_bounds = [true]}
      : vector<4xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
  return
}
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func @fold_masked_vector_transfer_write_with_subview
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?xf32, strided<[?, ?], offset: ?>>
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: vector<4xf32>
//  CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG5:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG6:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG7:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[MASK:[a-zA-Z0-9]+]]: vector<4xi1>
//   CHECK-DAG:    %[[IDX0:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG6]]]
//   CHECK-DAG:    %[[IDX1:.+]] = affine.apply #[[MAP1]]()[%[[ARG3]], %[[ARG7]]]
//   CHECK-DAG:    vector.transfer_write %[[ARG1]], %[[ARG0]][%[[IDX0]], %[[IDX1]]], %[[MASK]] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32

// -----

func.func @fold_masked_vector_transfer_write_with_rank_reducing_subview(
    %arg0 : memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>,
    %arg1 : vector<3x4xf32>, %arg2: index, %arg3 : index, %arg4 : index,
    %arg5: index, %arg6 : index, %arg7 : index, %mask : vector<4x3xi1>) {
  %cst = arith.constant 0.0 : f32
  %0 = memref.subview %arg0[0, %arg2, 0, %arg3] [1, %arg4, 1, %arg5] [1, 1, 1, 1]
      : memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>> to
        memref<?x?xf32, strided<[?, ?], offset: ?>>
  vector.transfer_write %arg1, %0[%arg6, %arg7], %mask {
        permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]}
      : vector<3x4xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
  return
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d1)>
//       CHECK: func @fold_masked_vector_transfer_write_with_rank_reducing_subview
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: vector<3x4xf32>
//  CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG5:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG6:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG7:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[MASK:[a-zA-Z0-9]+]]: vector<4x3xi1>
//   CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:    %[[IDX0:.+]] = affine.apply #[[MAP0]]()[%[[ARG2]], %[[ARG6]]]
//   CHECK-DAG:    %[[IDX1:.+]] = affine.apply #[[MAP0]]()[%[[ARG3]], %[[ARG7]]]
//   CHECK-DAG:    vector.transfer_write %[[ARG1]], %[[ARG0]][%[[C0]], %[[IDX0]], %[[C0]], %[[IDX1]]], %[[ARG8]] {in_bounds = [true, true], permutation_map = #[[MAP1]]} : vector<3x4xf32>, memref<?x?x?x?xf32

// -----

//  Test with affine.load/store ops. We only do a basic test here since the
//  logic is identical to that with memref.load/store ops. The same affine.apply
//  ops would be generated.

// CHECK-LABEL: func @fold_static_stride_subview_with_affine_load_store
func.func @fold_static_stride_subview_with_affine_load_store(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) -> f32 {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] : memref<12x32xf32> to memref<4x4xf32, strided<[64, 3], offset: ?>>
  %1 = affine.load %0[%arg3, %arg4] : memref<4x4xf32, strided<[64, 3], offset: ?>>
  // CHECK-NEXT: affine.apply
  // CHECK-NEXT: affine.apply
  // CHECK-NEXT: affine.load
  affine.store %1, %0[%arg3, %arg4] : memref<4x4xf32, strided<[64, 3], offset: ?>>
  // CHECK-NEXT: affine.apply
  // CHECK-NEXT: affine.apply
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: return
  return %1 : f32
}

// -----

// CHECK-DAG: #[[$MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 6 + s1)>
// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<12x32xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> f32 {
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1], [2]] output_shape [2, 6, 32] : memref<12x32xf32> into memref<2x6x32xf32>
  %1 = affine.load %0[%arg1, %arg2, %arg3] : memref<2x6x32xf32>
  return %1 : f32
}
// CHECK: %[[INDEX:.*]] = affine.apply #[[$MAP]]()[%[[ARG1]], %[[ARG2]]]
// CHECK-NEXT: %[[RESULT:.*]] = affine.load %[[ARG0]][%[[INDEX]], %[[ARG3]]] : memref<12x32xf32>
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0] -> (s0 floordiv 6)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 mod 6)>
// CHECK-LABEL: @fold_static_stride_subview_with_affine_load_store_collapse_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<2x6x32xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @fold_static_stride_subview_with_affine_load_store_collapse_shape(%arg0 : memref<2x6x32xf32>, %arg1 : index, %arg2 : index) -> f32 {
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<2x6x32xf32> into memref<12x32xf32>
  %1 = affine.load %0[%arg1, %arg2] : memref<12x32xf32>
  return %1 : f32
}
// CHECK-NEXT: %[[MODIFIED_INDEX0:.*]] = affine.apply #[[$MAP0]]()[%[[ARG1]]]
// CHECK-NEXT: %[[MODIFIED_INDEX1:.*]] = affine.apply #[[$MAP1]]()[%[[ARG1]]]
// CHECK-NEXT: %[[RESULT:.*]] = affine.load %[[ARG0]][%[[MODIFIED_INDEX0]], %[[MODIFIED_INDEX1]], %[[ARG2]]] : memref<2x6x32xf32>
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0] -> (s0 floordiv 6)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 mod 6)>
// CHECK-LABEL: @fold_dynamic_size_collapse_shape_with_affine_load
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x6x32xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @fold_dynamic_size_collapse_shape_with_affine_load(%arg0 : memref<?x6x32xf32>, %arg1 : index, %arg2 : index) -> f32 {
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<?x6x32xf32> into memref<?x32xf32>
  %1 = affine.load %0[%arg1, %arg2] : memref<?x32xf32>
  return %1 : f32
}
// CHECK-NEXT: %[[MODIFIED_INDEX0:.*]] = affine.apply #[[$MAP0]]()[%[[ARG1]]]
// CHECK-NEXT: %[[MODIFIED_INDEX1:.*]] = affine.apply #[[$MAP1]]()[%[[ARG1]]]
// CHECK-NEXT: %[[RESULT:.*]] = affine.load %[[ARG0]][%[[MODIFIED_INDEX0]], %[[MODIFIED_INDEX1]], %[[ARG2]]] : memref<?x6x32xf32>
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-DAG: #[[$MAP:.*]] = affine_map<()[s0, s1, s2] -> (s0 * 6 + s1 * 3 + s2)>
// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape_3d
// CHECK-SAME: (%[[ARG0:.*]]: memref<12x32xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index) -> f32 {
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape_3d(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4: index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1, 2], [3]] output_shape [2, 2, 3, 32] : memref<12x32xf32> into memref<2x2x3x32xf32>
  %1 = affine.load %0[%arg1, %arg2, %arg3, %arg4] : memref<2x2x3x32xf32>
  return %1 : f32
}
// CHECK: %[[INDEX:.*]] = affine.apply #[[$MAP]]()[%[[ARG1]], %[[ARG2]], %[[ARG3]]]
// CHECK-NEXT: %[[RESULT:.*]] = affine.load %[[ARG0]][%[[INDEX]], %[[ARG4]]] : memref<12x32xf32>
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-LABEL: fold_dynamic_subview_with_memref_load_expand_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<16x?xf32, strided<[16, 1]>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> f32
func.func @fold_dynamic_subview_with_memref_load_expand_shape(%arg0 : memref<16x?xf32, strided<[16, 1]>>, %arg1 : index, %arg2 : index, %sz0: index) -> f32 {
  %c0 = arith.constant 0 : index
  %expand_shape = memref.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [1, 16, %sz0, 1] : memref<16x?xf32, strided<[16, 1]>> into memref<1x16x?x1xf32, strided<[256, 16, 1, 1]>>
  %0 = memref.load %expand_shape[%c0, %arg1, %arg2, %c0] {nontemporal = true} : memref<1x16x?x1xf32, strided<[256, 16, 1, 1]>>
  return %0 : f32
}
// CHECK-NEXT: %[[VAL1:.*]] = memref.load %[[ARG0]][%[[ARG1]], %[[ARG2]]] {nontemporal = true} : memref<16x?xf32, strided<[16, 1]>>
// CHECK-NEXT: return %[[VAL1]] : f32

// -----

// CHECK-LABEL: fold_dynamic_subview_with_memref_store_expand_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<16x?xf32, strided<[16, 1]>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @fold_dynamic_subview_with_memref_store_expand_shape(%arg0 : memref<16x?xf32, strided<[16, 1]>>, %arg1 : index, %arg2 : index, %sz0 : index) {
  %c0 = arith.constant 0 : index
  %c1f32 = arith.constant 1.0 : f32
  %expand_shape = memref.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [1, 16, %sz0, 1] : memref<16x?xf32, strided<[16, 1]>> into memref<1x16x?x1xf32, strided<[256, 16, 1, 1]>>
  memref.store %c1f32, %expand_shape[%c0, %arg1, %arg2, %c0] {nontemporal = true} : memref<1x16x?x1xf32, strided<[256, 16, 1, 1]>>
  return
}
// CHECK: %[[C1F32:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT: memref.store %[[C1F32]], %[[ARG0]][%[[ARG1]], %[[ARG2]]] {nontemporal = true} : memref<16x?xf32, strided<[16, 1]>>
// CHECK-NEXT: return

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 * 3)>
// CHECK-LABEL: fold_memref_alias_expand_shape_subview_load_store_dynamic_dim
// CHECK-SAME: (%[[ARG0:.*]]: memref<2048x16xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index)
func.func @fold_memref_alias_expand_shape_subview_load_store_dynamic_dim(%alloc: memref<2048x16xf32>, %c10: index, %c5: index, %c0: index, %sz0: index) {
  %subview = memref.subview %alloc[%c5, 0] [%c10, 16] [1, 1] : memref<2048x16xf32> to memref<?x16xf32, strided<[16, 1], offset: ?>>
  %expand_shape = memref.expand_shape %subview [[0], [1, 2, 3]] output_shape [%sz0, 1, 8, 2] : memref<?x16xf32, strided<[16, 1], offset: ?>> into memref<?x1x8x2xf32, strided<[16, 16, 2, 1], offset: ?>>
  %dim = memref.dim %expand_shape, %c0 : memref<?x1x8x2xf32, strided<[16, 16, 2, 1], offset: ?>>

  affine.for %arg6 = 0 to %dim step 64 {
    affine.for %arg7 = 0 to 16 step 16 {
      %dummy_load = affine.load %expand_shape[%arg6, 0, %arg7, %arg7] : memref<?x1x8x2xf32, strided<[16, 16, 2, 1], offset: ?>>
      affine.store %dummy_load, %subview[%arg6, %arg7] : memref<?x16xf32, strided<[16, 1], offset: ?>>
    }
  }
  return
}
// CHECK-NEXT:   memref.subview
// CHECK-NEXT:   %[[EXPAND_SHAPE:.*]] = memref.expand_shape
// CHECK-NEXT:   %[[DIM:.*]] = memref.dim %[[EXPAND_SHAPE]], %[[ARG3]] : memref<?x1x8x2xf32, strided<[16, 16, 2, 1], offset: ?>>
// CHECK-NEXT:   affine.for %[[ARG4:.*]] = 0 to %[[DIM]] step 64 {
// CHECK-NEXT:   affine.for %[[ARG5:.*]] = 0 to 16 step 16 {
// CHECK-NEXT:   %[[VAL0:.*]] = affine.apply #[[$MAP0]]()[%[[ARG2]], %[[ARG4]]]
// CHECK-NEXT:   %[[VAL1:.*]] = affine.apply #[[$MAP1]]()[%[[ARG5]]]
// CHECK-NEXT:   %[[VAL2:.*]] = affine.load %[[ARG0]][%[[VAL0]], %[[VAL1]]] : memref<2048x16xf32>
// CHECK-NEXT:   %[[VAL3:.*]] = affine.apply #[[$MAP0]]()[%[[ARG2]], %[[ARG4]]]
// CHECK-NEXT:   affine.store %[[VAL2]], %[[ARG0]][%[[VAL3]], %[[ARG5]]] : memref<2048x16xf32>

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0, s1] -> (s0 * 1024 + s1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<1024x1024xf32>, %[[ARG1:.*]]: memref<1xf32>, %[[ARG2:.*]]: index)
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape(%arg0: memref<1024x1024xf32>, %arg1: memref<1xf32>, %arg2: index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [1, 1024, 1024, 1] : memref<1024x1024xf32> into memref<1x1024x1024x1xf32>
  affine.for %arg3 = 0 to 1 {
    affine.for %arg4 = 0 to 1024 {
      affine.for %arg5 = 0 to 1020 {
        affine.for %arg6 = 0 to 1 {
          %1 = affine.load %0[%arg3, %arg4, %arg5, %arg6] : memref<1x1024x1024x1xf32>
          affine.store %1, %arg1[%arg2] : memref<1xf32>
        }
      }
    }
  }
  %2 = affine.load %arg1[%arg2] : memref<1xf32>
  return %2 : f32
}
// CHECK-NEXT: affine.for %[[ARG3:.*]] = 0 to 1 {
// CHECK-NEXT:  affine.for %[[ARG4:.*]] = 0 to 1024 {
// CHECK-NEXT:   affine.for %[[ARG5:.*]] = 0 to 1020 {
// CHECK-NEXT:    affine.for %[[ARG6:.*]] = 0 to 1 {
// CHECK-NEXT:     %[[IDX1:.*]] = affine.apply #[[$MAP0]]()[%[[ARG3]], %[[ARG4]]]
// CHECK-NEXT:     %[[IDX2:.*]] = affine.apply #[[$MAP1]]()[%[[ARG5]], %[[ARG6]]]
// CHECK-NEXT:     affine.load %[[ARG0]][%[[IDX1]], %[[IDX2]]] : memref<1024x1024xf32>

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0 * 1024)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape_when_access_index_is_an_expression
// CHECK-SAME: (%[[ARG0:.*]]: memref<1024x1024xf32>, %[[ARG1:.*]]: memref<1xf32>, %[[ARG2:.*]]: index)
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape_when_access_index_is_an_expression(%arg0: memref<1024x1024xf32>, %arg1: memref<1xf32>, %arg2: index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [1, 1024, 1024, 1] : memref<1024x1024xf32> into memref<1x1024x1024x1xf32>
  affine.for %arg3 = 0 to 1 {
    affine.for %arg4 = 0 to 1024 {
      affine.for %arg5 = 0 to 1020 {
        affine.for %arg6 = 0 to 1 {
          %1 = affine.load %0[%arg3, %arg4 + %arg3, %arg5, %arg6] : memref<1x1024x1024x1xf32>
          affine.store %1, %arg1[%arg2] : memref<1xf32>
        }
      }
    }
  }
  %2 = affine.load %arg1[%arg2] : memref<1xf32>
  return %2 : f32
}
// CHECK-NEXT: affine.for %[[ARG3:.*]] = 0 to 1 {
// CHECK-NEXT:  affine.for %[[ARG4:.*]] = 0 to 1024 {
// CHECK-NEXT:   affine.for %[[ARG5:.*]] = 0 to 1020 {
// CHECK-NEXT:    affine.for %[[ARG6:.*]] = 0 to 1 {
// CHECK-NEXT:      %[[TMP1:.*]] = affine.apply #[[$MAP0]](%[[ARG3]], %[[ARG4]])[%[[ARG3]]]
// CHECK-NEXT:      %[[TMP3:.*]] = affine.apply #[[$MAP1]]()[%[[ARG5]], %[[ARG6]]]
// CHECK-NEXT:      affine.load %[[ARG0]][%[[TMP1]], %[[TMP3]]] : memref<1024x1024xf32>

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0] -> (s0 * 1024)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape_with_constant_access_index
// CHECK-SAME: (%[[ARG0:.*]]: memref<1024x1024xf32>, %[[ARG1:.*]]: memref<1xf32>, %[[ARG2:.*]]: index)
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape_with_constant_access_index(%arg0: memref<1024x1024xf32>, %arg1: memref<1xf32>, %arg2: index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [1, 1024, 1024, 1] : memref<1024x1024xf32> into memref<1x1024x1024x1xf32>
  %cst = arith.constant 0 : index
  affine.for %arg3 = 0 to 1 {
    affine.for %arg4 = 0 to 1024 {
      affine.for %arg5 = 0 to 1020 {
        affine.for %arg6 = 0 to 1 {
          %1 = memref.load %0[%arg3, %cst, %arg5, %arg6] : memref<1x1024x1024x1xf32>
          memref.store %1, %arg1[%arg2] : memref<1xf32>
        }
      }
    }
  }
  %2 = memref.load %arg1[%arg2] : memref<1xf32>
  return %2 : f32
}
// CHECK-NEXT:   affine.for %[[ARG3:.*]] = 0 to 1 {
// CHECK-NEXT:   affine.for %[[ARG4:.*]] = 0 to 1024 {
// CHECK-NEXT:    affine.for %[[ARG5:.*]] = 0 to 1020 {
// CHECK-NEXT:     affine.for %[[ARG6:.*]] = 0 to 1 {
// CHECK-NEXT:      %[[TMP1:.*]] = affine.apply #[[$MAP0]]()[%[[ARG3]]]
// CHECK-NEXT:      %[[TMP2:.*]] = affine.apply #[[$MAP1]]()[%[[ARG5]], %[[ARG6]]]
// CHECK-NEXT:      memref.load %[[ARG0]][%[[TMP1]], %[[TMP2]]] : memref<1024x1024xf32>

// -----

// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_collapse_shape_with_0d_result
// CHECK-SAME: (%[[ARG0:.*]]: memref<1xf32>, %[[ARG1:.*]]: memref<1xf32>)
func.func @fold_static_stride_subview_with_affine_load_store_collapse_shape_with_0d_result(%arg0: memref<1xf32>, %arg1: memref<1xf32>) -> memref<1xf32> {
  %0 = memref.collapse_shape %arg0 [] : memref<1xf32> into memref<f32>
  affine.for %arg2 = 0 to 3 {
    %1 = affine.load %0[] : memref<f32>
    affine.store %1, %arg1[0] : memref<1xf32>
  }
  return %arg1 : memref<1xf32>
}
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT: affine.for %{{.*}} = 0 to 3 {
// CHECK-NEXT:   affine.load %[[ARG0]][%[[ZERO]]] : memref<1xf32>

// -----

//       CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-LABEL: func @subview_of_subview(
//  CHECK-SAME:     %[[m:.*]]: memref<1x1024xf32, 3>, %[[pos:.*]]: index
//       CHECK:   %[[add:.*]] = affine.apply #[[$map]]()[%arg1]
//       CHECK:   memref.subview %arg0[4, %[[add]]] [1, 1] [1, 1] : memref<1x1024xf32, 3> to memref<f32, strided<[], offset: ?>, 3>
func.func @subview_of_subview(%m: memref<1x1024xf32, 3>, %pos: index)
    -> memref<f32, strided<[], offset: ?>, 3>
{
  %0 = memref.subview %m[3, %pos] [1, 2] [1, 1]
      : memref<1x1024xf32, 3>
        to memref<1x2xf32, strided<[1024, 1], offset: ?>, 3>
  %1 = memref.subview %0[1, 2] [1, 1] [1, 1]
      : memref<1x2xf32, strided<[1024, 1], offset: ?>, 3>
        to memref<f32, strided<[], offset: ?>, 3>
  return %1 : memref<f32, strided<[], offset: ?>, 3>
}

// -----

// CHECK-LABEL: func @subview_of_subview_rank_reducing(
//  CHECK-SAME:     %[[m:.*]]: memref<?x?x?xf32>
//       CHECK:   memref.subview %arg0[3, 7, 8] [1, 1, 1] [1, 1, 1] : memref<?x?x?xf32> to memref<f32, strided<[], offset: ?>>
func.func @subview_of_subview_rank_reducing(%m: memref<?x?x?xf32>,
                                            %sz: index, %pos: index)
    -> memref<f32, strided<[], offset: ?>>
{
  %0 = memref.subview %m[3, 1, 8] [1, %sz, 1] [1, 1, 1]
      : memref<?x?x?xf32>
        to memref<?xf32, strided<[?], offset: ?>>
  %1 = memref.subview %0[6] [1] [1]
      : memref<?xf32, strided<[?], offset: ?>>
        to memref<f32, strided<[], offset: ?>>
  return %1 : memref<f32, strided<[], offset: ?>>
}

// -----

// CHECK-LABEL: func @fold_load_keep_nontemporal(
//      CHECK:   memref.load %{{.+}}[%{{.+}}, %{{.+}}] {nontemporal = true}
func.func @fold_load_keep_nontemporal(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) -> f32 {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] : memref<12x32xf32> to memref<4x4xf32, strided<[64, 3], offset: ?>>
  %1 = memref.load %0[%arg3, %arg4] {nontemporal = true }: memref<4x4xf32, strided<[64, 3], offset: ?>>
  return %1 : f32
}

// -----

// CHECK-LABEL: func @fold_store_keep_nontemporal(
//      CHECK:   memref.store %{{.+}}, %{{.+}}[%{{.+}}, %{{.+}}]  {nontemporal = true} : memref<12x32xf32> 
func.func @fold_store_keep_nontemporal(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : f32) {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] :
    memref<12x32xf32> to memref<4x4xf32, strided<[64, 3], offset: ?>>
  memref.store %arg5, %0[%arg3, %arg4] {nontemporal=true}: memref<4x4xf32, strided<[64, 3], offset: ?>>
  return
}

// -----

func.func @fold_gpu_subgroup_mma_load_matrix_1d(%src: memref<?xvector<4xf32>>, %offset: index, %i: index) -> !gpu.mma_matrix<16x16xf16, "COp"> {
  %subview = memref.subview %src[%offset] [81920] [1] : memref<?xvector<4xf32>> to memref<81920xvector<4xf32>, strided<[1], offset: ?>>
  %matrix = gpu.subgroup_mma_load_matrix %subview[%i] {leadDimension = 160 : index} : memref<81920xvector<4xf32>, strided<[1], offset: ?>> -> !gpu.mma_matrix<16x16xf16, "COp">
  return %matrix: !gpu.mma_matrix<16x16xf16, "COp">
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//      CHECK: func.func @fold_gpu_subgroup_mma_load_matrix_1d
// CHECK-SAME: (%[[SRC:.+]]: memref<?xvector<4xf32>>, %[[OFFSET:.+]]: index, %[[I:.+]]: index)
//      CHECK:   %[[APPLY:.+]] = affine.apply #[[MAP]]()[%[[OFFSET]], %[[I]]]
//      CHECK:   %[[LOAD:.+]] = gpu.subgroup_mma_load_matrix %[[SRC]][%[[APPLY]]] {leadDimension = 160 : index} : memref<?xvector<4xf32>> -> !gpu.mma_matrix<16x16xf16, "COp">
//      CHECK:   return %[[LOAD]]

// -----

func.func @fold_gpu_subgroup_mma_store_matrix_1d(%dst: memref<?xvector<4xf32>>, %offset: index, %i: index, %matrix: !gpu.mma_matrix<16x16xf16, "COp">) {
  %subview = memref.subview %dst[%offset] [81920] [1] : memref<?xvector<4xf32>> to memref<81920xvector<4xf32>, strided<[1], offset: ?>>
  gpu.subgroup_mma_store_matrix %matrix, %subview[%i] {leadDimension = 160 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<81920xvector<4xf32>, strided<[1], offset: ?>>
  return
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//      CHECK: func.func @fold_gpu_subgroup_mma_store_matrix_1d
// CHECK-SAME: (%[[DST:.+]]: memref<?xvector<4xf32>>, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[VAL:.+]]: !gpu.mma_matrix<16x16xf16, "COp">)
//      CHECK:   %[[APPLY:.+]] = affine.apply #[[MAP]]()[%[[OFFSET]], %[[I0]]]
//      CHECK:   gpu.subgroup_mma_store_matrix %[[VAL]], %[[DST]][%[[APPLY]]] {leadDimension = 160 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<?xvector<4xf32>>

// -----

// CHECK-LABEL: func.func @fold_gpu_subgroup_mma_load_matrix_2d
//  CHECK-SAME: %[[SRC:.+]]: memref<128x128xf32>
func.func @fold_gpu_subgroup_mma_load_matrix_2d(%arg0 : memref<128x128xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) -> !gpu.mma_matrix<16x16xf16, "COp"> {
  %subview = memref.subview %arg0[%arg1, %arg2][64, 32][2, 1] : memref<128x128xf32> to memref<64x32xf32, strided<[256, 1], offset: ?>>
  // CHECK: gpu.subgroup_mma_load_matrix %[[SRC]][{{.+}}] {leadDimension = 32 : index} : memref<128x128xf32> -> !gpu.mma_matrix<16x16xf16, "COp">
  %matrix = gpu.subgroup_mma_load_matrix %subview[%arg3, %arg4] {leadDimension = 32 : index} : memref<64x32xf32, strided<[256, 1], offset: ?>> -> !gpu.mma_matrix<16x16xf16, "COp">
  return %matrix : !gpu.mma_matrix<16x16xf16, "COp">
}

// -----

// CHECK-LABEL: func.func @fold_gpu_subgroup_mma_load_matrix_2d
//  CHECK-SAME: %[[DST:.+]]: memref<128x128xf32>
func.func @fold_gpu_subgroup_mma_load_matrix_2d(%arg0 : memref<128x128xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %matrix: !gpu.mma_matrix<16x16xf16, "COp">) {
  %subview = memref.subview %arg0[%arg1, %arg2][64, 32][2, 1] : memref<128x128xf32> to memref<64x32xf32, strided<[256, 1], offset: ?>>
  // CHECK: gpu.subgroup_mma_store_matrix %{{.+}}, %[[DST]][{{.+}}] {leadDimension = 32 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<128x128xf32>
  gpu.subgroup_mma_store_matrix %matrix, %subview[%arg3, %arg4] {leadDimension = 32 : index} :  !gpu.mma_matrix<16x16xf16, "COp">, memref<64x32xf32, strided<[256, 1], offset: ?>>
  return
}

// -----


func.func @fold_nvgpu_device_async_copy_zero_sub_idx(%gmem_memref_3d : memref<2x128x768xf16>, %idx_1 : index, %idx_2 : index, %idx_3 : index) {

  %c0 = arith.constant 0 : index
  %smem_memref_4d = memref.alloc() : memref<5x1x64x64xf16, #gpu.address_space<workgroup>>
  %gmem_memref_subview_2d = memref.subview %gmem_memref_3d[%idx_1, %idx_2, %idx_3] [1, 1, 8] [1, 1, 1] : memref<2x128x768xf16> to memref<1x8xf16, strided<[98304, 1], offset: ?>>
  %async_token = nvgpu.device_async_copy %gmem_memref_subview_2d[%c0, %c0], %smem_memref_4d[%c0, %c0, %c0, %c0], 8 {bypassL1} : memref<1x8xf16, strided<[98304, 1], offset: ?>> to memref<5x1x64x64xf16, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: func.func @fold_nvgpu_device_async_copy_zero_sub_idx
//  CHECK-SAME: (%[[GMEM_MEMREF_3d:.+]]: memref<2x128x768xf16>, %[[IDX_1:.+]]: index, %[[IDX_2:.+]]: index, %[[IDX_3:.+]]: index)
//   CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[SMEM_MEMREF_4d:.+]] = memref.alloc() : memref<5x1x64x64xf16, #gpu.address_space<workgroup>>
//       CHECK: nvgpu.device_async_copy %[[GMEM_MEMREF_3d]][%[[IDX_1]], %[[IDX_2]], %[[IDX_3]]], %[[SMEM_MEMREF_4d]][%[[c0]], %[[c0]], %[[c0]], %[[c0]]], 8 {bypassL1} : memref<2x128x768xf16> to memref<5x1x64x64xf16, #gpu.address_space<workgroup>>

// -----


func.func @fold_src_nvgpu_device_async_copy(%gmem_memref_3d : memref<2x128x768xf16>, %src_idx_0 : index, %src_idx_1 : index, %src_idx_2 : index, %src_sub_idx_0 : index, %src_sub_idx_1 : index) {
  %c0 = arith.constant 0 : index
  %smem_memref_4d = memref.alloc() : memref<5x1x64x64xf16, #gpu.address_space<workgroup>>
  %gmem_memref_subview_2d = memref.subview %gmem_memref_3d[%src_idx_0, %src_idx_1, %src_idx_2] [1, 1, 8] [1, 1, 1] : memref<2x128x768xf16> to memref<1x8xf16, strided<[98304, 1], offset: ?>>
  %async_token = nvgpu.device_async_copy %gmem_memref_subview_2d[%src_sub_idx_0, %src_sub_idx_1], %smem_memref_4d[%c0, %c0, %c0, %c0], 8 {bypassL1} : memref<1x8xf16, strided<[98304, 1], offset: ?>> to memref<5x1x64x64xf16, #gpu.address_space<workgroup>>
  return
}

//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func.func @fold_src_nvgpu_device_async_copy
//  CHECK-SAME: (%[[GMEM_MEMREF_3d:.+]]: memref<2x128x768xf16>, %[[SRC_IDX_0:.+]]: index, %[[SRC_IDX_1:.+]]: index, %[[SRC_IDX_2:.+]]: index, %[[SRC_SUB_IDX_0:.+]]: index, %[[SRC_SUB_IDX_1:.+]]: index)
//   CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[RESOLVED_SRC_IDX_0:.+]] = affine.apply #[[MAP]]()[%[[SRC_IDX_0]], %[[SRC_SUB_IDX_0]]]
//   CHECK-DAG: %[[RESOLVED_SRC_IDX_1:.+]] = affine.apply #[[MAP]]()[%[[SRC_IDX_2]], %[[SRC_SUB_IDX_1]]]
//   CHECK-DAG: nvgpu.device_async_copy %[[GMEM_MEMREF_3d]][%[[RESOLVED_SRC_IDX_0]], %[[SRC_IDX_1]], %[[RESOLVED_SRC_IDX_1]]], %[[SMEM_MEMREF_4d]][%[[c0]], %[[c0]], %[[c0]], %[[c0]]], 8 {bypassL1} : memref<2x128x768xf16> to memref<5x1x64x64xf16, #gpu.address_space<workgroup>>

// -----


func.func @fold_src_fold_dest_nvgpu_device_async_copy(%gmem_memref_3d : memref<2x128x768xf16>, %src_idx_0 : index, %src_idx_1 : index, %src_idx_2 : index, %src_sub_idx_0 : index, %src_sub_idx_1 : index, %dest_idx_0 : index, %dest_idx_1 : index, %dest_idx_2 : index, %dest_idx_3 : index, %dest_sub_idx_0 : index, %dest_sub_idx_1 : index) {
  %c0 = arith.constant 0 : index
  %smem_memref_4d = memref.alloc() : memref<5x1x64x64xf16, #gpu.address_space<workgroup>>
  %gmem_memref_subview_2d = memref.subview %gmem_memref_3d[%src_idx_0, %src_idx_1, %src_idx_2] [1, 1, 8] [1, 1, 1] : memref<2x128x768xf16> to memref<1x8xf16, strided<[98304, 1], offset: ?>>
  %smem_memref_2d = memref.subview %smem_memref_4d[%dest_idx_0, %dest_idx_1, %dest_idx_2, %dest_idx_3] [1, 1, 1, 8] [1, 1, 1, 1] : memref<5x1x64x64xf16, #gpu.address_space<workgroup>> to memref<1x8xf16, strided<[4096, 1], offset: ?>, #gpu.address_space<workgroup>>
  %async_token = nvgpu.device_async_copy %gmem_memref_subview_2d[%src_sub_idx_0, %src_sub_idx_1], %smem_memref_2d[%dest_sub_idx_0, %dest_sub_idx_1], 8 {bypassL1} : memref<1x8xf16, strided<[98304, 1], offset: ?>> to memref<1x8xf16, strided<[4096, 1], offset: ?>, #gpu.address_space<workgroup>>
  return
}

//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func.func @fold_src_fold_dest_nvgpu_device_async_copy
//  CHECK-SAME: (%[[GMEM_MEMREF_3d:.+]]: memref<2x128x768xf16>, %[[SRC_IDX_0:.+]]: index, %[[SRC_IDX_1:.+]]: index, %[[SRC_IDX_2:.+]]: index, %[[SRC_SUB_IDX_0:.+]]: index, %[[SRC_SUB_IDX_1:.+]]: index, %[[DEST_IDX_0:.+]]: index, %[[DEST_IDX_1:.+]]: index, %[[DEST_IDX_2:.+]]: index, %[[DEST_IDX_3:.+]]: index, %[[DEST_SUB_IDX_0:.+]]: index, %[[DEST_SUB_IDX_1:.+]]: index)
//   CHECK-DAG: %[[RESOLVED_SRC_IDX_0:.+]] = affine.apply #[[MAP]]()[%[[SRC_IDX_0]], %[[SRC_SUB_IDX_0]]]
//   CHECK-DAG: %[[RESOLVED_SRC_IDX_1:.+]] = affine.apply #[[MAP]]()[%[[SRC_IDX_2]], %[[SRC_SUB_IDX_1]]]
//   CHECK-DAG: %[[RESOLVED_DST_IDX_1:.+]] = affine.apply #[[MAP]]()[%[[DEST_IDX_1]], %[[DEST_SUB_IDX_0]]]
//   CHECK-DAG: %[[RESOLVED_DST_IDX_3:.+]] = affine.apply #[[MAP]]()[%[[DEST_IDX_3]], %[[DEST_SUB_IDX_1]]]
//   CHECK-DAG: nvgpu.device_async_copy %[[GMEM_MEMREF_3d]][%[[RESOLVED_SRC_IDX_0]], %[[SRC_IDX_1]], %[[RESOLVED_SRC_IDX_1]]], %[[SMEM_MEMREF_4d]][%[[DEST_IDX_0]], %[[RESOLVED_DST_IDX_1]], %[[DEST_IDX_2]], %[[RESOLVED_DST_IDX_3]]], 8 {bypassL1} : memref<2x128x768xf16> to memref<5x1x64x64xf16, #gpu.address_space<workgroup>>

// -----

#map = affine_map<()[s0] -> (-s0 + 4)>
#map1 = affine_map<()[s0] -> (-s0 + 32)>

func.func @test_ldmatrix(%arg0: memref<4x32x32xf16, 3>, %arg1: index, %arg2: index, %arg3: index) -> vector<4x2xf16> {
  %c0 = arith.constant 0 : index
  %0 = affine.apply #map()[%arg1]
  %1 = affine.apply #map1()[%arg2]
  %2 = affine.apply #map1()[%arg3]
  %subview = memref.subview %arg0[%arg1, %arg2, %arg3] [%0, %1, %2] [1, 1, 1] : memref<4x32x32xf16, 3> to memref<?x?x?xf16, strided<[1024, 32, 1], offset: ?>, 3>
  %3 = nvgpu.ldmatrix %subview[%c0, %c0, %c0] {numTiles = 4 : i32, transpose = false} : memref<?x?x?xf16, strided<[1024, 32, 1], offset: ?>, 3> -> vector<4x2xf16>
  return %3 : vector<4x2xf16>
}

//      CHECK: func @test_ldmatrix
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<4x32x32xf16, 3>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
//      CHECK:   nvgpu.ldmatrix %[[ARG0]][%[[ARG1]], %[[ARG2]], %[[ARG3]]] {numTiles = 4 : i32, transpose = false} : memref<4x32x32xf16, 3> -> vector<4x2xf16>

// -----

func.func @fold_vector_load_subview(
  %arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index) -> vector<12x32xf32> {
  %0 = memref.subview %arg0[%arg1, %arg2][1, 1][1, 1] : memref<12x32xf32> to memref<f32, strided<[], offset: ?>>
  %1 = vector.load %0[] : memref<f32, strided<[], offset: ?>>, vector<12x32xf32>
  return %1 : vector<12x32xf32>
}

//      CHECK: func @fold_vector_load_subview
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
//      CHECK:   vector.load %[[ARG0]][%[[ARG1]], %[[ARG2]]] :  memref<12x32xf32>, vector<12x32xf32>

// -----

func.func @fold_vector_maskedload_subview(
  %arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3: vector<32xi1>, %arg4: vector<32xf32>) -> vector<32xf32> {
  %0 = memref.subview %arg0[%arg1, %arg2][1, 1][1, 1] : memref<12x32xf32> to memref<f32, strided<[], offset: ?>>
  %1 = vector.maskedload %0[], %arg3, %arg4 : memref<f32, strided<[], offset: ?>>, vector<32xi1>, vector<32xf32> into vector<32xf32>
  return %1 : vector<32xf32>
}

//      CHECK: func @fold_vector_maskedload_subview
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: vector<32xi1>
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: vector<32xf32>
//      CHECK:   vector.maskedload %[[ARG0]][%[[ARG1]], %[[ARG2]]], %[[ARG3]], %[[ARG4]] : memref<12x32xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>

// -----

func.func @fold_vector_store_subview(
  %arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3: vector<2x32xf32>) -> () {
  %0 = memref.subview %arg0[%arg1, %arg2][1, 1][1, 1] : memref<12x32xf32> to memref<f32, strided<[], offset: ?>>
  vector.store %arg3, %0[] : memref<f32, strided<[], offset: ?>>, vector<2x32xf32>
  return
}

//      CHECK: func @fold_vector_store_subview
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: vector<2x32xf32>
//      CHECK:   vector.store %[[ARG3]], %[[ARG0]][%[[ARG1]], %[[ARG2]]] :  memref<12x32xf32>, vector<2x32xf32>
//      CHECK:   return

// -----

func.func @fold_vector_maskedstore_subview(
  %arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3: vector<32xi1>, %arg4: vector<32xf32>) -> () {
  %0 = memref.subview %arg0[%arg1, %arg2][1, 1][1, 1] : memref<12x32xf32> to memref<f32, strided<[], offset: ?>>
  vector.maskedstore %0[], %arg3, %arg4 : memref<f32, strided<[], offset: ?>>, vector<32xi1>, vector<32xf32>
  return
}

//      CHECK: func @fold_vector_maskedstore_subview
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: vector<32xi1>
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: vector<32xf32>
//      CHECK:   vector.maskedstore %[[ARG0]][%[[ARG1]], %[[ARG2]]], %[[ARG3]], %[[ARG4]] : memref<12x32xf32>, vector<32xi1>, vector<32xf32>
//      CHECK:   return

// -----

func.func @fold_vector_load_expand_shape(
  %arg0 : memref<32xf32>, %arg1 : index) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %0 = memref.expand_shape %arg0 [[0, 1]] output_shape [4, 8] : memref<32xf32> into memref<4x8xf32>
  %1 = vector.load %0[%arg1, %c0] {nontemporal = true} : memref<4x8xf32>, vector<8xf32>
  return %1 : vector<8xf32>
}

//   CHECK-DAG: #[[$MAP:.*]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-LABEL: func @fold_vector_load_expand_shape
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<32xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//       CHECK:   %[[IDX:.*]] = affine.apply #[[$MAP]]()[%[[ARG1]]]
//       CHECK:   vector.load %[[ARG0]][%[[IDX]]] {nontemporal = true}

// -----

func.func @fold_vector_maskedload_expand_shape(
  %arg0 : memref<32xf32>, %arg1 : index, %arg3: vector<8xi1>, %arg4: vector<8xf32>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %0 = memref.expand_shape %arg0 [[0, 1]] output_shape [4, 8] : memref<32xf32> into memref<4x8xf32>
  %1 = vector.maskedload %0[%arg1, %c0], %arg3, %arg4 : memref<4x8xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
  return %1 : vector<8xf32>
}

//   CHECK-DAG: #[[$MAP:.*]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-LABEL: func @fold_vector_maskedload_expand_shape
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<32xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: vector<8xi1>
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: vector<8xf32>
//       CHECK:   %[[IDX:.*]] = affine.apply #[[$MAP]]()[%[[ARG1]]]
//       CHECK:   vector.maskedload %[[ARG0]][%[[IDX]]], %[[ARG3]], %[[ARG4]]

// -----

func.func @fold_vector_store_expand_shape(
  %arg0 : memref<32xf32>, %arg1 : index, %val : vector<8xf32>) {
  %c0 = arith.constant 0 : index
  %0 = memref.expand_shape %arg0 [[0, 1]] output_shape [4, 8] : memref<32xf32> into memref<4x8xf32>
  vector.store %val, %0[%arg1, %c0] {nontemporal = true} : memref<4x8xf32>, vector<8xf32>
  return
}

//   CHECK-DAG: #[[$MAP:.*]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-LABEL: func @fold_vector_store_expand_shape
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<32xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//       CHECK:   %[[IDX:.*]] = affine.apply #[[$MAP]]()[%[[ARG1]]]
//       CHECK:   vector.store %{{.*}}, %[[ARG0]][%[[IDX]]] {nontemporal = true}

// -----

func.func @fold_vector_maskedstore_expand_shape(
  %arg0 : memref<32xf32>, %arg1 : index, %arg3: vector<8xi1>, %arg4: vector<8xf32>) {
  %c0 = arith.constant 0 : index
  %0 = memref.expand_shape %arg0 [[0, 1]] output_shape [4, 8] : memref<32xf32> into memref<4x8xf32>
  vector.maskedstore %0[%arg1, %c0], %arg3, %arg4 : memref<4x8xf32>, vector<8xi1>, vector<8xf32>
  return
}

//   CHECK-DAG: #[[$MAP:.*]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-LABEL: func @fold_vector_maskedstore_expand_shape
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<32xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: vector<8xi1>
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: vector<8xf32>
//       CHECK:   %[[IDX:.*]] = affine.apply #[[$MAP]]()[%[[ARG1]]]
//       CHECK:   vector.maskedstore %[[ARG0]][%[[IDX]]], %[[ARG3]], %[[ARG4]]

// -----

func.func @fold_vector_load_collapse_shape(
  %arg0 : memref<4x8xf32>, %arg1 : index) -> vector<8xf32> {
  %0 = memref.collapse_shape %arg0 [[0, 1]] : memref<4x8xf32> into memref<32xf32>
  %1 = vector.load %0[%arg1] {nontemporal = true} : memref<32xf32>, vector<8xf32>
  return %1 : vector<8xf32>
}

//   CHECK-DAG: #[[$MAP:.*]]  = affine_map<()[s0] -> (s0 floordiv 8)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 mod 8)>
// CHECK-LABEL: func @fold_vector_load_collapse_shape
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<4x8xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//       CHECK:   %[[IDX:.*]] = affine.apply  #[[$MAP]]()[%[[ARG1]]]
//       CHECK:   %[[IDX1:.*]] = affine.apply #[[$MAP1]]()[%[[ARG1]]]
//       CHECK:   vector.load %[[ARG0]][%[[IDX]], %[[IDX1]]] {nontemporal = true}

// -----

func.func @fold_vector_maskedload_collapse_shape(
  %arg0 : memref<4x8xf32>, %arg1 : index, %arg3: vector<8xi1>, %arg4: vector<8xf32>) -> vector<8xf32> {
  %0 = memref.collapse_shape %arg0 [[0, 1]] : memref<4x8xf32> into memref<32xf32>
  %1 = vector.maskedload %0[%arg1], %arg3, %arg4 : memref<32xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
  return %1 : vector<8xf32>
}

//   CHECK-DAG: #[[$MAP:.*]]  = affine_map<()[s0] -> (s0 floordiv 8)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 mod 8)>
// CHECK-LABEL: func @fold_vector_maskedload_collapse_shape
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<4x8xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: vector<8xi1>
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: vector<8xf32>
//       CHECK:   %[[IDX:.*]] = affine.apply  #[[$MAP]]()[%[[ARG1]]]
//       CHECK:   %[[IDX1:.*]] = affine.apply #[[$MAP1]]()[%[[ARG1]]]
//       CHECK:   vector.maskedload %[[ARG0]][%[[IDX]], %[[IDX1]]], %[[ARG3]], %[[ARG4]]

// -----

func.func @fold_vector_store_collapse_shape(
  %arg0 : memref<4x8xf32>, %arg1 : index, %val : vector<8xf32>) {
  %0 = memref.collapse_shape %arg0 [[0, 1]] : memref<4x8xf32> into memref<32xf32>
  vector.store %val, %0[%arg1] {nontemporal = true} : memref<32xf32>, vector<8xf32>
  return
}

//   CHECK-DAG: #[[$MAP:.*]]  = affine_map<()[s0] -> (s0 floordiv 8)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 mod 8)>
// CHECK-LABEL: func @fold_vector_store_collapse_shape
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<4x8xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//       CHECK:   %[[IDX:.*]] = affine.apply  #[[$MAP]]()[%[[ARG1]]]
//       CHECK:   %[[IDX1:.*]] = affine.apply #[[$MAP1]]()[%[[ARG1]]]
//       CHECK:   vector.store %{{.*}}, %[[ARG0]][%[[IDX]], %[[IDX1]]] {nontemporal = true}

// -----

func.func @fold_vector_maskedstore_collapse_shape(
  %arg0 : memref<4x8xf32>, %arg1 : index, %arg3: vector<8xi1>, %arg4: vector<8xf32>) {
  %0 = memref.collapse_shape %arg0 [[0, 1]] : memref<4x8xf32> into memref<32xf32>
  vector.maskedstore %0[%arg1], %arg3, %arg4 : memref<32xf32>, vector<8xi1>, vector<8xf32>
  return
}

//   CHECK-DAG: #[[$MAP:.*]]  = affine_map<()[s0] -> (s0 floordiv 8)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 mod 8)>
// CHECK-LABEL: func @fold_vector_maskedstore_collapse_shape
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<4x8xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: vector<8xi1>
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: vector<8xf32>
//       CHECK:   %[[IDX:.*]] = affine.apply  #[[$MAP]]()[%[[ARG1]]]
//       CHECK:   %[[IDX1:.*]] = affine.apply #[[$MAP1]]()[%[[ARG1]]]
//       CHECK:   vector.maskedstore %[[ARG0]][%[[IDX]], %[[IDX1]]], %[[ARG3]], %[[ARG4]]
