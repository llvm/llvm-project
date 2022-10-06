// RUN: mlir-opt -fold-memref-alias-ops -split-input-file %s -o - | FileCheck %s

func.func @fold_static_stride_subview_with_load(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) -> f32 {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] : memref<12x32xf32> to memref<4x4xf32, strided<[64, 3], offset: ?>>
  %1 = memref.load %0[%arg3, %arg4] : memref<4x4xf32, strided<[64, 3], offset: ?>>
  return %1 : f32
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (d0 * 3 + s0)>
//      CHECK: func @fold_static_stride_subview_with_load
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP0]](%[[ARG3]])[%[[ARG1]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP1]](%[[ARG4]])[%[[ARG2]]]
//      CHECK:   memref.load %[[ARG0]][%[[I1]], %[[I2]]]

// -----

func.func @fold_dynamic_stride_subview_with_load(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index) -> f32 {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] :
    memref<12x32xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
  %1 = memref.load %0[%arg3, %arg4] : memref<4x4xf32, strided<[?, ?], offset: ?>>
  return %1 : f32
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
//      CHECK: func @fold_dynamic_stride_subview_with_load
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG5:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG6:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP]](%[[ARG3]])[%[[ARG5]], %[[ARG1]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP]](%[[ARG4]])[%[[ARG6]], %[[ARG2]]]
//      CHECK:   memref.load %[[ARG0]][%[[I1]], %[[I2]]]

// -----

func.func @fold_static_stride_subview_with_store(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : f32) {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] :
    memref<12x32xf32> to memref<4x4xf32, strided<[64, 3], offset: ?>>
  memref.store %arg5, %0[%arg3, %arg4] : memref<4x4xf32, strided<[64, 3], offset: ?>>
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (d0 * 3 + s0)>
//      CHECK: func @fold_static_stride_subview_with_store
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP0]](%[[ARG3]])[%[[ARG1]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP1]](%[[ARG4]])[%[[ARG2]]]
//      CHECK:   memref.store %{{.+}}, %[[ARG0]][%[[I1]], %[[I2]]]

// -----

func.func @fold_dynamic_stride_subview_with_store(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index, %arg7 : f32) {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] :
    memref<12x32xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
  memref.store %arg7, %0[%arg3, %arg4] : memref<4x4xf32, strided<[?, ?], offset: ?>>
  return
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
//      CHECK: func @fold_dynamic_stride_subview_with_store
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG5:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG6:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP]](%[[ARG3]])[%[[ARG5]], %[[ARG1]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP]](%[[ARG4]])[%[[ARG6]], %[[ARG2]]]
//      CHECK:   memref.store %{{.+}}, %[[ARG0]][%[[I1]], %[[I2]]]

// -----

func.func @fold_subview_with_transfer_read_0d(
  %arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index)
    -> vector<f32> {
  %f1 = arith.constant 1.0 : f32
  %0 = memref.subview %arg0[%arg1, %arg2][1, 1][2, %arg3] : memref<12x32xf32> to memref<f32, strided<[], offset: ?>>
  %1 = vector.transfer_read %0[], %f1 : memref<f32, strided<[], offset: ?>>, vector<f32>
  return %1 : vector<f32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
//      CHECK: func @fold_subview_with_transfer_read_0d
// CHECK-SAME:   %[[MEM:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[SZ0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[SZ1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ST1:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP0]](%[[C0]])[%[[SZ0]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP1]](%[[C0]])[%[[ST1]], %[[SZ1]]]
//      CHECK:   vector.transfer_read %[[MEM]][%[[I1]], %[[I2]]]

// -----

func.func @fold_subview_with_transfer_read(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index) -> vector<4xf32> {
  %f1 = arith.constant 1.0 : f32
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] : memref<12x32xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
  %1 = vector.transfer_read %0[%arg3, %arg4], %f1 {in_bounds = [true]} : memref<4x4xf32, strided<[?, ?], offset: ?>>, vector<4xf32>
  return %1 : vector<4xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
//      CHECK: func @fold_subview_with_transfer_read
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG5:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG6:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP]](%[[ARG3]])[%[[ARG5]], %[[ARG1]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP]](%[[ARG4]])[%[[ARG6]], %[[ARG2]]]
//      CHECK:   vector.transfer_read %[[ARG0]][%[[I1]], %[[I2]]]

// -----

func.func @fold_static_stride_subview_with_transfer_write_0d(
    %arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, 
    %v : vector<f32>) {
  %f1 = arith.constant 1.0 : f32
  %0 = memref.subview %arg0[%arg1, %arg2][1, 1][2, %arg3] : memref<12x32xf32> to memref<f32, strided<[], offset: ?>>
  vector.transfer_write %v, %0[] {in_bounds = []} : vector<f32>, memref<f32, strided<[], offset: ?>>
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
//      CHECK: func @fold_static_stride_subview_with_transfer_write_0d
// CHECK-SAME:   %[[MEM:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[SZ0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[SZ1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ST1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[V:[a-zA-Z0-9_]+]]: vector<f32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP0]](%[[C0]])[%[[SZ0]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP1]](%[[C0]])[%[[ST1]], %[[SZ1]]]
//      CHECK:   vector.transfer_write %[[V]], %[[MEM]][%[[I1]], %[[I2]]]

// -----

func.func @fold_static_stride_subview_with_transfer_write(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5: index, %arg6 : index, %arg7 : vector<4xf32>) {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] :
    memref<12x32xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
  vector.transfer_write %arg7, %0[%arg3, %arg4] {in_bounds = [true]} : vector<4xf32>, memref<4x4xf32, strided<[?, ?], offset: ?>>
  return
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
//      CHECK: func @fold_static_stride_subview_with_transfer_write
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<12x32xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG5:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG6:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP]](%[[ARG3]])[%[[ARG5]], %[[ARG1]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP]](%[[ARG4]])[%[[ARG6]], %[[ARG2]]]
//      CHECK:   vector.transfer_write %{{.+}}, %[[ARG0]][%[[I1]], %[[I2]]]

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
//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
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
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP]](%[[ARG13]])[%[[ARG7]], %[[ARG1]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP]](%[[C0]])[%[[ARG8]], %[[ARG2]]]
//  CHECK-DAG:   %[[I3:.+]] = affine.apply #[[MAP]](%[[ARG14]])[%[[ARG9]], %[[ARG3]]]
//  CHECK-DAG:   %[[I4:.+]] = affine.apply #[[MAP]](%[[ARG15]])[%[[ARG10]], %[[ARG4]]]
//  CHECK-DAG:   %[[I5:.+]] = affine.apply #[[MAP]](%[[ARG16]])[%[[ARG11]], %[[ARG5]]]
//  CHECK-DAG:   %[[I6:.+]] = affine.apply #[[MAP]](%[[C0]])[%[[ARG12]], %[[ARG6]]]
//      CHECK:   memref.load %[[ARG0]][%[[I1]], %[[I2]], %[[I3]], %[[I4]], %[[I5]], %[[I6]]]

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
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//       CHECK: func @fold_vector_transfer_read_with_rank_reduced_subview
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG5:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG6:[a-zA-Z0-9]+]]: index
//   CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:    %[[IDX0:.+]] = affine.apply #[[MAP1]](%[[ARG5]])[%[[ARG1]]]
//   CHECK-DAG:    %[[IDX1:.+]] = affine.apply #[[MAP1]](%[[ARG6]])[%[[ARG2]]]
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
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
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
//   CHECK-DAG:    %[[IDX0:.+]] = affine.apply #[[MAP1]](%[[ARG6]])[%[[ARG2]]]
//   CHECK-DAG:    %[[IDX1:.+]] = affine.apply #[[MAP1]](%[[ARG7]])[%[[ARG3]]]
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
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
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
//   CHECK-DAG:    %[[IDX0:.+]] = affine.apply #[[MAP1]](%[[ARG6]])[%[[ARG2]]]
//   CHECK-DAG:    %[[IDX1:.+]] = affine.apply #[[MAP1]](%[[ARG7]])[%[[ARG3]]]
//   CHECK-DAG:    vector.transfer_write %[[ARG1]], %[[ARG0]][%[[IDX0]], %[[IDX1]], %[[C0]]]
//  CHECK-SAME:    {in_bounds = [true], permutation_map = #[[MAP2]]} : vector<4xf32>, memref<?x?x?xf32

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

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0 * 6 + d1)>
// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<12x32xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> f32 {
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1], [2]] : memref<12x32xf32> into memref<2x6x32xf32>
  %1 = affine.load %0[%arg1, %arg2, %arg3] : memref<2x6x32xf32>
  return %1 : f32
}
// CHECK: %[[INDEX:.*]] = affine.apply #[[$MAP]](%[[ARG1]], %[[ARG2]])
// CHECK-NEXT: %[[RESULT:.*]] = affine.load %[[ARG0]][%[[INDEX]], %[[ARG3]]] : memref<12x32xf32>
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0) -> (d0 floordiv 6)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (d0 mod 6)>
// CHECK-LABEL: @fold_static_stride_subview_with_affine_load_store_collapse_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<2x6x32xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @fold_static_stride_subview_with_affine_load_store_collapse_shape(%arg0 : memref<2x6x32xf32>, %arg1 : index, %arg2 : index) -> f32 {
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<2x6x32xf32> into memref<12x32xf32>
  %1 = affine.load %0[%arg1, %arg2] : memref<12x32xf32>
  return %1 : f32
}
// CHECK-NEXT: %[[MODIFIED_INDEX0:.*]] = affine.apply #[[$MAP0]](%[[ARG1]])
// CHECK-NEXT: %[[MODIFIED_INDEX1:.*]] = affine.apply #[[$MAP1]](%[[ARG1]])
// CHECK-NEXT: %[[RESULT:.*]] = affine.load %[[ARG0]][%[[MODIFIED_INDEX0]], %[[MODIFIED_INDEX1]], %[[ARG2]]] : memref<2x6x32xf32>
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0, d1, d2) -> (d0 * 6 + d1 * 3 + d2)>
// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape_3d
// CHECK-SAME: (%[[ARG0:.*]]: memref<12x32xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index) -> f32 {
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape_3d(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4: index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1, 2], [3]] : memref<12x32xf32> into memref<2x2x3x32xf32>
  %1 = affine.load %0[%arg1, %arg2, %arg3, %arg4] : memref<2x2x3x32xf32>
  return %1 : f32
}
// CHECK: %[[INDEX:.*]] = affine.apply #[[$MAP]](%[[ARG1]], %[[ARG2]], %[[ARG3]])
// CHECK-NEXT: %[[RESULT:.*]] = affine.load %[[ARG0]][%[[INDEX]], %[[ARG4]]] : memref<12x32xf32>
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0 * 1024 + d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<1024x1024xf32>, %[[ARG1:.*]]: memref<1xf32>, %[[ARG2:.*]]: index)
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape(%arg0: memref<1024x1024xf32>, %arg1: memref<1xf32>, %arg2: index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1], [2, 3]] : memref<1024x1024xf32> into memref<1x1024x1024x1xf32>
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
// CHECK-NEXT:     %[[IDX1:.*]] = affine.apply #[[$MAP0]](%[[ARG3]], %[[ARG4]])
// CHECK-NEXT:     %[[IDX2:.*]] = affine.apply #[[$MAP1]](%[[ARG5]], %[[ARG6]])
// CHECK-NEXT:     affine.load %[[ARG0]][%[[IDX1]], %[[IDX2]]] : memref<1024x1024xf32>

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1 + d0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0 * 1024 + d1)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape_when_access_index_is_an_expression
// CHECK-SAME: (%[[ARG0:.*]]: memref<1024x1024xf32>, %[[ARG1:.*]]: memref<1xf32>, %[[ARG2:.*]]: index)
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape_when_access_index_is_an_expression(%arg0: memref<1024x1024xf32>, %arg1: memref<1xf32>, %arg2: index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1], [2, 3]] : memref<1024x1024xf32> into memref<1x1024x1024x1xf32>
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
// CHECK-NEXT:      %[[TMP1:.*]] = affine.apply #[[$MAP0]](%[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]])
// CHECK-NEXT:      %[[TMP2:.*]] = affine.apply #[[$MAP1]](%[[ARG3]], %[[TMP1]])
// CHECK-NEXT:      %[[TMP3:.*]] = affine.apply #map2(%[[ARG5]], %[[ARG6]])
// CHECK-NEXT:      affine.load %[[ARG0]][%[[TMP2]], %[[TMP3]]] : memref<1024x1024xf32>

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0 * 1024 + d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape_with_constant_access_index
// CHECK-SAME: (%[[ARG0:.*]]: memref<1024x1024xf32>, %[[ARG1:.*]]: memref<1xf32>, %[[ARG2:.*]]: index)
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape_with_constant_access_index(%arg0: memref<1024x1024xf32>, %arg1: memref<1xf32>, %arg2: index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1], [2, 3]] : memref<1024x1024xf32> into memref<1x1024x1024x1xf32>
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
// CHECK-NEXT:   %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:   affine.for %[[ARG3:.*]] = 0 to 1 {
// CHECK-NEXT:   affine.for %[[ARG4:.*]] = 0 to 1024 {
// CHECK-NEXT:    affine.for %[[ARG5:.*]] = 0 to 1020 {
// CHECK-NEXT:     affine.for %[[ARG6:.*]] = 0 to 1 {
// CHECK-NEXT:      %[[TMP1:.*]] = affine.apply #[[$MAP0]](%[[ARG3]], %[[ZERO]])
// CHECK-NEXT:      %[[TMP2:.*]] = affine.apply #[[$MAP1]](%[[ARG5]], %[[ARG6]])
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
