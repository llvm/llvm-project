// RUN: mlir-opt -fold-memref-subview-ops -split-input-file %s -o - | FileCheck %s

func @fold_static_stride_subview_with_load(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) -> f32 {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] : memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [64, 3]>
  %1 = memref.load %0[%arg3, %arg4] : memref<4x4xf32, offset:?, strides: [64, 3]>
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

func @fold_dynamic_stride_subview_with_load(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index) -> f32 {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] :
    memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [?, ?]>
  %1 = memref.load %0[%arg3, %arg4] : memref<4x4xf32, offset:?, strides: [?, ?]>
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

func @fold_static_stride_subview_with_store(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : f32) {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] :
    memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [64, 3]>
  memref.store %arg5, %0[%arg3, %arg4] : memref<4x4xf32, offset:?, strides: [64, 3]>
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

func @fold_dynamic_stride_subview_with_store(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index, %arg7 : f32) {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] :
    memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [?, ?]>
  memref.store %arg7, %0[%arg3, %arg4] : memref<4x4xf32, offset:?, strides: [?, ?]>
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

func @fold_subview_with_transfer_read(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index) -> vector<4xf32> {
  %f1 = constant 1.0 : f32
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] : memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [?, ?]>
  %1 = vector.transfer_read %0[%arg3, %arg4], %f1 {in_bounds = [true]} : memref<4x4xf32, offset:?, strides: [?, ?]>, vector<4xf32>
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

func @fold_static_stride_subview_with_transfer_write(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5: index, %arg6 : index, %arg7 : vector<4xf32>) {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] :
    memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [?, ?]>
  vector.transfer_write %arg7, %0[%arg3, %arg4] {in_bounds = [true]} : vector<4xf32>, memref<4x4xf32, offset:?, strides: [?, ?]>
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

func @fold_rank_reducing_subview_with_load
    (%arg0 : memref<?x?x?x?x?x?xf32>, %arg1 : index, %arg2 : index,
     %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index,
     %arg7 : index, %arg8 : index, %arg9 : index, %arg10: index,
     %arg11 : index, %arg12 : index, %arg13 : index, %arg14: index,
     %arg15 : index, %arg16 : index) -> f32 {
  %0 = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4, %arg5, %arg6][4, 1, 1, 4, 1, 1][%arg7, %arg8, %arg9, %arg10, %arg11, %arg12] : memref<?x?x?x?x?x?xf32> to memref<4x1x4x1xf32, offset:?, strides: [?, ?, ?, ?]>
  %1 = memref.load %0[%arg13, %arg14, %arg15, %arg16] : memref<4x1x4x1xf32, offset:?, strides: [?, ?, ?, ?]>
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
//  CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:   %[[I1:.+]] = affine.apply #[[MAP]](%[[ARG13]])[%[[ARG7]], %[[ARG1]]]
//  CHECK-DAG:   %[[I2:.+]] = affine.apply #[[MAP]](%[[ARG14]])[%[[ARG8]], %[[ARG2]]]
//  CHECK-DAG:   %[[I3:.+]] = affine.apply #[[MAP]](%[[C0]])[%[[ARG9]], %[[ARG3]]]
//  CHECK-DAG:   %[[I4:.+]] = affine.apply #[[MAP]](%[[ARG15]])[%[[ARG10]], %[[ARG4]]]
//  CHECK-DAG:   %[[I5:.+]] = affine.apply #[[MAP]](%[[ARG16]])[%[[ARG11]], %[[ARG5]]]
//  CHECK-DAG:   %[[I6:.+]] = affine.apply #[[MAP]](%[[C0]])[%[[ARG12]], %[[ARG6]]]
//      CHECK:   memref.load %[[ARG0]][%[[I1]], %[[I2]], %[[I3]], %[[I4]], %[[I5]], %[[I6]]]
