// RUN: mlir-opt %s -test-vector-transfer-full-partial-split | FileCheck %s
// RUN: mlir-opt %s -test-vector-transfer-full-partial-split=use-linalg-copy | FileCheck %s --check-prefix=LINALG

// CHECK-DAG: #[[$map_p4:.*]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #[[$map_p8:.*]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #[[$map_2d_stride_1:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>

// LINALG-DAG: #[[$map_p4:.*]] = affine_map<()[s0] -> (s0 + 4)>
// LINALG-DAG: #[[$map_p8:.*]] = affine_map<()[s0] -> (s0 + 8)>
// LINALG-DAG: #[[$map_2d_stride_1:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// LINALG-DAG: #[[$map_2d_stride_8x1:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 8 + s0 + d1)>
// LINALG-DAG: #[[$bounds_map_4:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1, 4)>
// LINALG-DAG: #[[$bounds_map_8:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1, 8)>

// CHECK-LABEL: split_vector_transfer_read_2d(
//  CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: memref
//  CHECK-SAME: %[[i:[a-zA-Z0-9]*]]: index
//  CHECK-SAME: %[[j:[a-zA-Z0-9]*]]: index

// LINALG-LABEL: split_vector_transfer_read_2d(
//  LINALG-SAME: %[[A:[a-zA-Z0-9]*]]: memref
//  LINALG-SAME: %[[i:[a-zA-Z0-9]*]]: index
//  LINALG-SAME: %[[j:[a-zA-Z0-9]*]]: index
func @split_vector_transfer_read_2d(%A: memref<?x8xf32>, %i: index, %j: index) -> vector<4x8xf32> {
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32

  //  CHECK-DAG: %[[c0:.*]] = constant 0 : index
  //  CHECK-DAG: %[[c8:.*]] = constant 8 : index
  //  CHECK-DAG: %[[cst:.*]] = constant 0.000000e+00 : f32
  // alloca for boundary full tile
  //      CHECK: %[[alloc:.*]] = memref.alloca() {alignment = 32 : i64} : memref<4x8xf32>
  // %i + 4 <= dim(%A, 0)
  //      CHECK: %[[idx0:.*]] = affine.apply #[[$map_p4]]()[%[[i]]]
  //      CHECK: %[[d0:.*]] = memref.dim %[[A]], %[[c0]] : memref<?x8xf32>
  //      CHECK: %[[cmp0:.*]] = cmpi sle, %[[idx0]], %[[d0]] : index
  // %j + 8 <= dim(%A, 1)
  //      CHECK: %[[idx1:.*]] = affine.apply #[[$map_p8]]()[%[[j]]]
  //      CHECK: %[[cmp1:.*]] = cmpi sle, %[[idx1]], %[[c8]] : index
  // are both conds true
  //      CHECK: %[[cond:.*]] = and %[[cmp0]], %[[cmp1]] : i1
  //      CHECK: %[[ifres:.*]]:3 = scf.if %[[cond]] -> (memref<?x8xf32>, index, index) {
  //               inBounds, just yield %A
  //      CHECK:   scf.yield %[[A]], %[[i]], %[[j]] : memref<?x8xf32>, index, index
  //      CHECK: } else {
  //               slow path, fill tmp alloc and yield a memref_casted version of it
  //      CHECK:   %[[slow:.*]] = vector.transfer_read %[[A]][%[[i]], %[[j]]], %cst :
  // CHECK-SAME:     memref<?x8xf32>, vector<4x8xf32>
  //      CHECK:   %[[cast_alloc:.*]] = vector.type_cast %[[alloc]] :
  // CHECK-SAME:     memref<4x8xf32> to memref<vector<4x8xf32>>
  //      CHECK:   store %[[slow]], %[[cast_alloc]][] : memref<vector<4x8xf32>>
  //      CHECK:   %[[yielded:.*]] = memref.cast %[[alloc]] :
  // CHECK-SAME:     memref<4x8xf32> to memref<?x8xf32>
  //      CHECK:   scf.yield %[[yielded]], %[[c0]], %[[c0]] :
  // CHECK-SAME:     memref<?x8xf32>, index, index
  //      CHECK: }
  //      CHECK: %[[res:.*]] = vector.transfer_read %[[ifres]]#0[%[[ifres]]#1, %[[ifres]]#2], %[[cst]]
  // CHECK_SAME:   {masked = [false, false]} : memref<?x8xf32>, vector<4x8xf32>

  //  LINALG-DAG: %[[c0:.*]] = constant 0 : index
  //  LINALG-DAG: %[[c4:.*]] = constant 4 : index
  //  LINALG-DAG: %[[c8:.*]] = constant 8 : index
  //  LINALG-DAG: %[[cst:.*]] = constant 0.000000e+00 : f32
  // alloca for boundary full tile
  //      LINALG: %[[alloc:.*]] = memref.alloca() {alignment = 32 : i64} : memref<4x8xf32>
  // %i + 4 <= dim(%A, 0)
  //      LINALG: %[[idx0:.*]] = affine.apply #[[$map_p4]]()[%[[i]]]
  //      LINALG: %[[d0:.*]] = memref.dim %[[A]], %[[c0]] : memref<?x8xf32>
  //      LINALG: %[[cmp0:.*]] = cmpi sle, %[[idx0]], %[[d0]] : index
  // %j + 8 <= dim(%A, 1)
  //      LINALG: %[[idx1:.*]] = affine.apply #[[$map_p8]]()[%[[j]]]
  //      LINALG: %[[cmp1:.*]] = cmpi sle, %[[idx1]], %[[c8]] : index
  // are both conds true
  //      LINALG: %[[cond:.*]] = and %[[cmp0]], %[[cmp1]] : i1
  //      LINALG: %[[ifres:.*]]:3 = scf.if %[[cond]] -> (memref<?x8xf32>, index, index) {
  //               inBounds, just yield %A
  //      LINALG:   scf.yield %[[A]], %[[i]], %[[j]] : memref<?x8xf32>, index, index
  //      LINALG: } else {
  //               slow path, fill tmp alloc and yield a memref_casted version of it
  //      LINALG:   linalg.fill(%[[alloc]], %[[cst]]) : memref<4x8xf32>, f32
  //      LINALG:   %[[d0:.*]] = memref.dim %[[A]], %[[c0]] : memref<?x8xf32>
  //      LINALG:   %[[sv0:.*]] = affine.min #[[$bounds_map_4]](%[[d0]], %[[i]], %[[c4]])
  //      LINALG:   %[[sv1:.*]] = affine.min #[[$bounds_map_8]](%[[c8]], %[[j]], %[[c8]])
  //      LINALG:   %[[sv:.*]] = memref.subview %[[A]][%[[i]], %[[j]]] [%[[sv0]], %[[sv1]]] [1, 1]
  // LINALG-SAME:     memref<?x8xf32> to memref<?x?xf32, #[[$map_2d_stride_8x1]]>
  //      LINALG:   linalg.copy(%[[sv]], %[[alloc]]) : memref<?x?xf32, #[[$map_2d_stride_8x1]]>, memref<4x8xf32>
  //      LINALG:   %[[yielded:.*]] = memref.cast %[[alloc]] :
  // LINALG-SAME:     memref<4x8xf32> to memref<?x8xf32>
  //      LINALG:   scf.yield %[[yielded]], %[[c0]], %[[c0]] :
  // LINALG-SAME:     memref<?x8xf32>, index, index
  //      LINALG: }
  //      LINALG: %[[res:.*]] = vector.transfer_read %[[ifres]]#0[%[[ifres]]#1, %[[ifres]]#2], %[[cst]]
  // LINALG_SAME:   {masked = [false, false]} : memref<?x8xf32>, vector<4x8xf32>
  %1 = vector.transfer_read %A[%i, %j], %f0 : memref<?x8xf32>, vector<4x8xf32>

  // LINALG: return %[[res]] : vector<4x8xf32>
  return %1: vector<4x8xf32>
}

// CHECK-LABEL: split_vector_transfer_read_strided_2d(
//  CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: memref
//  CHECK-SAME: %[[i:[a-zA-Z0-9]*]]: index
//  CHECK-SAME: %[[j:[a-zA-Z0-9]*]]: index

// LINALG-LABEL: split_vector_transfer_read_strided_2d(
//  LINALG-SAME: %[[A:[a-zA-Z0-9]*]]: memref
//  LINALG-SAME: %[[i:[a-zA-Z0-9]*]]: index
//  LINALG-SAME: %[[j:[a-zA-Z0-9]*]]: index
func @split_vector_transfer_read_strided_2d(
    %A: memref<7x8xf32, offset:?, strides:[?, 1]>,
    %i: index, %j: index) -> vector<4x8xf32> {
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32

  //  CHECK-DAG: %[[c0:.*]] = constant 0 : index
  //  CHECK-DAG: %[[c7:.*]] = constant 7 : index
  //  CHECK-DAG: %[[c8:.*]] = constant 8 : index
  //  CHECK-DAG: %[[cst:.*]] = constant 0.000000e+00 : f32
  // alloca for boundary full tile
  //      CHECK: %[[alloc:.*]] = memref.alloca() {alignment = 32 : i64} : memref<4x8xf32>
  // %i + 4 <= dim(%A, 0)
  //      CHECK: %[[idx0:.*]] = affine.apply #[[$map_p4]]()[%[[i]]]
  //      CHECK: %[[cmp0:.*]] = cmpi sle, %[[idx0]], %[[c7]] : index
  // %j + 8 <= dim(%A, 1)
  //      CHECK: %[[idx1:.*]] = affine.apply #[[$map_p8]]()[%[[j]]]
  //      CHECK: %[[cmp1:.*]] = cmpi sle, %[[idx1]], %[[c8]] : index
  // are both conds true
  //      CHECK: %[[cond:.*]] = and %[[cmp0]], %[[cmp1]] : i1
  //      CHECK: %[[ifres:.*]]:3 = scf.if %[[cond]] -> (memref<?x8xf32, #[[$map_2d_stride_1]]>, index, index) {
  //               inBounds but not cast-compatible: yield a memref_casted form of %A
  //      CHECK:   %[[casted:.*]] = memref.cast %arg0 :
  // CHECK-SAME:     memref<7x8xf32, #[[$map_2d_stride_1]]> to memref<?x8xf32, #[[$map_2d_stride_1]]>
  //      CHECK:   scf.yield %[[casted]], %[[i]], %[[j]] :
  // CHECK-SAME:     memref<?x8xf32, #[[$map_2d_stride_1]]>, index, index
  //      CHECK: } else {
  //               slow path, fill tmp alloc and yield a memref_casted version of it
  //      CHECK:   %[[slow:.*]] = vector.transfer_read %[[A]][%[[i]], %[[j]]], %cst :
  // CHECK-SAME:     memref<7x8xf32, #[[$map_2d_stride_1]]>, vector<4x8xf32>
  //      CHECK:   %[[cast_alloc:.*]] = vector.type_cast %[[alloc]] :
  // CHECK-SAME:     memref<4x8xf32> to memref<vector<4x8xf32>>
  //      CHECK:   store %[[slow]], %[[cast_alloc]][] :
  // CHECK-SAME:     memref<vector<4x8xf32>>
  //      CHECK:   %[[yielded:.*]] = memref.cast %[[alloc]] :
  // CHECK-SAME:     memref<4x8xf32> to memref<?x8xf32, #[[$map_2d_stride_1]]>
  //      CHECK:   scf.yield %[[yielded]], %[[c0]], %[[c0]] :
  // CHECK-SAME:     memref<?x8xf32, #[[$map_2d_stride_1]]>, index, index
  //      CHECK: }
  //      CHECK: %[[res:.*]] = vector.transfer_read {{.*}} {masked = [false, false]} :
  // CHECK-SAME:   memref<?x8xf32, #[[$map_2d_stride_1]]>, vector<4x8xf32>

  //  LINALG-DAG: %[[c0:.*]] = constant 0 : index
  //  LINALG-DAG: %[[c4:.*]] = constant 4 : index
  //  LINALG-DAG: %[[c7:.*]] = constant 7 : index
  //  LINALG-DAG: %[[c8:.*]] = constant 8 : index
  //  LINALG-DAG: %[[cst:.*]] = constant 0.000000e+00 : f32
  // alloca for boundary full tile
  //      LINALG: %[[alloc:.*]] = memref.alloca() {alignment = 32 : i64} : memref<4x8xf32>
  // %i + 4 <= dim(%A, 0)
  //      LINALG: %[[idx0:.*]] = affine.apply #[[$map_p4]]()[%[[i]]]
  //      LINALG: %[[cmp0:.*]] = cmpi sle, %[[idx0]], %[[c7]] : index
  // %j + 8 <= dim(%A, 1)
  //      LINALG: %[[idx1:.*]] = affine.apply #[[$map_p8]]()[%[[j]]]
  //      LINALG: %[[cmp1:.*]] = cmpi sle, %[[idx1]], %[[c8]] : index
  // are both conds true
  //      LINALG: %[[cond:.*]] = and %[[cmp0]], %[[cmp1]] : i1
  //      LINALG: %[[ifres:.*]]:3 = scf.if %[[cond]] -> (memref<?x8xf32, #[[$map_2d_stride_1]]>, index, index) {
  //               inBounds but not cast-compatible: yield a memref_casted form of %A
  //      LINALG:   %[[casted:.*]] = memref.cast %arg0 :
  // LINALG-SAME:     memref<7x8xf32, #[[$map_2d_stride_1]]> to memref<?x8xf32, #[[$map_2d_stride_1]]>
  //      LINALG:   scf.yield %[[casted]], %[[i]], %[[j]] :
  // LINALG-SAME:     memref<?x8xf32, #[[$map_2d_stride_1]]>, index, index
  //      LINALG: } else {
  //               slow path, fill tmp alloc and yield a memref_casted version of it
  //      LINALG:   linalg.fill(%[[alloc]], %[[cst]]) : memref<4x8xf32>, f32
  //      LINALG:   %[[sv0:.*]] = affine.min #[[$bounds_map_4]](%[[c7]], %[[i]], %[[c4]])
  //      LINALG:   %[[sv1:.*]] = affine.min #[[$bounds_map_8]](%[[c8]], %[[j]], %[[c8]])
  //      LINALG:   %[[sv:.*]] = memref.subview %[[A]][%[[i]], %[[j]]] [%[[sv0]], %[[sv1]]] [1, 1]
  // LINALG-SAME:     memref<7x8xf32, #[[$map_2d_stride_1]]> to memref<?x?xf32, #[[$map_2d_stride_1]]>
  //      LINALG:   linalg.copy(%[[sv]], %[[alloc]]) : memref<?x?xf32, #[[$map_2d_stride_1]]>, memref<4x8xf32>
  //      LINALG:   %[[yielded:.*]] = memref.cast %[[alloc]] :
  // LINALG-SAME:     memref<4x8xf32> to memref<?x8xf32, #[[$map_2d_stride_1]]>
  //      LINALG:   scf.yield %[[yielded]], %[[c0]], %[[c0]] :
  // LINALG-SAME:     memref<?x8xf32, #[[$map_2d_stride_1]]>, index, index
  //      LINALG: }
  //      LINALG: %[[res:.*]] = vector.transfer_read {{.*}} {masked = [false, false]} :
  // LINALG-SAME:   memref<?x8xf32, #[[$map_2d_stride_1]]>, vector<4x8xf32>
  %1 = vector.transfer_read %A[%i, %j], %f0 :
    memref<7x8xf32, offset:?, strides:[?, 1]>, vector<4x8xf32>

  // CHECK: return %[[res]] : vector<4x8xf32>
  return %1 : vector<4x8xf32>
}
