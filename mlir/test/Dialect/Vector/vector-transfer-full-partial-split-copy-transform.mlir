// RUN: mlir-opt %s --test-transform-dialect-interpreter --split-input-file | FileCheck %s

// CHECK-DAG: #[[$map_p4:.*]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #[[$map_p8:.*]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #[[$bounds_map_4:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1, 4)>
// CHECK-DAG: #[[$bounds_map_8:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1, 8)>

// CHECK-LABEL: split_vector_transfer_read_2d(
//  CHECK-SAME: %[[A:[a-zA-Z0-9_]*]]: memref
//  CHECK-SAME: %[[i:[a-zA-Z0-9_]*]]: index
//  CHECK-SAME: %[[j:[a-zA-Z0-9_]*]]: index
func.func @split_vector_transfer_read_2d(%A: memref<?x8xf32>, %i: index, %j: index) -> vector<4x8xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

  //  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  //  CHECK-DAG: %[[c8:.*]] = arith.constant 8 : index
  // alloca for boundary full tile
  //      CHECK: %[[alloc:.*]] = memref.alloca() {alignment = 32 : i64} : memref<4x8xf32>
  // %i + 4 <= dim(%A, 0)
  //      CHECK: %[[idx0:.*]] = affine.apply #[[$map_p4]]()[%[[i]]]
  //      CHECK: %[[d0:.*]] = memref.dim %[[A]], %[[c0]] : memref<?x8xf32>
  //      CHECK: %[[cmp0:.*]] = arith.cmpi sle, %[[idx0]], %[[d0]] : index
  // %j + 8 <= dim(%A, 1)
  //      CHECK: %[[idx1:.*]] = affine.apply #[[$map_p8]]()[%[[j]]]
  //      CHECK: %[[cmp1:.*]] = arith.cmpi sle, %[[idx1]], %[[c8]] : index
  // are both conds true
  //      CHECK: %[[cond:.*]] = arith.andi %[[cmp0]], %[[cmp1]] : i1
  //      CHECK: %[[ifres:.*]]:3 = scf.if %[[cond]] -> (memref<?x8xf32>, index, index) {
  //               inBounds, just yield %A
  //      CHECK:   scf.yield %[[A]], %[[i]], %[[j]] : memref<?x8xf32>, index, index
  //      CHECK: } else {
  //               slow path, fill tmp alloc and yield a memref_casted version of it
  //      CHECK:   linalg.fill ins(%cst : f32) outs(%[[alloc]] : memref<4x8xf32>)
  //      CHECK:   %[[d0:.*]] = memref.dim %[[A]], %[[c0]] : memref<?x8xf32>
  //      CHECK:   %[[sv0:.*]] = affine.min #[[$bounds_map_4]](%[[d0]], %[[i]], %[[c4]])
  //      CHECK:   %[[sv1:.*]] = affine.min #[[$bounds_map_8]](%[[c8]], %[[j]], %[[c8]])
  //      CHECK:   %[[sv:.*]] = memref.subview %[[A]][%[[i]], %[[j]]] [%[[sv0]], %[[sv1]]] [1, 1]
  // CHECK-SAME:     memref<?x8xf32> to memref<?x?xf32, strided<[8, 1], offset: ?>>
  //      CHECK:   %[[alloc_view:.*]] = memref.subview %[[alloc]][0, 0] [%[[sv0]], %[[sv1]]] [1, 1]
  //      CHECK:   memref.copy %[[sv]], %[[alloc_view]] : memref<?x?xf32, strided<[8, 1], offset: ?>> to memref<?x?xf32, strided{{.*}}>
  //      CHECK:   %[[yielded:.*]] = memref.cast %[[alloc]] :
  // CHECK-SAME:     memref<4x8xf32> to memref<?x8xf32>
  //      CHECK:   scf.yield %[[yielded]], %[[c0]], %[[c0]] :
  // CHECK-SAME:     memref<?x8xf32>, index, index
  //      CHECK: }
  //      CHECK: %[[res:.*]] = vector.transfer_read %[[ifres]]#0[%[[ifres]]#1, %[[ifres]]#2], %cst
  // CHECK-SAME:   {in_bounds = [true, true]} : memref<?x8xf32>, vector<4x8xf32>
  %1 = vector.transfer_read %A[%i, %j], %f0 : memref<?x8xf32>, vector<4x8xf32>

  // CHECK: return %[[res]] : vector<4x8xf32>
  return %1: vector<4x8xf32>
}

// CHECK-LABEL: split_vector_transfer_read_strided_2d(
//  CHECK-SAME: %[[A:[a-zA-Z0-9_]*]]: memref
//  CHECK-SAME: %[[i:[a-zA-Z0-9_]*]]: index
//  CHECK-SAME: %[[j:[a-zA-Z0-9_]*]]: index
func.func @split_vector_transfer_read_strided_2d(
    %A: memref<7x8xf32, strided<[?, 1], offset: ?>>,
    %i: index, %j: index) -> vector<4x8xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

  
  //  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  //  CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
  //  CHECK-DAG: %[[c8:.*]] = arith.constant 8 : index
  // alloca for boundary full tile
  //      CHECK: %[[alloc:.*]] = memref.alloca() {alignment = 32 : i64} : memref<4x8xf32>
  // %i + 4 <= dim(%A, 0)
  //      CHECK: %[[idx0:.*]] = affine.apply #[[$map_p4]]()[%[[i]]]
  //      CHECK: %[[cmp0:.*]] = arith.cmpi sle, %[[idx0]], %[[c7]] : index
  // %j + 8 <= dim(%A, 1)
  //      CHECK: %[[idx1:.*]] = affine.apply #[[$map_p8]]()[%[[j]]]
  //      CHECK: %[[cmp1:.*]] = arith.cmpi sle, %[[idx1]], %[[c8]] : index
  // are both conds true
  //      CHECK: %[[cond:.*]] = arith.andi %[[cmp0]], %[[cmp1]] : i1
  //      CHECK: %[[ifres:.*]]:3 = scf.if %[[cond]] -> (memref<?x8xf32, strided<[?, 1], offset: ?>>, index, index) {
  //               inBounds but not cast-compatible: yield a memref_casted form of %A
  //      CHECK:   %[[casted:.*]] = memref.cast %arg0 :
  // CHECK-SAME:     memref<7x8xf32, strided<[?, 1], offset: ?>> to memref<?x8xf32, strided<[?, 1], offset: ?>>
  //      CHECK:   scf.yield %[[casted]], %[[i]], %[[j]] :
  // CHECK-SAME:     memref<?x8xf32, strided<[?, 1], offset: ?>>, index, index
  //      CHECK: } else {
  //               slow path, fill tmp alloc and yield a memref_casted version of it
  //      CHECK:   linalg.fill ins(%cst : f32) outs(%[[alloc]] : memref<4x8xf32>)
  //      CHECK:   %[[sv0:.*]] = affine.min #[[$bounds_map_4]](%[[c7]], %[[i]], %[[c4]])
  //      CHECK:   %[[sv1:.*]] = affine.min #[[$bounds_map_8]](%[[c8]], %[[j]], %[[c8]])
  //      CHECK:   %[[sv:.*]] = memref.subview %[[A]][%[[i]], %[[j]]] [%[[sv0]], %[[sv1]]] [1, 1]
  // CHECK-SAME:     memref<7x8xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
  //      CHECK:   %[[alloc_view:.*]] = memref.subview %[[alloc]][0, 0] [%[[sv0]], %[[sv1]]] [1, 1]
  //      CHECK:   memref.copy %[[sv]], %[[alloc_view]] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided{{.*}}>
  //      CHECK:   %[[yielded:.*]] = memref.cast %[[alloc]] :
  // CHECK-SAME:     memref<4x8xf32> to memref<?x8xf32, strided<[?, 1], offset: ?>>
  //      CHECK:   scf.yield %[[yielded]], %[[c0]], %[[c0]] :
  // CHECK-SAME:     memref<?x8xf32, strided<[?, 1], offset: ?>>, index, index
  //      CHECK: }
  //      CHECK: %[[res:.*]] = vector.transfer_read {{.*}} {in_bounds = [true, true]} :
  // CHECK-SAME:   memref<?x8xf32, strided<[?, 1], offset: ?>>, vector<4x8xf32>
  %1 = vector.transfer_read %A[%i, %j], %f0 :
    memref<7x8xf32, strided<[?, 1], offset: ?>>, vector<4x8xf32>

  return %1 : vector<4x8xf32>
}

transform.sequence failures(propagate) {
^bb1(%func_op: !transform.op<"func.func">):
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "linalg-copy"
  } : !transform.op<"func.func">
}

// -----

func.func @split_vector_transfer_write_2d(%V: vector<4x8xf32>, %A: memref<?x8xf32>, %i: index, %j: index) {
  vector.transfer_write %V, %A[%i, %j] :
    vector<4x8xf32>, memref<?x8xf32>
  return
}

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1, 4)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1, 8)>

// CHECK-LABEL:     func @split_vector_transfer_write_2d(
// CHECK-SAME:                                         %[[VEC:.*]]: vector<4x8xf32>,
// CHECK-SAME:                                         %[[DEST:.*]]: memref<?x8xf32>,
// CHECK-SAME:                                         %[[I:.*]]: index,
// CHECK-SAME:                                         %[[J:.*]]: index) {
// CHECK-DAG:       %[[CT:.*]] = arith.constant true
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[TEMP:.*]] = memref.alloca() {alignment = 32 : i64} : memref<4x8xf32>
// CHECK:           %[[IDX0:.*]] = affine.apply #[[$MAP0]]()[%[[I]]]
// CHECK:           %[[DIM0:.*]] = memref.dim %[[DEST]], %[[C0]] : memref<?x8xf32>
// CHECK:           %[[DIM0_IN:.*]] = arith.cmpi sle, %[[IDX0]], %[[DIM0]] : index
// CHECK:           %[[DIM1:.*]] = affine.apply #[[$MAP1]]()[%[[J]]]
// CHECK:           %[[DIM1_IN:.*]] = arith.cmpi sle, %[[DIM1]], %[[C8]] : index
// CHECK:           %[[IN_BOUNDS:.*]] = arith.andi %[[DIM0_IN]], %[[DIM1_IN]] : i1
// CHECK:           %[[IN_BOUND_DEST:.*]]:3 = scf.if %[[IN_BOUNDS]]
// CHECK-SAME:          -> (memref<?x8xf32>, index, index) {
// CHECK:             scf.yield %[[DEST]], %[[I]], %[[J]] : memref<?x8xf32>, index, index
// CHECK:           } else {
// CHECK:             %[[VAL_16:.*]] = memref.cast %[[TEMP]] : memref<4x8xf32> to memref<?x8xf32>
// CHECK:             scf.yield %[[VAL_16]], %[[C0]], %[[C0]] : memref<?x8xf32>, index, index
// CHECK:           }
// CHECK:           vector.transfer_write %[[VEC]],
// CHECK-SAME:          %[[IN_BOUND_DEST:.*]]#0[%[[IN_BOUND_DEST]]#1, %[[IN_BOUND_DEST]]#2]
// CHECK-SAME:          {in_bounds = [true, true]} : vector<4x8xf32>, memref<?x8xf32>
// CHECK:           %[[OUT_BOUNDS:.*]] = arith.xori %[[IN_BOUNDS]], %[[CT]] : i1
// CHECK:           scf.if %[[OUT_BOUNDS]] {
// CHECK:             %[[VAL_19:.*]] = memref.dim %[[DEST]], %[[C0]] : memref<?x8xf32>
// CHECK-DAG:         %[[VAL_20:.*]] = affine.min #[[$MAP2]](%[[VAL_19]], %[[I]], %[[C4]])
// CHECK-DAG:         %[[VAL_21:.*]] = affine.min #[[$MAP3]](%[[C8]], %[[J]], %[[C8]])
// CHECK:             %[[VAL_22:.*]] = memref.subview %[[TEMP]]
// CHECK-SAME:            [%[[I]], %[[J]]] [%[[VAL_20]], %[[VAL_21]]]
// CHECK-SAME:            [1, 1] : memref<4x8xf32> to memref<?x?xf32, strided<[8, 1], offset: ?>>
// CHECK:             %[[DEST_VIEW:.*]] = memref.subview %[[DEST]][0, 0] [%[[VAL_20]], %[[VAL_21]]] [1, 1]
// CHECK:             memref.copy %[[VAL_22]], %[[DEST_VIEW]]
// CHECK-SAME:            : memref<?x?xf32, strided<[8, 1], offset: ?>> to memref<?x?xf32, strided{{.*}}>
// CHECK:           }
// CHECK:           return
// CHECK:         }

transform.sequence failures(propagate) {
^bb1(%func_op: !transform.op<"func.func">):
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "linalg-copy"
  } : !transform.op<"func.func">
}

// -----

func.func @split_vector_transfer_write_strided_2d(
    %V: vector<4x8xf32>, %A: memref<7x8xf32, strided<[?, 1], offset: ?>>,
    %i: index, %j: index) {
  vector.transfer_write %V, %A[%i, %j] :
    vector<4x8xf32>, memref<7x8xf32, strided<[?, 1], offset: ?>>
  return
}

// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1, 4)>
// CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1, 8)>
// CHECK-LABEL:   func @split_vector_transfer_write_strided_2d(
// CHECK-SAME:                                                 %[[VEC:.*]]: vector<4x8xf32>,
// CHECK-SAME:                                                 %[[DEST:.*]]: memref<7x8xf32, strided<[?, 1], offset: ?>>,
// CHECK-SAME:                                                 %[[I:.*]]: index,
// CHECK-SAME:                                                 %[[J:.*]]: index) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[CT:.*]] = arith.constant true
// CHECK-DAG:       %[[C7:.*]] = arith.constant 7 : index
// CHECK-DAG:       %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[TEMP:.*]] = memref.alloca() {alignment = 32 : i64} : memref<4x8xf32>
// CHECK:           %[[DIM0:.*]] = affine.apply #[[$MAP1]]()[%[[I]]]
// CHECK:           %[[DIM0_IN:.*]] = arith.cmpi sle, %[[DIM0]], %[[C7]] : index
// CHECK:           %[[DIM1:.*]] = affine.apply #[[$MAP2]]()[%[[J]]]
// CHECK:           %[[DIM1_IN:.*]] = arith.cmpi sle, %[[DIM1]], %[[C8]] : index
// CHECK:           %[[IN_BOUNDS:.*]] = arith.andi %[[DIM0_IN]], %[[DIM1_IN]] : i1
// CHECK:           %[[IN_BOUND_DEST:.*]]:3 = scf.if %[[IN_BOUNDS]]
// CHECK-SAME:          -> (memref<?x8xf32, strided<[?, 1], offset: ?>>, index, index) {
// CHECK:             %[[VAL_16:.*]] = memref.cast %[[DEST]]
// CHECK-SAME:            : memref<7x8xf32, strided<[?, 1], offset: ?>> to memref<?x8xf32, strided<[?, 1], offset: ?>>
// CHECK:             scf.yield %[[VAL_16]], %[[I]], %[[J]]
// CHECK-SAME:            : memref<?x8xf32, strided<[?, 1], offset: ?>>, index, index
// CHECK:           } else {
// CHECK:             %[[VAL_17:.*]] = memref.cast %[[TEMP]]
// CHECK-SAME:            : memref<4x8xf32> to memref<?x8xf32, strided<[?, 1], offset: ?>>
// CHECK:             scf.yield %[[VAL_17]], %[[C0]], %[[C0]]
// CHECK-SAME:            : memref<?x8xf32, strided<[?, 1], offset: ?>>, index, index
// CHECK:           }
// CHECK:           vector.transfer_write %[[VEC]],
// CHECK-SAME:          %[[IN_BOUND_DEST:.*]]#0
// CHECK-SAME:          [%[[IN_BOUND_DEST]]#1, %[[IN_BOUND_DEST]]#2]
// CHECK-SAME:          {in_bounds = [true, true]}
// CHECK-SAME:          : vector<4x8xf32>, memref<?x8xf32, strided<[?, 1], offset: ?>>
// CHECK:           %[[OUT_BOUNDS:.*]] = arith.xori %[[IN_BOUNDS]], %[[CT]] : i1
// CHECK:           scf.if %[[OUT_BOUNDS]] {
// CHECK-DAG:         %[[VAL_20:.*]] = affine.min #[[$MAP3]](%[[C7]], %[[I]], %[[C4]])
// CHECK-DAG:         %[[VAL_21:.*]] = affine.min #[[$MAP4]](%[[C8]], %[[J]], %[[C8]])
// CHECK:             %[[VAL_22:.*]] = memref.subview %[[TEMP]]
// CHECK-SAME:            [%[[I]], %[[J]]] [%[[VAL_20]], %[[VAL_21]]]
// CHECK-SAME:            [1, 1] : memref<4x8xf32> to memref<?x?xf32, strided<[8, 1], offset: ?>>
// CHECK:             %[[DEST_VIEW:.*]] = memref.subview %[[DEST]][0, 0] [%[[VAL_20]], %[[VAL_21]]] [1, 1]
// CHECK:             memref.copy %[[VAL_22]], %[[DEST_VIEW]]
// CHECK-SAME:            : memref<?x?xf32, strided<[8, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:           }
// CHECK:           return
// CHECK:         }

transform.sequence failures(propagate) {
^bb1(%func_op: !transform.op<"func.func">):
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "linalg-copy"
  } : !transform.op<"func.func">
}
