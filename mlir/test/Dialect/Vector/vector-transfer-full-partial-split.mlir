// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s


// CHECK-DAG: #[[$map_p4:.*]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #[[$map_p8:.*]] = affine_map<()[s0] -> (s0 + 8)>

// CHECK-LABEL: split_vector_transfer_read_2d(
//  CHECK-SAME: %[[A:[a-zA-Z0-9_]*]]: memref
//  CHECK-SAME: %[[i:[a-zA-Z0-9_]*]]: index
//  CHECK-SAME: %[[j:[a-zA-Z0-9_]*]]: index

func.func @split_vector_transfer_read_2d(%A: memref<?x8xf32>, %i: index, %j: index) -> vector<4x8xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

  //  CHECK-DAG: %[[c8:.*]] = arith.constant 8 : index
  //  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
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
  //      CHECK: %[[res:.*]] = vector.transfer_read %[[ifres]]#0[%[[ifres]]#1, %[[ifres]]#2], %cst
  // CHECK-SAME:   {in_bounds = [true, true]} : memref<?x8xf32>, vector<4x8xf32>

  %1 = vector.transfer_read %A[%i, %j], %f0 : memref<?x8xf32>, vector<4x8xf32>

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

  //  CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
  //  CHECK-DAG: %[[c8:.*]] = arith.constant 8 : index
  //  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
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
  //      CHECK:   %[[slow:.*]] = vector.transfer_read %[[A]][%[[i]], %[[j]]], %cst :
  // CHECK-SAME:     memref<7x8xf32, strided<[?, 1], offset: ?>>, vector<4x8xf32>
  //      CHECK:   %[[cast_alloc:.*]] = vector.type_cast %[[alloc]] :
  // CHECK-SAME:     memref<4x8xf32> to memref<vector<4x8xf32>>
  //      CHECK:   store %[[slow]], %[[cast_alloc]][] :
  // CHECK-SAME:     memref<vector<4x8xf32>>
  //      CHECK:   %[[yielded:.*]] = memref.cast %[[alloc]] :
  // CHECK-SAME:     memref<4x8xf32> to memref<?x8xf32, strided<[?, 1], offset: ?>>
  //      CHECK:   scf.yield %[[yielded]], %[[c0]], %[[c0]] :
  // CHECK-SAME:     memref<?x8xf32, strided<[?, 1], offset: ?>>, index, index
  //      CHECK: }
  //      CHECK: %[[res:.*]] = vector.transfer_read {{.*}} {in_bounds = [true, true]} :
  // CHECK-SAME:   memref<?x8xf32, strided<[?, 1], offset: ?>>, vector<4x8xf32>
  %1 = vector.transfer_read %A[%i, %j], %f0 :
    memref<7x8xf32, strided<[?, 1], offset: ?>>, vector<4x8xf32>

  // CHECK: return %[[res]] : vector<4x8xf32>
  return %1 : vector<4x8xf32>
}

func.func @split_vector_transfer_read_mem_space(%A: memref<?x8xf32, 3>, %i: index, %j: index) -> vector<4x8xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

  //      CHECK: scf.if {{.*}} -> (memref<?x8xf32, strided<[8, 1]>>, index, index) {
  //               inBounds with a different memory space
  //      CHECK:   %[[space_cast:.*]] = memref.memory_space_cast %{{.*}} :
  // CHECK-SAME:     memref<?x8xf32, 3> to memref<?x8xf32>
  //      CHECK:   %[[cast:.*]] = memref.cast %[[space_cast]] :
  // CHECK-SAME:     memref<?x8xf32> to memref<?x8xf32, strided<[8, 1]>>
  //      CHECK:   scf.yield %[[cast]], {{.*}} : memref<?x8xf32, strided<[8, 1]>>, index, index
  //      CHECK: } else {
  //               slow path, fill tmp alloc and yield a memref_casted version of it
  //      CHECK:   %[[slow:.*]] = vector.transfer_read %[[A]][%[[i]], %[[j]]], %cst :
  // CHECK-SAME:     memref<?x8xf32, 3>, vector<4x8xf32>
  //      CHECK:   %[[cast_alloc:.*]] = vector.type_cast %[[alloc]] :
  // CHECK-SAME:     memref<4x8xf32> to memref<vector<4x8xf32>>
  //      CHECK:   store %[[slow]], %[[cast_alloc]][] : memref<vector<4x8xf32>>
  //      CHECK:   %[[yielded:.*]] = memref.cast %[[alloc]] :
  // CHECK-SAME:     memref<4x8xf32> to memref<?x8xf32, strided<[8, 1]>>
  //      CHECK:   scf.yield %[[yielded]], %[[c0]], %[[c0]] :
  // CHECK-SAME:     memref<?x8xf32, strided<[8, 1]>>, index, index
  //      CHECK: }
  //      CHECK: %[[res:.*]] = vector.transfer_read %[[ifres]]#0[%[[ifres]]#1, %[[ifres]]#2], %cst
  // CHECK-SAME:   {in_bounds = [true, true]} : memref<?x8xf32, strided<[8, 1]>>, vector<4x8xf32>

  %1 = vector.transfer_read %A[%i, %j], %f0 : memref<?x8xf32, 3>, vector<4x8xf32>

  return %1: vector<4x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%func_op: !transform.op<"func.func"> {transform.readonly}) {
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "vector-transfer"
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

func.func @split_vector_transfer_write_2d(%V: vector<4x8xf32>, %A: memref<?x8xf32>, %i: index, %j: index) {
  vector.transfer_write %V, %A[%i, %j] :
    vector<4x8xf32>, memref<?x8xf32>
  return
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK:     func @split_vector_transfer_write_2d(
// CHECK-SAME:                                         %[[VEC:.*]]: vector<4x8xf32>,
// CHECK-SAME:                                         %[[DEST:.*]]: memref<?x8xf32>,
// CHECK-SAME:                                         %[[I:.*]]: index,
// CHECK-SAME:                                         %[[J:.*]]: index) {
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[CT:.*]] = arith.constant true
// CHECK:           %[[TEMP:.*]] = memref.alloca() {alignment = 32 : i64} : memref<4x8xf32>
// CHECK:           %[[VAL_8:.*]] = affine.apply #[[MAP0]]()[%[[I]]]
// CHECK:           %[[DIM0:.*]] = memref.dim %[[DEST]], %[[C0]] : memref<?x8xf32>
// CHECK:           %[[DIM0_IN:.*]] = arith.cmpi sle, %[[VAL_8]], %[[DIM0]] : index
// CHECK:           %[[DIM1:.*]] = affine.apply #[[MAP1]]()[%[[J]]]
// CHECK:           %[[DIM1_IN:.*]] = arith.cmpi sle, %[[DIM1]], %[[C8]] : index
// CHECK:           %[[IN_BOUNDS:.*]] = arith.andi %[[DIM0_IN]], %[[DIM1_IN]] : i1
// CHECK:           %[[IN_BOUND_DEST:.*]]:3 = scf.if %[[IN_BOUNDS]] ->
// CHECK-SAME:          (memref<?x8xf32>, index, index) {
// CHECK:             scf.yield %[[DEST]], %[[I]], %[[J]] : memref<?x8xf32>, index, index
// CHECK:           } else {
// CHECK:             %[[VAL_15:.*]] = memref.cast %[[TEMP]]
// CHECK-SAME:            : memref<4x8xf32> to memref<?x8xf32>
// CHECK:             scf.yield %[[VAL_15]], %[[C0]], %[[C0]]
// CHECK-SAME:            : memref<?x8xf32>, index, index
// CHECK:           }
// CHECK:           vector.transfer_write %[[VEC]],
// CHECK-SAME:           %[[IN_BOUND_DEST:.*]]#0[%[[IN_BOUND_DEST]]#1, %[[IN_BOUND_DEST]]#2]
// CHECK-SAME:           {in_bounds = [true, true]} : vector<4x8xf32>, memref<?x8xf32>
// CHECK:           %[[OUT_BOUNDS:.*]] = arith.xori %[[IN_BOUNDS]], %[[CT]] : i1
// CHECK:           scf.if %[[OUT_BOUNDS]] {
// CHECK:             %[[CASTED:.*]] = vector.type_cast %[[TEMP]]
// CHECK-SAME:            : memref<4x8xf32> to memref<vector<4x8xf32>>
// CHECK:             %[[RESULT_COPY:.*]] = memref.load %[[CASTED]][]
// CHECK-SAME:            : memref<vector<4x8xf32>>
// CHECK:             vector.transfer_write %[[RESULT_COPY]],
// CHECK-SAME:            %[[DEST]][%[[I]], %[[J]]]
// CHECK-SAME:            : vector<4x8xf32>, memref<?x8xf32>
// CHECK:           }
// CHECK:           return
// CHECK:         }


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%func_op: !transform.op<"func.func"> {transform.readonly}) {
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "vector-transfer"
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

func.func @split_vector_transfer_write_strided_2d(
    %V: vector<4x8xf32>, %A: memref<7x8xf32, strided<[?, 1], offset: ?>>,
    %i: index, %j: index) {
  vector.transfer_write %V, %A[%i, %j] :
    vector<4x8xf32>, memref<7x8xf32, strided<[?, 1], offset: ?>>
  return
}

// CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK:   func @split_vector_transfer_write_strided_2d(
// CHECK-SAME:                                                 %[[VEC:.*]]: vector<4x8xf32>,
// CHECK-SAME:                                                 %[[DEST:.*]]: memref<7x8xf32, strided<[?, 1], offset: ?>>,
// CHECK-SAME:                                                 %[[I:.*]]: index,
// CHECK-SAME:                                                 %[[J:.*]]: index) {
// CHECK-DAG:       %[[C7:.*]] = arith.constant 7 : index
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[CT:.*]] = arith.constant true
// CHECK:           %[[TEMP:.*]] = memref.alloca() {alignment = 32 : i64} : memref<4x8xf32>
// CHECK:           %[[DIM0:.*]] = affine.apply #[[MAP1]]()[%[[I]]]
// CHECK:           %[[DIM0_IN:.*]] = arith.cmpi sle, %[[DIM0]], %[[C7]] : index
// CHECK:           %[[DIM1:.*]] = affine.apply #[[MAP2]]()[%[[J]]]
// CHECK:           %[[DIM1_IN:.*]] = arith.cmpi sle, %[[DIM1]], %[[C8]] : index
// CHECK:           %[[IN_BOUNDS:.*]] = arith.andi %[[DIM0_IN]], %[[DIM1_IN]] : i1
// CHECK:           %[[IN_BOUND_DEST:.*]]:3 = scf.if %[[IN_BOUNDS]]
// CHECK-SAME:          -> (memref<?x8xf32, strided<[?, 1], offset: ?>>, index, index) {
// CHECK:             %[[VAL_15:.*]] = memref.cast %[[DEST]]
// CHECK-SAME:            : memref<7x8xf32, strided<[?, 1], offset: ?>> to memref<?x8xf32, strided<[?, 1], offset: ?>>
// CHECK:             scf.yield %[[VAL_15]], %[[I]], %[[J]]
// CHECK-SAME:            : memref<?x8xf32, strided<[?, 1], offset: ?>>, index, index
// CHECK:           } else {
// CHECK:             %[[VAL_16:.*]] = memref.cast %[[TEMP]]
// CHECK-SAME:            : memref<4x8xf32> to memref<?x8xf32, strided<[?, 1], offset: ?>>
// CHECK:             scf.yield %[[VAL_16]], %[[C0]], %[[C0]]
// CHECK-SAME:            : memref<?x8xf32, strided<[?, 1], offset: ?>>, index, index
// CHECK:           }
// CHECK:           vector.transfer_write %[[VEC]],
// CHECK-SAME:          %[[IN_BOUND_DEST:.*]]#0
// CHECK-SAME:          [%[[IN_BOUND_DEST]]#1, %[[IN_BOUND_DEST]]#2]
// CHECK-SAME:          {in_bounds = [true, true]} : vector<4x8xf32>, memref<?x8xf32, strided<[?, 1], offset: ?>>
// CHECK:           %[[OUT_BOUNDS:.*]] = arith.xori %[[IN_BOUNDS]], %[[CT]] : i1
// CHECK:           scf.if %[[OUT_BOUNDS]] {
// CHECK:             %[[VAL_19:.*]] = vector.type_cast %[[TEMP]]
// CHECK-SAME:            : memref<4x8xf32> to memref<vector<4x8xf32>>
// CHECK:             %[[VAL_20:.*]] = memref.load %[[VAL_19]][]
// CHECK-SAME:            : memref<vector<4x8xf32>>
// CHECK:             vector.transfer_write %[[VAL_20]], %[[DEST]][%[[I]], %[[J]]]
// CHECK-SAME:            : vector<4x8xf32>, memref<7x8xf32, strided<[?, 1], offset: ?>>
// CHECK:           }
// CHECK:           return
// CHECK:         }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%func_op: !transform.op<"func.func"> {transform.readonly}) {
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "vector-transfer"
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

func.func @split_vector_transfer_write_mem_space(%V: vector<4x8xf32>, %A: memref<?x8xf32, 3>, %i: index, %j: index) {
  vector.transfer_write %V, %A[%i, %j] :
    vector<4x8xf32>, memref<?x8xf32, 3>
  return
}

// CHECK:     func @split_vector_transfer_write_mem_space(
// CHECK:           scf.if {{.*}} -> (memref<?x8xf32, strided<[8, 1]>>, index, index) {
// CHECK:             %[[space_cast:.*]] = memref.memory_space_cast %{{.*}} :
// CHECK-SAME:          memref<?x8xf32, 3> to memref<?x8xf32>
// CHECK:             %[[cast:.*]] = memref.cast %[[space_cast]] :
// CHECK-SAME:          memref<?x8xf32> to memref<?x8xf32, strided<[8, 1]>>
// CHECK:             scf.yield %[[cast]], {{.*}} : memref<?x8xf32, strided<[8, 1]>>, index, index
// CHECK:           } else {
// CHECK:             %[[VAL_15:.*]] = memref.cast %[[TEMP]]
// CHECK-SAME:            : memref<4x8xf32> to memref<?x8xf32, strided<[8, 1]>>
// CHECK:             scf.yield %[[VAL_15]], %[[C0]], %[[C0]]
// CHECK-SAME:            : memref<?x8xf32, strided<[8, 1]>>, index, index
// CHECK:           }
// CHECK:           vector.transfer_write %[[VEC]],
// CHECK-SAME:           %[[IN_BOUND_DEST:.*]]#0[%[[IN_BOUND_DEST]]#1, %[[IN_BOUND_DEST]]#2]
// CHECK-SAME:           {in_bounds = [true, true]} : vector<4x8xf32>, memref<?x8xf32, strided<[8, 1]>>


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%func_op: !transform.op<"func.func"> {transform.readonly}) {
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "vector-transfer"
    } : !transform.op<"func.func">
    transform.yield
  }
}


// -----

func.func private @fake_side_effecting_fun(%0: vector<2x2xf32>) -> ()

// CHECK-LABEL: transfer_read_within_async_execute
func.func @transfer_read_within_async_execute(%A : memref<?x?xf32>) -> !async.token {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  // CHECK-NOT: alloca
  //     CHECK: async.execute
  //     CHECK:   alloca
  %token = async.execute {
    %0 = vector.transfer_read %A[%c0, %c0], %f0 : memref<?x?xf32>, vector<2x2xf32>
    func.call @fake_side_effecting_fun(%0) : (vector<2x2xf32>) -> ()
    async.yield
  }
  return %token : !async.token
}

// Ensure that `alloca`s are inserted outside of loops even though loops are
// consdered allocation scopes.
// CHECK-LABEL: transfer_read_within_scf_for
func.func @transfer_read_within_scf_for(%A : memref<?x?xf32>, %lb : index, %ub : index, %step : index) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  // CHECK: memref.alloca
  // CHECK: scf.for
  // CHECK-NOT: memref.alloca
  scf.for %i = %lb to %ub step %step {
    %0 = vector.transfer_read %A[%c0, %c0], %f0 : memref<?x?xf32>, vector<2x2xf32>
    func.call @fake_side_effecting_fun(%0) : (vector<2x2xf32>) -> ()
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%func_op: !transform.op<"func.func"> {transform.readonly}) {
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "vector-transfer"
    } : !transform.op<"func.func">
    transform.yield
  }
}
