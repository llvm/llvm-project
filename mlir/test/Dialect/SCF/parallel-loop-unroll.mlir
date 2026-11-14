// RUN: mlir-opt %s -test-parallel-loop-unrolling='unroll-factors=1,2' -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-parallel-loop-unrolling='unroll-factors=1,2 loop-depth=1' -split-input-file | FileCheck %s --check-prefix CHECK-UNROLL-INNER
// RUN: mlir-opt %s -test-parallel-loop-unrolling='unroll-factors=3,1' -split-input-file | FileCheck %s --check-prefix CHECK-UNROLL-BY-3
// RUN: mlir-opt %s -test-parallel-loop-unrolling='unroll-factors=3,2' -split-input-file | FileCheck %s --check-prefix CHECK-UNROLL-3-2

func.func @unroll_simple_parallel_loop(%src: memref<1x16x12xf32>, %dst: memref<1x16x12xf32>) {
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c1, %c16, %c12) step (%c1, %c1, %c1) {
    %read = memref.load %src[%arg2, %arg3, %arg4] : memref<1x16x12xf32>
    memref.store %read, %dst[%arg2, %arg3, %arg4] : memref<1x16x12xf32>
    scf.reduce
  }
  return
}

// CHECK-LABEL:   func @unroll_simple_parallel_loop
// CHECK-SAME:     ([[ARG0:%.*]]: memref<1x16x12xf32>, [[ARG1:%.*]]: memref<1x16x12xf32>)
// CHECK-DAG:      [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:      [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:      [[C2:%.*]] = arith.constant 2 : index
// CHECK-DAG:      [[C12:%.*]] = arith.constant 12 : index
// CHECK-DAG:      [[C16:%.*]] = arith.constant 16 : index
// CHECK:           scf.parallel ([[IV0:%.*]], [[IV1:%.*]], [[IV2:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[C1]], [[C16]], [[C12]]) step ([[C1]], [[C1]], [[C2]])
// CHECK:             [[LOADED1:%.*]] = memref.load [[ARG0]][[[IV0]], [[IV1]], [[IV2]]] : memref<1x16x12xf32>
// CHECK:             memref.store [[LOADED1]], [[ARG1]][[[IV0]], [[IV1]], [[IV2]]] : memref<1x16x12xf32>
// CHECK:             [[UNR_IV2:%.*]] = affine.apply {{.*}}([[IV2]])
// CHECK:             [[LOADED2:%.*]] = memref.load [[ARG0]][[[IV0]], [[IV1]], [[UNR_IV2]]] : memref<1x16x12xf32>
// CHECK:             memref.store [[LOADED2]], [[ARG1]][[[IV0]], [[IV1]], [[UNR_IV2]]] : memref<1x16x12xf32>

// -----

func.func @negative_unroll_factors_dont_divide_evenly(%src: memref<1x16x12xf32>, %dst: memref<1x16x12xf32>) {
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c1, %c16, %c12) step (%c1, %c1, %c1) {
    %read = memref.load %src[%arg2, %arg3, %arg4] : memref<1x16x12xf32>
    memref.store %read, %dst[%arg2, %arg3, %arg4] : memref<1x16x12xf32>
    scf.reduce
  }
  return
}

// CHECK-UNROLL-BY-3-LABEL:   func @negative_unroll_factors_dont_divide_evenly
// CHECK-UNROLL-BY-3-SAME:     ([[ARG0:%.*]]: memref<1x16x12xf32>, [[ARG1:%.*]]: memref<1x16x12xf32>)
// CHECK-UNROLL-BY-3:           [[C1:%.*]] = arith.constant 1 : index
// CHECK-UNROLL-BY-3:           scf.parallel ([[IV0:%.*]], [[IV1:%.*]], [[IV2:%.*]]) = {{.*}} step ([[C1]], [[C1]], [[C1]])
// CHECK-UNROLL-BY-3:             [[LOADED:%.*]] = memref.load [[ARG0]][[[IV0]], [[IV1]], [[IV2]]] : memref<1x16x12xf32>
// CHECK-UNROLL-BY-3:             memref.store [[LOADED]], [[ARG1]][[[IV0]], [[IV1]], [[IV2]]] : memref<1x16x12xf32>
// CHECK-UNROLL-BY-3-NOT:         affine.apply
// CHECK-UNROLL-BY-3-NOT:         memref.load
// CHECK-UNROLL-BY-3-NOT:         memref.store

// -----

func.func @unroll_outer_nested_parallel_loop(%src: memref<5x16x12x4x4xf32>, %dst: memref<5x16x12x4x4xf32>) {
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %c5 = arith.constant 5 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c5, %c16, %c12) step (%c1, %c1, %c1) {
    scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%c4, %c4) step (%c1, %c1) {
      %0 = affine.apply affine_map<(d0, d1) -> (d0 + (d1 floordiv 4) * 4)>(%arg4, %arg6)
      %1 = affine.apply affine_map<(d0, d1) -> (d0 + (d1 floordiv 4) * 4)>(%arg5, %arg7)
      %subv_in = memref.subview %src[%arg3, %0, %1, 0, 0] [1, 1, 1, 4, 4] [1, 1, 1, 1, 1] : memref<5x16x12x4x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
      %subv_out = memref.subview %dst[%arg3, %0, %1, 0, 0] [1, 1, 1, 4, 4] [1, 1, 1, 1, 1] : memref<5x16x12x4x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
      linalg.elementwise <erf> ins(%subv_in : memref<4x4xf32, strided<[4, 1], offset: ?>>) outs(%subv_out : memref<4x4xf32, strided<[4, 1], offset: ?>>)
      scf.reduce
    }
    scf.reduce
  }
  return
}

// CHECK-UNROLL-BY-3-LABEL:   func @unroll_outer_nested_parallel_loop
// CHECK-LABEL:   func @unroll_outer_nested_parallel_loop
// CHECK-SAME:     ([[ARG0:%.*]]: memref<5x16x12x4x4xf32>, [[ARG1:%.*]]: memref<5x16x12x4x4xf32>)
// CHECK-DAG:      [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:      [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:      [[C2:%.*]] = arith.constant 2 : index
// CHECK-DAG:      [[C4:%.*]] = arith.constant 4 : index
// CHECK-DAG:      [[C5:%.*]] = arith.constant 5 : index
// CHECK-DAG:      [[C12:%.*]] = arith.constant 12 : index
// CHECK-DAG:      [[C16:%.*]] = arith.constant 16 : index
// CHECK:           scf.parallel ([[OUTV0:%.*]], [[OUTV1:%.*]], [[OUTV2:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[C5]], [[C16]], [[C12]]) step ([[C1]], [[C1]], [[C2]])
// CHECK:             scf.parallel ([[INV0:%.*]], [[INV1:%.*]]) = ([[C0]], [[C0]]) to ([[C4]], [[C4]]) step ([[C1]], [[C1]])
// CHECK:               affine.apply {{.*}}([[OUTV1]], [[INV0]])
// CHECK:               affine.apply {{.*}}([[OUTV2]], [[INV1]])
// CHECK:               linalg.elementwise <erf>

// CHECK:             [[UNR_OUTV2:%.*]] = affine.apply {{.*}}([[OUTV2]])
// CHECK:             scf.parallel ([[INV0B:%.*]], [[INV1B:%.*]]) = ([[C0]], [[C0]]) to ([[C4]], [[C4]]) step ([[C1]], [[C1]])
// CHECK:               affine.apply {{.*}}([[OUTV1]], [[INV0B]])
// CHECK:               affine.apply {{.*}}([[UNR_OUTV2]], [[INV1B]])
// CHECK:               linalg.elementwise <erf>

// -----

func.func @negative_unroll_dynamic_parallel_loop(%src: memref<1x16x12xf32>, %dst: memref<1x16x12xf32>, %ub3: index) {
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c1, %c16, %ub3) step (%c1, %c1, %c1) {
    %read = memref.load %src[%arg2, %arg3, %arg4] : memref<1x16x12xf32>
    memref.store %read, %dst[%arg2, %arg3, %arg4] : memref<1x16x12xf32>
    scf.reduce
  }
  return
}

// CHECK-LABEL:   func @negative_unroll_dynamic_parallel_loop
// CHECK-SAME:     ([[ARG0:%.*]]: memref<1x16x12xf32>, [[ARG1:%.*]]: memref<1x16x12xf32>, [[UB3:%.*]]: index)
// CHECK-DAG:       [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:       [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:       [[C16:%.*]] = arith.constant 16 : index
// CHECK:           scf.parallel ([[IV0:%.*]], [[IV1:%.*]], [[IV2:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[C1]], [[C16]], [[UB3]]) step ([[C1]], [[C1]], [[C1]])
// CHECK:             [[LOADED:%.*]] = memref.load [[ARG0]][[[IV0]], [[IV1]], [[IV2]]] : memref<1x16x12xf32>
// CHECK:             memref.store [[LOADED]], [[ARG1]][[[IV0]], [[IV1]], [[IV2]]] : memref<1x16x12xf32>
// CHECK-NOT:         affine.apply
// CHECK-NOT:         memref.load
// CHECK-NOT:         memref.store

// -----

func.func @unroll_inner_nested_parallel_loop(%src: memref<5x16x12x4x4xf32>, %dst: memref<5x16x12x4x4xf32>) {
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %c5 = arith.constant 5 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c5, %c16, %c12) step (%c1, %c1, %c1) {
    scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%c4, %c4) step (%c1, %c1) {
      %0 = affine.apply affine_map<(d0, d1) -> (d0 + (d1 floordiv 4) * 4)>(%arg4, %arg6)
      %1 = affine.apply affine_map<(d0, d1) -> (d0 + (d1 floordiv 4) * 4)>(%arg5, %arg7)
      %subv_in = memref.subview %src[%arg3, %0, %1, 0, 0] [1, 1, 1, 4, 4] [1, 1, 1, 1, 1] : memref<5x16x12x4x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
      %subv_out = memref.subview %dst[%arg3, %0, %1, 0, 0] [1, 1, 1, 4, 4] [1, 1, 1, 1, 1] : memref<5x16x12x4x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
      linalg.elementwise <erf> ins(%subv_in : memref<4x4xf32, strided<[4, 1], offset: ?>>) outs(%subv_out : memref<4x4xf32, strided<[4, 1], offset: ?>>)
      scf.reduce
    }
    scf.reduce
  }
  return
}

// CHECK-LABEL:                func @unroll_inner_nested_parallel_loop
// CHECK-UNROLL-INNER-LABEL:   func @unroll_inner_nested_parallel_loop
// CHECK-UNROLL-INNER-SAME:     ([[ARG0:%.*]]: memref<5x16x12x4x4xf32>, [[ARG1:%.*]]: memref<5x16x12x4x4xf32>)
// CHECK-UNROLL-INNER-DAG:      [[C0:%.*]] = arith.constant 0 : index
// CHECK-UNROLL-INNER-DAG:      [[C1:%.*]] = arith.constant 1 : index
// CHECK-UNROLL-INNER-DAG:      [[C4:%.*]] = arith.constant 4 : index
// CHECK-UNROLL-INNER-DAG:      [[C5:%.*]] = arith.constant 5 : index
// CHECK-UNROLL-INNER-DAG:      [[C12:%.*]] = arith.constant 12 : index
// CHECK-UNROLL-INNER-DAG:      [[C16:%.*]] = arith.constant 16 : index
// CHECK-UNROLL-INNER:          scf.parallel ([[OUTV0:%.*]], [[OUTV1:%.*]], [[OUTV2:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[C5]], [[C16]], [[C12]]) step ([[C1]], [[C1]], [[C1]])
// CHECK-UNROLL-INNER-DAG:        [[C2:%.*]] = arith.constant 2 : index
// CHECK-UNROLL-INNER:            scf.parallel ([[INV0:%.*]], [[INV1:%.*]]) = ([[C0]], [[C0]]) to ([[C4]], [[C4]]) step ([[C1]], [[C2]])
// CHECK-UNROLL-INNER:              affine.apply {{.*}}([[OUTV1]], [[INV0]])
// CHECK-UNROLL-INNER:              affine.apply {{.*}}([[OUTV2]], [[INV1]])
// CHECK-UNROLL-INNER:              linalg.elementwise <erf>

// CHECK-UNROLL-INNER:              [[UNR_INV1:%.*]] = affine.apply {{.*}}([[INV1]])
// CHECK-UNROLL-INNER:              affine.apply {{.*}}([[OUTV1]], [[INV0]])
// CHECK-UNROLL-INNER:              affine.apply {{.*}}([[OUTV2]], [[UNR_INV1]])
// CHECK-UNROLL-INNER:              linalg.elementwise <erf>

// -----

func.func @unroll_parallel_loop_with_reductions(%src: memref<6x4xf32>) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %res:2 = scf.parallel (%i, %j) = (%c0, %c0) to (%c6, %c4) step (%c1, %c1) init (%zero, %one) -> (f32, f32) {
    %read = memref.load %src[%i, %j] : memref<6x4xf32>
    scf.reduce(%read, %read : f32, f32) {
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        scf.reduce.return %sum : f32
    }, {
      ^bb0(%lhs: f32, %rhs: f32):
        %prod = arith.mulf %lhs, %rhs : f32
        scf.reduce.return %prod : f32
    }
  }
  return %res#0, %res#1 : f32, f32
}

// The values of the unrolled iterations are combined with the reduction
// functions.
// CHECK-LABEL:   func @unroll_parallel_loop_with_reductions
// CHECK-SAME:     ([[SRC:%.*]]: memref<6x4xf32>)
// CHECK:           scf.parallel ([[I:%.*]], [[J:%.*]]) =
// CHECK:             [[READ0:%.*]] = memref.load [[SRC]][[[I]], [[J]]] : memref<6x4xf32>
// CHECK:             [[J1:%.*]] = affine.apply {{.*}}([[J]])
// CHECK:             [[READ1:%.*]] = memref.load [[SRC]][[[I]], [[J1]]] : memref<6x4xf32>
// CHECK:             [[SUM:%.*]] = arith.addf [[READ0]], [[READ1]] : f32
// CHECK:             [[PROD:%.*]] = arith.mulf [[READ0]], [[READ1]] : f32
// CHECK:             scf.reduce([[SUM]], [[PROD]] : f32, f32)

// CHECK-UNROLL-BY-3-LABEL:   func @unroll_parallel_loop_with_reductions
// CHECK-UNROLL-BY-3-SAME:     ([[SRC:%.*]]: memref<6x4xf32>)
// CHECK-UNROLL-BY-3:           scf.parallel ([[I:%.*]], [[J:%.*]]) =
// CHECK-UNROLL-BY-3:             [[READ0:%.*]] = memref.load [[SRC]][[[I]], [[J]]] : memref<6x4xf32>
// CHECK-UNROLL-BY-3:             [[I1:%.*]] = affine.apply {{.*}}([[I]])
// CHECK-UNROLL-BY-3:             [[READ1:%.*]] = memref.load [[SRC]][[[I1]], [[J]]] : memref<6x4xf32>
// CHECK-UNROLL-BY-3:             [[I2:%.*]] = affine.apply {{.*}}([[I]])
// CHECK-UNROLL-BY-3:             [[READ2:%.*]] = memref.load [[SRC]][[[I2]], [[J]]] : memref<6x4xf32>
// CHECK-UNROLL-BY-3:             [[SUM1:%.*]] = arith.addf [[READ0]], [[READ1]] : f32
// CHECK-UNROLL-BY-3:             [[SUM2:%.*]] = arith.addf [[SUM1]], [[READ2]] : f32
// CHECK-UNROLL-BY-3:             [[PROD1:%.*]] = arith.mulf [[READ0]], [[READ1]] : f32
// CHECK-UNROLL-BY-3:             [[PROD2:%.*]] = arith.mulf [[PROD1]], [[READ2]] : f32
// CHECK-UNROLL-BY-3:             scf.reduce([[SUM2]], [[PROD2]] : f32, f32)

// Unrolling a second dimension combines the values combined for the first one.
// CHECK-UNROLL-3-2-LABEL:   func @unroll_parallel_loop_with_reductions
// CHECK-UNROLL-3-2:           scf.parallel ([[I:%.*]], [[J:%.*]]) =
// CHECK-UNROLL-3-2-COUNT-3:     memref.load {{.*}}, [[J]]]
// CHECK-UNROLL-3-2:             [[SUM1:%.*]] = arith.addf
// CHECK-UNROLL-3-2:             [[SUM2:%.*]] = arith.addf [[SUM1]],
// CHECK-UNROLL-3-2:             [[PROD1:%.*]] = arith.mulf
// CHECK-UNROLL-3-2:             [[PROD2:%.*]] = arith.mulf [[PROD1]],
// CHECK-UNROLL-3-2:             [[J1:%.*]] = affine.apply {{.*}}([[J]])
// CHECK-UNROLL-3-2-COUNT-3:     memref.load {{.*}}, [[J1]]]
// CHECK-UNROLL-3-2:             [[SUM3:%.*]] = arith.addf
// CHECK-UNROLL-3-2:             [[SUM4:%.*]] = arith.addf [[SUM3]],
// CHECK-UNROLL-3-2:             [[PROD3:%.*]] = arith.mulf
// CHECK-UNROLL-3-2:             [[PROD4:%.*]] = arith.mulf [[PROD3]],
// CHECK-UNROLL-3-2:             [[SUM:%.*]] = arith.addf [[SUM2]], [[SUM4]] : f32
// CHECK-UNROLL-3-2:             [[PROD:%.*]] = arith.mulf [[PROD2]], [[PROD4]] : f32
// CHECK-UNROLL-3-2:             scf.reduce([[SUM]], [[PROD]] : f32, f32)

// -----

func.func @unroll_parallel_loop_reducing_iv_and_invariant(%dst: memref<6x4xindex>, %val: index) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %res:2 = scf.parallel (%i, %j) = (%c0, %c0) to (%c6, %c4) step (%c1, %c1) init (%c0, %c0) -> (index, index) {
    memref.store %j, %dst[%i, %j] : memref<6x4xindex>
    scf.reduce(%j, %val : index, index) {
      ^bb0(%lhs: index, %rhs: index):
        %sum = arith.addi %lhs, %rhs : index
        scf.reduce.return %sum : index
    }, {
      ^bb0(%lhs: index, %rhs: index):
        %sum = arith.addi %lhs, %rhs : index
        scf.reduce.return %sum : index
    }
  }
  return %res#0, %res#1 : index, index
}

// CHECK-LABEL:   func @unroll_parallel_loop_reducing_iv_and_invariant
// CHECK-SAME:     ([[DST:%.*]]: memref<6x4xindex>, [[VAL:%.*]]: index)
// CHECK:           scf.parallel ([[I:%.*]], [[J:%.*]]) =
// CHECK:             memref.store [[J]], [[DST]][[[I]], [[J]]] : memref<6x4xindex>
// CHECK:             [[J1:%.*]] = affine.apply {{.*}}([[J]])
// CHECK:             memref.store [[J1]], [[DST]][[[I]], [[J1]]] : memref<6x4xindex>
// CHECK:             [[IV_SUM:%.*]] = arith.addi [[J]], [[J1]] : index
// CHECK:             [[VAL_SUM:%.*]] = arith.addi [[VAL]], [[VAL]] : index
// CHECK:             scf.reduce([[IV_SUM]], [[VAL_SUM]] : index, index)

// -----

func.func @unroll_parallel_loop_reduction_using_outer_value(%src: memref<6x4xf32>, %limit: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %zero = arith.constant 0.0 : f32
  %res = scf.parallel (%i, %j) = (%c0, %c0) to (%c6, %c4) step (%c1, %c1) init (%zero) -> f32 {
    %read = memref.load %src[%i, %j] : memref<6x4xf32>
    scf.reduce(%read : f32) {
      ^bb0(%lhs: f32, %rhs: f32):
        %max = arith.maximumf %lhs, %rhs : f32
        %clamped = arith.minimumf %max, %limit : f32
        scf.reduce.return %clamped : f32
    }
  }
  return %res : f32
}

// A reduction function may use values defined outside of the loop.
// CHECK-LABEL:   func @unroll_parallel_loop_reduction_using_outer_value
// CHECK-SAME:     ([[SRC:%.*]]: memref<6x4xf32>, [[LIMIT:%.*]]: f32)
// CHECK:           scf.parallel ([[I:%.*]], [[J:%.*]]) =
// CHECK:             [[READ0:%.*]] = memref.load [[SRC]][[[I]], [[J]]] : memref<6x4xf32>
// CHECK:             [[J1:%.*]] = affine.apply {{.*}}([[J]])
// CHECK:             [[READ1:%.*]] = memref.load [[SRC]][[[I]], [[J1]]] : memref<6x4xf32>
// CHECK:             [[MAX:%.*]] = arith.maximumf [[READ0]], [[READ1]] : f32
// CHECK:             [[CLAMPED:%.*]] = arith.minimumf [[MAX]], [[LIMIT]] : f32
// CHECK:             scf.reduce([[CLAMPED]] : f32)

// -----

func.func @negative_unroll_reduction_using_loop_value(%src: memref<6x4xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %zero = arith.constant 0.0 : f32
  %res = scf.parallel (%i, %j) = (%c0, %c0) to (%c6, %c4) step (%c1, %c1) init (%zero) -> f32 {
    %read = memref.load %src[%i, %j] : memref<6x4xf32>
    scf.reduce(%read : f32) {
      ^bb0(%lhs: f32, %rhs: f32):
        %scaled = arith.mulf %rhs, %read : f32
        %sum = arith.addf %lhs, %scaled : f32
        scf.reduce.return %sum : f32
    }
  }
  return %res : f32
}

// A reduction function using a value defined in the loop is not unrolled.
// CHECK-LABEL:   func @negative_unroll_reduction_using_loop_value
// CHECK:           scf.parallel ({{.*}}) = {{.*}} step ([[C1:%.*]], [[C1]])
// CHECK-NOT:         affine.apply
// CHECK:             scf.reduce
