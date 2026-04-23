// RUN: mlir-opt %s -test-parallel-loop-unrolling='unroll-factors=1,2' -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-parallel-loop-unrolling='unroll-factors=1,2 loop-depth=1' -split-input-file | FileCheck %s --check-prefix CHECK-UNROLL-INNER
// RUN: mlir-opt %s -test-parallel-loop-unrolling='unroll-factors=3,1' -split-input-file | FileCheck %s --check-prefix CHECK-UNROLL-BY-3

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
      linalg.erf ins(%subv_in : memref<4x4xf32, strided<[4, 1], offset: ?>>) outs(%subv_out : memref<4x4xf32, strided<[4, 1], offset: ?>>)
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
// CHECK:               linalg.erf

// CHECK:             [[UNR_OUTV2:%.*]] = affine.apply {{.*}}([[OUTV2]])
// CHECK:             scf.parallel ([[INV0B:%.*]], [[INV1B:%.*]]) = ([[C0]], [[C0]]) to ([[C4]], [[C4]]) step ([[C1]], [[C1]])
// CHECK:               affine.apply {{.*}}([[OUTV1]], [[INV0B]])
// CHECK:               affine.apply {{.*}}([[UNR_OUTV2]], [[INV1B]])
// CHECK:               linalg.erf

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
      linalg.erf ins(%subv_in : memref<4x4xf32, strided<[4, 1], offset: ?>>) outs(%subv_out : memref<4x4xf32, strided<[4, 1], offset: ?>>)
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
// CHECK-UNROLL-INNER:              linalg.erf

// CHECK-UNROLL-INNER:              [[UNR_INV1:%.*]] = affine.apply {{.*}}([[INV1]])
// CHECK-UNROLL-INNER:              affine.apply {{.*}}([[OUTV1]], [[INV0]])
// CHECK-UNROLL-INNER:              affine.apply {{.*}}([[OUTV2]], [[UNR_INV1]])
// CHECK-UNROLL-INNER:              linalg.erf
