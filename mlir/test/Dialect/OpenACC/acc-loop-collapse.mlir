// RUN: mlir-opt %s -acc-loop-collapse -split-input-file | FileCheck %s

// A two-dimensional parallel loop in a compute region becomes
// one-dimensional. The original induction variables are recovered with
// remsi/divsi, and the parallel dimensions and the collapse depth are kept on
// the new loop.

// CHECK-LABEL: func.func @collapse_2d
// CHECK:         acc.compute_region
// CHECK-NOT:     scf.parallel (%{{[^)]*}},
// CHECK:         scf.parallel (%[[IV:arg[0-9]+]])
// CHECK-NEXT:      %[[REM:.*]] = arith.remsi %[[IV]], %c10
// CHECK-NEXT:      %[[DIV:.*]] = arith.divsi %[[IV]], %c10
// CHECK-NEXT:      %[[LOAD:.*]] = memref.load %{{.*}}[%[[DIV]], %[[REM]]]
// CHECK-NEXT:      memref.store %[[LOAD]], %{{.*}}[%[[DIV]], %[[REM]]]
// CHECK-NEXT:      scf.reduce
// CHECK-NEXT:    } {acc.collapse_count = 2 : i64, acc.par_dims = #acc<par_dims[sequential]>}
func.func @collapse_2d(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) {
  acc.compute_region ins(%in = %arg0, %out = %arg1)
      : (memref<10x10xf32>, memref<10x10xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    scf.parallel (%i, %j) = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
      %0 = memref.load %in[%i, %j] : memref<10x10xf32>
      memref.store %0, %out[%i, %j] : memref<10x10xf32>
    } {acc.par_dims = #acc<par_dims[sequential]>}
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

// -----

// A loop with a reduction keeps its init value, its reduction body and its
// result.

// CHECK-LABEL: func.func @collapse_reduction
// CHECK-NOT:     scf.parallel (%{{[^)]*}},
// CHECK:         %[[RES:.*]] = scf.parallel (%[[IV:arg[0-9]+]]) = (%c0{{.*}}) to (%{{.*}}) step (%c1{{.*}}) init (%{{.*}}) -> f32
// CHECK:           %[[LOAD:.*]] = memref.load
// CHECK-NEXT:      scf.reduce(%[[LOAD]] : f32) {
// CHECK:             arith.addf
// CHECK:             scf.reduce.return
// CHECK:         } {acc.collapse_count = 2 : i64, acc.par_dims = #acc<par_dims[sequential]>}
// CHECK:         memref.store %[[RES]], %{{.*}}[]
func.func @collapse_reduction(%arg0: memref<10x10xf32>, %arg1: memref<f32>) {
  acc.compute_region ins(%in = %arg0, %out = %arg1)
      : (memref<10x10xf32>, memref<f32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %0 = scf.parallel (%i, %j) = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) init (%cst) -> f32 {
      %1 = memref.load %in[%i, %j] : memref<10x10xf32>
      scf.reduce(%1 : f32) {
      ^bb0(%a: f32, %b: f32):
        %2 = arith.addf %a, %b : f32
        scf.reduce.return %2 : f32
      }
    } {acc.par_dims = #acc<par_dims[sequential]>}
    memref.store %0, %out[] : memref<f32>
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

// -----

// Nested multi-dimensional loops collapse independently. Each loop keeps its
// own parallel dimensions.

// CHECK-LABEL: func.func @collapse_nested
// CHECK-NOT:     scf.parallel (%{{[^)]*}},
// CHECK:         scf.parallel (%[[OUTER:arg[0-9]+]])
// CHECK-NEXT:      %[[OREM:.*]] = arith.remsi %[[OUTER]], %c10
// CHECK-NEXT:      %[[ODIV:.*]] = arith.divsi %[[OUTER]], %c10
// CHECK-NOT:       scf.parallel (%{{[^)]*}},
// CHECK:           scf.parallel (%[[INNER:arg[0-9]+]])
// CHECK-NEXT:        %[[IREM:.*]] = arith.remsi %[[INNER]], %c4
// CHECK-NEXT:        %[[IDIV:.*]] = arith.divsi %[[INNER]], %c4
// CHECK-NEXT:        memref.store {{.*}}, %{{.*}}[%[[ODIV]], %[[OREM]], %[[IDIV]], %[[IREM]]]
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      } {acc.collapse_count = 2 : i64, acc.par_dims = #acc<par_dims[thread_x]>}
// CHECK-NEXT:      scf.reduce
// CHECK-NEXT:    } {acc.collapse_count = 2 : i64, acc.par_dims = #acc<par_dims[block_x]>}
func.func @collapse_nested(%arg0: memref<10x10x8x4xi32>) {
  acc.compute_region ins(%a = %arg0) : (memref<10x10x8x4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c10 = arith.constant 10 : index
    %c1_i32 = arith.constant 1 : i32
    scf.parallel (%i, %j) = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
      scf.parallel (%k, %l) = (%c0, %c0) to (%c8, %c4) step (%c1, %c1) {
        memref.store %c1_i32, %a[%i, %j, %k, %l] : memref<10x10x8x4xi32>
      } {acc.par_dims = #acc<par_dims[thread_x]>}
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

// -----

// Every dimension of a three-dimensional loop is collapsed into one.

// CHECK-LABEL: func.func @collapse_3d
// CHECK:         acc.compute_region
// CHECK-NOT:     scf.parallel (%{{[^)]*}},
// CHECK:         scf.parallel (%[[IV:arg[0-9]+]])
// CHECK:           arith.remsi %[[IV]]
// CHECK:           arith.divsi %[[IV]]
// CHECK:           arith.remsi
// CHECK:           arith.divsi
// CHECK:         } {acc.collapse_count = 3 : i64, acc.par_dims = #acc<par_dims[block_x]>}
func.func @collapse_3d(%arg0: memref<4x5x6xi32>) {
  acc.compute_region ins(%a = %arg0) : (memref<4x5x6xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7_i32 = arith.constant 7 : i32
    scf.parallel (%i, %j, %k) = (%c0, %c0, %c0) to (%c4, %c5, %c6)
        step (%c1, %c1, %c1) {
      memref.store %c7_i32, %a[%i, %j, %k] : memref<4x5x6xi32>
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

// -----

// Dynamic bounds, a non-unit step and a non-zero lower bound are normalized:
// the trip count is computed first, and the original induction variable is
// rebuilt from the collapsed one.

// CHECK-LABEL: func.func @collapse_dynamic
// CHECK:         acc.compute_region
// CHECK:           affine.apply
// CHECK:           %[[PRODUCT:[0-9]+]] = arith.muli %{{[0-9]+}}, %arg{{[0-9]+}} : index
// CHECK:           scf.parallel (%[[IV:arg[0-9]+]]) = (%c0{{.*}}) to (%[[PRODUCT]])
// CHECK:             arith.remsi %[[IV]]
// CHECK:             arith.divsi %[[IV]]
// CHECK:             affine.apply
// CHECK:           } {acc.collapse_count = 2 : i64, acc.par_dims = #acc<par_dims[block_x]>}
func.func @collapse_dynamic(%arg0: memref<?x?xf32>, %lb: index, %ub0: index,
                            %ub1: index, %step: index) {
  acc.compute_region ins(%a = %arg0, %lb_in = %lb, %ub0_in = %ub0,
                         %ub1_in = %ub1, %step_in = %step)
      : (memref<?x?xf32>, index, index, index, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.parallel (%i, %j) = (%lb_in, %c0) to (%ub0_in, %ub1_in)
        step (%step_in, %c1) {
      memref.store %cst, %a[%i, %j] : memref<?x?xf32>
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

// -----

// All discardable attributes of the original loop are kept on the collapsed
// loop, not only the parallel dimensions.

// CHECK-LABEL: func.func @collapse_keeps_attributes
// CHECK:         scf.parallel (%{{.*}}) =
// CHECK:         } {acc.chunk_size = #acc.chunk_size<4>, acc.collapse_count = 2 : i64, acc.gpu_block_redundant = #acc.gpu_block_redundant, acc.par_dims = #acc<par_dims[thread_x]>, test.marker = 1 : i32}
func.func @collapse_keeps_attributes(%arg0: memref<10x10xf32>) {
  acc.compute_region ins(%a = %arg0) : (memref<10x10xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.parallel (%i, %j) = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
      memref.store %cst, %a[%i, %j] : memref<10x10xf32>
    } {acc.chunk_size = #acc.chunk_size<4>,
       acc.gpu_block_redundant = #acc.gpu_block_redundant,
       acc.par_dims = #acc<par_dims[thread_x]>, test.marker = 1 : i32}
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

// -----

// A multi-dimensional loop outside an acc.compute_region is not collapsed.

// CHECK-LABEL: func.func @outside_compute_region
// CHECK:         scf.parallel (%{{.*}}, %{{.*}}) = (%c0, %c0) to (%c10, %c10) step (%c1, %c1)
// CHECK-NOT:     acc.collapse_count
// CHECK:         } {acc.par_dims = #acc<par_dims[block_x]>}
func.func @outside_compute_region(%arg0: memref<10x10xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 0.000000e+00 : f32
  scf.parallel (%i, %j) = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
    memref.store %cst, %arg0[%i, %j] : memref<10x10xf32>
  } {acc.par_dims = #acc<par_dims[block_x]>}
  return
}

// -----

// A loop that is already one-dimensional is left untouched.

// CHECK-LABEL: func.func @one_dimensional
// CHECK:         scf.parallel (%{{.*}}) = (%c0) to (%c10) step (%c1)
// CHECK-NOT:     acc.collapse_count
// CHECK:         } {acc.par_dims = #acc<par_dims[block_x]>}
func.func @one_dimensional(%arg0: memref<10xf32>) {
  acc.compute_region ins(%a = %arg0) : (memref<10xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.parallel (%i) = (%c0) to (%c10) step (%c1) {
      memref.store %cst, %a[%i] : memref<10xf32>
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}
