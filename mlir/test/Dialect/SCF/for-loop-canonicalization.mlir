// RUN: mlir-opt %s -for-loop-canonicalization -split-input-file | FileCheck %s

// CHECK-LABEL: func @scf_for_canonicalize_min
//       CHECK:   %[[C2:.*]] = constant 2 : i64
//       CHECK:   scf.for
//       CHECK:     memref.store %[[C2]], %{{.*}}[] : memref<i64>
func @scf_for_canonicalize_min(%A : memref<i64>) {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index

  scf.for %i = %c0 to %c4 step %c2 {
    %1 = affine.min affine_map<(d0, d1)[] -> (2, d1 - d0)> (%i, %c4)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}

// -----

// CHECK-LABEL: func @scf_for_canonicalize_max
//       CHECK:   %[[Cneg2:.*]] = constant -2 : i64
//       CHECK:   scf.for
//       CHECK:     memref.store %[[Cneg2]], %{{.*}}[] : memref<i64>
func @scf_for_canonicalize_max(%A : memref<i64>) {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index

  scf.for %i = %c0 to %c4 step %c2 {
    %1 = affine.max affine_map<(d0, d1)[] -> (-2, -(d1 - d0))> (%i, %c4)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}

// -----

// CHECK-LABEL: func @scf_for_max_not_canonicalizable
//       CHECK:   scf.for
//       CHECK:     affine.max
//       CHECK:     index_cast
func @scf_for_max_not_canonicalizable(%A : memref<i64>) {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index

  scf.for %i = %c0 to %c4 step %c2 {
    %1 = affine.max affine_map<(d0, d1)[] -> (-2, -(d1 - d0))> (%i, %c3)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}

// -----

// CHECK-LABEL: func @scf_for_loop_nest_canonicalize_min
//       CHECK:   %[[C5:.*]] = constant 5 : i64
//       CHECK:   scf.for
//       CHECK:     scf.for
//       CHECK:       memref.store %[[C5]], %{{.*}}[] : memref<i64>
func @scf_for_loop_nest_canonicalize_min(%A : memref<i64>) {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c6 = constant 6 : index

  scf.for %i = %c0 to %c4 step %c2 {
    scf.for %j = %c0 to %c6 step %c3 {
      %1 = affine.min affine_map<(d0, d1, d2, d3)[] -> (5, d1 + d3 - d0 - d2)> (%i, %c4, %j, %c6)
      %2 = index_cast %1: index to i64
      memref.store %2, %A[]: memref<i64>
    }
  }
  return
}

// -----

// CHECK-LABEL: func @scf_for_not_canonicalizable_1
//       CHECK:   scf.for
//       CHECK:     affine.min
//       CHECK:     index_cast
func @scf_for_not_canonicalizable_1(%A : memref<i64>) {
  // This should not canonicalize because: 4 - %i may take the value 1 < 2.
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index

  scf.for %i = %c1 to %c4 step %c2 {
    %1 = affine.min affine_map<(d0)[s0] -> (2, s0 - d0)> (%i)[%c4]
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}

// -----

// CHECK-LABEL: func @scf_for_canonicalize_partly
//       CHECK:   scf.for
//       CHECK:     affine.apply
//       CHECK:     index_cast
func @scf_for_canonicalize_partly(%A : memref<i64>) {
  // This should canonicalize only partly: 256 - %i <= 256.
  %c1 = constant 1 : index
  %c16 = constant 16 : index
  %c256 = constant 256 : index

  scf.for %i = %c1 to %c256 step %c16 {
    %1 = affine.min affine_map<(d0) -> (256, 256 - d0)> (%i)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}

// -----

// CHECK-LABEL: func @scf_for_not_canonicalizable_2
//       CHECK: scf.for
//       CHECK:   affine.min
//       CHECK:   index_cast
func @scf_for_not_canonicalizable_2(%A : memref<i64>, %step : index) {
  // This example should simplify but affine_map is currently missing
  // semi-affine canonicalizations: `((s0 * 42 - 1) floordiv s0) * s0`
  // should evaluate to 41 * s0.
  // Note that this may require positivity assumptions on `s0`.
  // Revisit when support is added.
  %c0 = constant 0 : index

  %ub = affine.apply affine_map<(d0) -> (42 * d0)> (%step)
  scf.for %i = %c0 to %ub step %step {
    %1 = affine.min affine_map<(d0, d1, d2) -> (d0, d1 - d2)> (%step, %ub, %i)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}

// -----

// CHECK-LABEL: func @scf_for_not_canonicalizable_3
//       CHECK: scf.for
//       CHECK:   affine.min
//       CHECK:   index_cast
func @scf_for_not_canonicalizable_3(%A : memref<i64>, %step : index) {
  // This example should simplify but affine_map is currently missing
  // semi-affine canonicalizations: `-(((s0 * s0 - 1) floordiv s0) * s0)`
  // should evaluate to (s0 - 1) * s0.
  // Note that this may require positivity assumptions on `s0`.
  // Revisit when support is added.
  %c0 = constant 0 : index

  %ub2 = affine.apply affine_map<(d0)[s0] -> (s0 * d0)> (%step)[%step]
  scf.for %i = %c0 to %ub2 step %step {
    %1 = affine.min affine_map<(d0, d1, d2) -> (d0, d2 - d1)> (%step, %i, %ub2)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}

// -----

// CHECK-LABEL: func @scf_for_invalid_loop
//       CHECK: scf.for
//       CHECK:   affine.min
//       CHECK:   index_cast
func @scf_for_invalid_loop(%A : memref<i64>, %step : index) {
  // This is an invalid loop. It should not be touched by the canonicalization
  // pattern.
  %c1 = constant 1 : index
  %c7 = constant 7 : index
  %c256 = constant 256 : index

  scf.for %i = %c256 to %c1 step %c1 {
    %1 = affine.min affine_map<(d0)[s0] -> (s0 + d0, 0)> (%i)[%c7]
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}

// -----

// CHECK-LABEL: func @scf_parallel_canonicalize_min_1
//       CHECK:   %[[C2:.*]] = constant 2 : i64
//       CHECK:   scf.parallel
//  CHECK-NEXT:     memref.store %[[C2]], %{{.*}}[] : memref<i64>
func @scf_parallel_canonicalize_min_1(%A : memref<i64>) {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index

  scf.parallel (%i) = (%c0) to (%c4) step (%c2) {
    %1 = affine.min affine_map<(d0, d1)[] -> (2, d1 - d0)> (%i, %c4)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}

// -----

// CHECK-LABEL: func @scf_parallel_canonicalize_min_2
//       CHECK:   %[[C2:.*]] = constant 2 : i64
//       CHECK:   scf.parallel
//  CHECK-NEXT:     memref.store %[[C2]], %{{.*}}[] : memref<i64>
func @scf_parallel_canonicalize_min_2(%A : memref<i64>) {
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c7 = constant 7 : index

  scf.parallel (%i) = (%c1) to (%c7) step (%c2) {
    %1 = affine.min affine_map<(d0)[s0] -> (2, s0 - d0)> (%i)[%c7]
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}

// -----

// CHECK-LABEL: func @tensor_dim_of_iter_arg(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>
//       CHECK:   scf.for
//       CHECK:     tensor.dim %[[t]]
func @tensor_dim_of_iter_arg(%t : tensor<?x?xf32>) -> index {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %0, %1 = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg0 = %t, %arg1 = %c0)
      -> (tensor<?x?xf32>, index) {
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    scf.yield %arg0, %dim : tensor<?x?xf32>, index
  }
  return %1 : index
}
