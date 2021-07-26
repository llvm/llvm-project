// RUN: mlir-opt %s -test-linalg-transform-patterns=test-affine-min-scf-canonicalization-patterns | FileCheck %s

// CHECK-LABEL: scf_for
func @scf_for(%A : memref<i64>, %step : index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c7 = constant 7 : index
  %c4 = constant 4 : index
  %c16 = constant 16 : index
  %c1024 = constant 1024 : index

  // CHECK:      %[[C2:.*]] = constant 2 : i64
  //      CHECK: scf.for
  // CHECK-NEXT:   memref.store %[[C2]], %{{.*}}[] : memref<i64>
  scf.for %i = %c0 to %c4 step %c2 {
    %1 = affine.min affine_map<(d0, d1)[] -> (2, d1 - d0)> (%i, %c4)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }

  //      CHECK: scf.for
  // CHECK-NEXT:   memref.store %[[C2]], %{{.*}}[] : memref<i64>
  scf.for %i = %c1 to %c7 step %c2 {
    %1 = affine.min affine_map<(d0)[s0] -> (s0 - d0, 2)> (%i)[%c7]
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }

  // This should not canonicalize because: 4 - %i may take the value 1 < 2.
  //     CHECK:   scf.for
  //     CHECK:     affine.min
  //     CHECK:     index_cast
  scf.for %i = %c1 to %c4 step %c2 {
    %1 = affine.min affine_map<(d0)[s0] -> (2, s0 - d0)> (%i)[%c4]
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }

  // This should not canonicalize because: 16 - %i may take the value 15 < 1024.
  //     CHECK:   scf.for
  //     CHECK:     affine.min
  //     CHECK:     index_cast
  scf.for %i = %c1 to %c16 step %c1024 {
    %1 = affine.min affine_map<(d0) -> (1024, 16 - d0)> (%i)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }

  // This example should simplify but affine_map is currently missing
  // semi-affine canonicalizations: `((s0 * 42 - 1) floordiv s0) * s0`
  // should evaluate to 41 * s0.
  // Note that this may require positivity assumptions on `s0`.
  // Revisit when support is added.
  // CHECK: scf.for
  // CHECK:   affine.min
  // CHECK:   index_cast
  %ub = affine.apply affine_map<(d0) -> (42 * d0)> (%step)
  scf.for %i = %c0 to %ub step %step {
    %1 = affine.min affine_map<(d0, d1, d2) -> (d0, d1 - d2)> (%step, %ub, %i)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }

  // This example should simplify but affine_map is currently missing
  // semi-affine canonicalizations.
  // This example should simplify but affine_map is currently missing
  // semi-affine canonicalizations: ` -(((s0 * s0 - 1) floordiv s0) * s0)`
  // should evaluate to (s0 - 1) * s0.
  // Note that this may require positivity assumptions on `s0`.
  // Revisit when support is added.
  // CHECK: scf.for
  // CHECK:   affine.min
  // CHECK:   index_cast
  %ub2 = affine.apply affine_map<(d0)[s0] -> (s0 * d0)> (%step)[%step]
  scf.for %i = %c0 to %ub2 step %step {
    %1 = affine.min affine_map<(d0, d1, d2) -> (d0, d2 - d1)> (%step, %i, %ub2)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }

  return
}

// CHECK-LABEL: scf_parallel
func @scf_parallel(%A : memref<i64>, %step : index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c7 = constant 7 : index
  %c4 = constant 4 : index

  // CHECK:   %[[C2:.*]] = constant 2 : i64
  // CHECK: scf.parallel
  // CHECK-NEXT:   memref.store %[[C2]], %{{.*}}[] : memref<i64>
  scf.parallel (%i) = (%c0) to (%c4) step (%c2) {
    %1 = affine.min affine_map<(d0, d1)[] -> (2, d1 - d0)> (%i, %c4)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }

  // CHECK: scf.parallel
  // CHECK-NEXT:   memref.store %[[C2]], %{{.*}}[] : memref<i64>
  scf.parallel (%i) = (%c1) to (%c7) step (%c2) {
    %1 = affine.min affine_map<(d0)[s0] -> (2, s0 - d0)> (%i)[%c7]
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }

  // This example should simplify but affine_map is currently missing
  // semi-affine canonicalizations.
  // This affine map does not currently evaluate to (0, 0):
  //   (d0)[s0] -> (s0 mod s0, (-((d0 floordiv s0) * s0) + s0 * 42) mod s0)
  // TODO: Revisit when support is added.
  // CHECK: scf.parallel
  // CHECK:   affine.min
  // CHECK:   index_cast
  %ub = affine.apply affine_map<(d0) -> (42 * d0)> (%step)
  scf.parallel (%i) = (%c0) to (%ub) step (%step) {
    %1 = affine.min affine_map<(d0, d1, d2) -> (d0, d2 - d1)> (%step, %i, %ub)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }

  // This example should simplify but affine_map is currently missing
  // semi-affine canonicalizations.
  // This affine map does not currently evaluate to (0, 0):
  //   (d0)[s0] -> (s0 mod s0, (-((d0 floordiv s0) * s0) + s0 * s0) mod s0)
  // TODO: Revisit when support is added.
  // CHECK: scf.parallel
  // CHECK:   affine.min
  // CHECK:   index_cast
  %ub2 = affine.apply affine_map<(d0)[s0] -> (s0 * d0)> (%step)[%step]
  scf.parallel (%i) = (%c0) to (%ub2) step (%step) {
    %1 = affine.min affine_map<(d0, d1, d2) -> (d0, d2 - d1)> (%step, %i, %ub2)
    %2 = index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }

  return
}
