// RUN: mlir-opt %s -convert-linalg-to-loops="interchange-vector=4,0,3,1,2" -split-input-file | FileCheck --check-prefix=LOOP %s
// RUN: mlir-opt %s -convert-linalg-to-parallel-loops="interchange-vector=4,0,3,1,2" -split-input-file | FileCheck --check-prefix=PARALLEL %s
// RUN: mlir-opt %s -convert-linalg-to-affine-loops="interchange-vector=4,0,3,1,2" -split-input-file | FileCheck --check-prefix=AFFINE %s

func @copy(%input: memref<1x2x3x4x5xf32>, %output: memref<1x2x3x4x5xf32>) {
  linalg.copy(%input, %output): memref<1x2x3x4x5xf32>, memref<1x2x3x4x5xf32>
  return
}

// LOOP: scf.for %{{.*}} = %c0 to %c5 step %c1
// LOOP:   scf.for %{{.*}} = %c0 to %c1 step %c1
// LOOP:     scf.for %{{.*}} = %c0 to %c4 step %c1
// LOOP:       scf.for %{{.*}} = %c0 to %c2 step %c1
// LOOP:         scf.for %{{.*}} = %c0 to %c3 step %c1

// PARALLEL: 			scf.parallel
// PARALLEL-SAME:   to (%c5, %c1, %c4, %c2, %c3)

// AFFINE: affine.for %{{.*}} = 0 to 5
// AFFINE:   affine.for %{{.*}} = 0 to 1
// AFFINE:     affine.for %{{.*}} = 0 to 4
// AFFINE:       affine.for %{{.*}} = 0 to 2
// AFFINE:         affine.for %{{.*}} = 0 to 3

// -----

func @index_op(%arg0: memref<4x8xindex>) {
  linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]}
  outs(%arg0 : memref<4x8xindex>) {
  ^bb0(%arg1: index):   // no predecessors
    %0 = linalg.index 1 : index
    linalg.yield %0 : index
  }
  return
}
// LOOP-LABEL: @index_op
//      LOOP:   linalg.generic

// PARALLEL-LABEL: @index_op
//      PARALLEL:   linalg.generic

// AFFINE-LABEL: @index_op
//      AFFINE:   linalg.generic
