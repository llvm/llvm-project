// RUN: mlir-opt %s -test-transform-dialect-interpreter -canonicalize -split-input-file | FileCheck %s -check-prefix=TILE-10n25

func.func @indexed_vector(%arg0: memref<50xindex>) {
  linalg.generic {indexing_maps = [affine_map<(i) -> (i)>],
                  iterator_types = ["parallel"]}
     outs(%arg0 : memref<50xindex>) {
    ^bb0(%a: index):
      %i = linalg.index 0 : index
      linalg.yield %i : index
  }
  return
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1, %loop:2 = transform.structured.tile %0 [10, 25]
}

// TILE-10n25-DAG: [[$MAP:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0 + d1)>
// TILE-10n25-LABEL: func @indexed_vector
// TILE-10n25: %[[C10:.*]] = arith.constant 10 : index
// TILE-10n25: scf.for %[[J:.*]] = {{.*}} step %[[C10]]
// TILE-10n25:   linalg.generic
// TILE-10n25:     %[[I:.*]] = linalg.index 0 : index
// TILE-10n25:     %[[NEW_I:.*]] = affine.apply [[$MAP]](%[[I]], %[[J]])
// TILE-10n25:     linalg.yield %[[NEW_I]] : index

// -----

func.func @indexed_matrix(%arg0: memref<50x50xindex>) {
  linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>],
                  iterator_types = ["parallel", "parallel"]}
    outs(%arg0 : memref<50x50xindex>) {
    ^bb0(%a: index):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %sum = arith.addi %i, %j : index
      linalg.yield %sum : index
  }
  return
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1, %loop:2 = transform.structured.tile %0 [10, 25]
}

// TILE-10n25-DAG: [[$MAP:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0 + d1)>
// TILE-10n25-LABEL: func @indexed_matrix
// TILE-10n25-DAG: %[[C25:.*]] = arith.constant 25 : index
// TILE-10n25-DAG: %[[C10:.*]] = arith.constant 10 : index
// TILE-10n25: scf.for %[[K:.*]] = {{.*}} step %[[C10]]
// TILE-10n25:   scf.for %[[L:.*]] = {{.*}} step %[[C25]]
// TILE-10n25:     linalg.generic
// TILE-10n25:       %[[I:.*]] = linalg.index 0 : index
// TILE-10n25:       %[[NEW_I:.*]] = affine.apply [[$MAP]](%[[I]], %[[K]])
// TILE-10n25:       %[[J:.*]] = linalg.index 1 : index
// TILE-10n25:       %[[NEW_J:.*]] = affine.apply [[$MAP]](%[[J]], %[[L]])
// TILE-10n25:       %[[SUM:.*]] = arith.addi %[[NEW_I]], %[[NEW_J]] : index
// TILE-10n25:       linalg.yield %[[SUM]] : index
