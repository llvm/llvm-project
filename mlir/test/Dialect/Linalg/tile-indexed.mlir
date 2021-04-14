// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=10,25" -split-input-file | FileCheck %s -check-prefix=TILE-10n25
// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=25,0" -split-input-file | FileCheck %s -check-prefix=TILE-25n0
// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=0,25" -split-input-file | FileCheck %s -check-prefix=TILE-0n25

func @indexed_vector(%arg0: memref<50xindex>) {
  linalg.generic {indexing_maps = [affine_map<(i) -> (i)>],
                  iterator_types = ["parallel"]}
     outs(%arg0 : memref<50xindex>) {
    ^bb0(%a: index):
      %i = linalg.index 0 : index
      linalg.yield %i : index
  }
  return
}
// TILE-10n25-DAG: [[$MAP:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0 + d1)>
// TILE-10n25-LABEL: func @indexed_vector
// TILE-10n25: %[[C10:.*]] = constant 10 : index
// TILE-10n25: scf.for %[[J:.*]] = {{.*}} step %[[C10]]
// TILE-10n25:   linalg.generic
// TILE-10n25:     %[[I:.*]] = linalg.index 0 : index
// TILE-10n25:     %[[NEW_I:.*]] = affine.apply [[$MAP]](%[[I]], %[[J]])
// TILE-10n25:     linalg.yield %[[NEW_I]] : index

// TILE-25n0-DAG: [[$MAP:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0 + d1)>
// TILE-25n0-LABEL: func @indexed_vector
// TILE-25n0: %[[C25:.*]] = constant 25 : index
// TILE-25n0: scf.for %[[J:.*]] = {{.*}} step %[[C25]]
// TILE-25n0:   linalg.generic
// TILE-25n0:     %[[I:.*]] = linalg.index 0 : index
// TILE-25n0:     %[[NEW_I:.*]] = affine.apply [[$MAP]](%[[I]], %[[J]])
// TILE-25n0:     linalg.yield %[[NEW_I]] : index

// TILE-0n25-LABEL: func @indexed_vector
// TILE-0n25-NOT: scf.for %[[J:.*]] = {{.*}} step %
// TILE-0n25: linalg.generic

// -----

func @indexed_matrix(%arg0: memref<50x50xindex>) {
  linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>],
                  iterator_types = ["parallel", "parallel"]}
    outs(%arg0 : memref<50x50xindex>) {
    ^bb0(%a: index):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %sum = addi %i, %j : index
      linalg.yield %sum : index
  }
  return
}
// TILE-10n25-DAG: [[$MAP:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0 + d1)>
// TILE-10n25-LABEL: func @indexed_matrix
// TILE-10n25-DAG: %[[C25:.*]] = constant 25 : index
// TILE-10n25-DAG: %[[C10:.*]] = constant 10 : index
// TILE-10n25: scf.for %[[K:.*]] = {{.*}} step %[[C10]]
// TILE-10n25:   scf.for %[[L:.*]] = {{.*}} step %[[C25]]
// TILE-10n25:     linalg.generic
// TILE-10n25:       %[[I:.*]] = linalg.index 0 : index
// TILE-10n25:       %[[NEW_I:.*]] = affine.apply [[$MAP]](%[[I]], %[[K]])
// TILE-10n25:       %[[J:.*]] = linalg.index 1 : index
// TILE-10n25:       %[[NEW_J:.*]] = affine.apply [[$MAP]](%[[J]], %[[L]])
// TILE-10n25:       %[[SUM:.*]] = addi %[[NEW_I]], %[[NEW_J]] : index
// TILE-10n25:       linalg.yield %[[SUM]] : index

// TILE-25n0-DAG: [[$MAP:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0 + d1)>
// TILE-25n0-LABEL: func @indexed_matrix
// TILE-25n0: %[[C25:.*]] = constant 25 : index
// TILE-25n0: scf.for %[[L:.*]] = {{.*}} step %[[C25]]
// TILE-25n0:   linalg.generic
// TILE-25n0:     %[[I:.*]] = linalg.index 0 : index
// TILE-25n0:     %[[NEW_I:.*]] = affine.apply [[$MAP]](%[[I]], %[[L]])
// TILE-25n0:     %[[J:.*]] = linalg.index 1 : index
// TILE-25n0:     %[[SUM:.*]] = addi %[[NEW_I]], %[[J]] : index
// TILE-25n0:     linalg.yield %[[SUM]] : index

// TILE-0n25-DAG: [[$MAP:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0 + d1)>
// TILE-0n25-LABEL: func @indexed_matrix
// TILE-0n25: %[[C25:.*]] = constant 25 : index
// TILE-0n25: scf.for %[[L:.*]] = {{.*}} step %[[C25]]
// TILE-0n25:   linalg.generic
// TILE-0n25:     %[[I:.*]] = linalg.index 0 : index
// TILE-0n25:     %[[J:.*]] = linalg.index 1 : index
// TILE-0n25:     %[[NEW_J:.*]] = affine.apply [[$MAP]](%[[J]], %[[L]])
// TILE-0n25:     %[[SUM:.*]] = addi %[[I]], %[[NEW_J]] : index
// TILE-0n25:     linalg.yield %[[SUM]] : index
