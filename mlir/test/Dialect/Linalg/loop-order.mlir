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

#map = affine_map<(i, j, k, l, m) -> (i, j, k, l, m)>
func @generic(%output: memref<1x2x3x4x5xindex>) {
  linalg.generic {indexing_maps = [#map],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    outs(%output : memref<1x2x3x4x5xindex>) {
    ^bb0(%arg0 : index):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %k = linalg.index 2 : index
    %l = linalg.index 3 : index
    %m = linalg.index 4 : index
    %0 = addi %i, %j : index
    %1 = addi %0, %k : index
    %2 = addi %1, %l : index
    %3 = addi %2, %m : index
    linalg.yield %3: index
  }
  return
}

// LOOP: scf.for %[[m:.*]] = %c0 to %c5 step %c1
// LOOP:   scf.for %[[i:.*]] = %c0 to %c1 step %c1
// LOOP:     scf.for %[[l:.*]] = %c0 to %c4 step %c1
// LOOP:       scf.for %[[j:.*]] = %c0 to %c2 step %c1
// LOOP:         scf.for %[[k:.*]] = %c0 to %c3 step %c1
// LOOP:           %{{.*}} = addi %[[i]], %[[j]] : index
// LOOP:           %{{.*}} = addi %{{.*}}, %[[k]] : index
// LOOP:           %{{.*}} = addi %{{.*}}, %[[l]] : index
// LOOP:           %{{.*}} = addi %{{.*}}, %[[m]] : index

// PARALLEL: 			scf.parallel (%[[m:.*]], %[[i:.*]], %[[l:.*]], %[[j:.*]], %[[k:.*]]) =
// PARALLEL-SAME:   to (%c5, %c1, %c4, %c2, %c3)
// PARALLEL:        %{{.*}} = addi %[[i]], %[[j]] : index
// PARALLEL:        %{{.*}} = addi %{{.*}}, %[[k]] : index
// PARALLEL:        %{{.*}} = addi %{{.*}}, %[[l]] : index
// PARALLEL:        %{{.*}} = addi %{{.*}}, %[[m]] : index

// AFFINE: affine.for %[[m:.*]] = 0 to 5
// AFFINE:   affine.for %[[i:.*]] = 0 to 1
// AFFINE:     affine.for %[[l:.*]] = 0 to 4
// AFFINE:       affine.for %[[j:.*]] = 0 to 2
// AFFINE:         affine.for %[[k:.*]] = 0 to 3
// AFFINE:           %{{.*}} = addi %[[i]], %[[j]] : index
// AFFINE:           %{{.*}} = addi %{{.*}}, %[[k]] : index
// AFFINE:           %{{.*}} = addi %{{.*}}, %[[l]] : index
// AFFINE:           %{{.*}} = addi %{{.*}}, %[[m]] : index
