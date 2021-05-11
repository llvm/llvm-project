// RUN: mlir-opt %s -test-linalg-greedy-fusion -split-input-file | FileCheck %s

#map = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#id_2d = affine_map<(d0, d1) -> (d0, d1)>
#pointwise_2d_trait = {
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}
func @fuse_indexed_consumer(%A: memref<?x?xf32>,
                                    %B: memref<?x?xf32>,
                                    %C: memref<?x?xf32>,
                                    %D: memref<?x?xf32>) {
  linalg.generic #pointwise_2d_trait
    ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
   outs(%C : memref<?x?xf32>) {
  ^bb0(%e: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %2 = addf %e, %arg5 : f32
    linalg.yield %2 : f32
  }
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c25 = constant 25 : index
  %c10 = constant 10 : index
  %0 = memref.dim %C, %c0 : memref<?x?xf32>
  %1 = memref.dim %C, %c1 : memref<?x?xf32>
  %2 = memref.dim %D, %c0 : memref<?x?xf32>
  %3 = memref.dim %D, %c1 : memref<?x?xf32>
  scf.for %arg2 = %c0 to %0 step %c10 {
    scf.for %arg3 = %c0 to %1 step %c25 {
      %4 = memref.subview %C[%arg2, %arg3][%c10, %c25][%c1, %c1] :
          memref<?x?xf32> to memref<?x?xf32, #map>
      %5 = memref.subview %D[%arg2, %arg3][%c10, %c25][%c1, %c1] :
          memref<?x?xf32> to memref<?x?xf32, #map>
      linalg.generic {
        indexing_maps = [#id_2d, #id_2d],
        iterator_types = ["parallel", "parallel"]}
        ins(%4 : memref<?x?xf32, #map>)
       outs(%5 : memref<?x?xf32, #map>) {
      ^bb0(%arg4: f32, %arg5: f32):
        %idx0 = linalg.index 0 : index
        %idx1 = linalg.index 1 : index
        %6 = addi %idx0, %arg2 : index
        %7 = addi %idx1, %arg3 : index
        %8 = index_cast %6 : index to i32
        %9 = sitofp %8 : i32 to f32
        %10 = index_cast %7 : index to i32
        %11 = sitofp %10 : i32 to f32
        %12 = addf %9, %11 : f32
        linalg.yield %12 : f32
      }
    }
  }
  return
}
// CHECK-LABEL: func @fuse_indexed_consumer
// CHECK:  scf.for
// CHECK:    scf.for
// CHECK-NOT:  scf.for
// CHECK:      linalg.generic
// CHECK-NOT:    affine.apply
// CHECK:        addf
// CHECK:      linalg.generic
// CHECK:        index_cast

// -----

#map = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
func @fuse_indexed_producer(%A: memref<?x?xindex>,
                            %B: memref<?x?xindex>) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c25 = constant 25 : index
  %c10 = constant 10 : index
  linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (j, i)>],
    iterator_types = ["parallel", "parallel"]}
    outs(%A : memref<?x?xindex>) {
  ^bb0(%a: index):   // no predecessors
    %idx0 = linalg.index 0 : index
    %idx1 = linalg.index 1 : index
    %0 = addi %idx0, %idx1 : index
    linalg.yield %0 : index
  }
  %A_X = memref.dim %A, %c0 : memref<?x?xindex>
  %A_Y = memref.dim %A, %c1 : memref<?x?xindex>
  scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%A_X, %A_Y) step (%c10, %c25) {
    %A_view = memref.subview %A[%arg2, %arg3][%c10, %c25][%c1, %c1] :
        memref<?x?xindex> to memref<?x?xindex, #map>
    %B_view = memref.subview %B[%arg2, %arg3][%c10, %c25][%c1, %c1] :
        memref<?x?xindex> to memref<?x?xindex, #map>
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%A_view : memref<?x?xindex, #map>)
      outs(%B_view : memref<?x?xindex, #map>) {
    ^bb0(%a: index, %b: index):
      linalg.yield %a : index
    }
  }
  return
}
// CHECK: [[$MAP:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: func @fuse_indexed_producer
// CHECK:  scf.parallel ([[I:%.*]], [[J:%.*]]) =
// CHECK:    linalg.generic
// CHECK:      [[idx0:%.*]] = linalg.index 0 : index
// CHECK:      [[i_new:%.*]] = affine.apply [[$MAP]]([[idx0]], [[J]])
// CHECK:      [[idx1:%.*]] = linalg.index 1 : index
// CHECK:      [[j_new:%.*]] = affine.apply [[$MAP]]([[idx1]], [[I]])
// CHECK:      [[sum:%.*]] = addi [[i_new]], [[j_new]] : index
// CHECK:      linalg.yield [[sum]] : index
// CHECK:    linalg.generic

// -----

#map = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
func @fuse_indexed_producer_tiled_second_dim_only(%A: memref<?x?xindex>,
                                                  %B: memref<?x?xindex>) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c25 = constant 25 : index
  linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]}
    outs(%A : memref<?x?xindex>) {
  ^bb0(%a: index):   // no predecessors
    %idx0 = linalg.index 0 : index
    %idx1 = linalg.index 1 : index
    %0 = addi %idx0, %idx1 : index
    linalg.yield %0 : index
  }
  %A_X = memref.dim %A, %c0 : memref<?x?xindex>
  %A_Y = memref.dim %A, %c1 : memref<?x?xindex>
  scf.parallel (%arg3) = (%c0) to (%A_Y) step (%c25) {
    %A_view = memref.subview %A[%c0, %arg3][%A_X, %c25][%c1, %c1] :
        memref<?x?xindex> to memref<?x?xindex, #map>
    %B_view = memref.subview %B[%c0, %arg3][%A_X, %c25][%c1, %c1] :
        memref<?x?xindex> to memref<?x?xindex, #map>
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%A_view : memref<?x?xindex, #map>)
      outs(%B_view : memref<?x?xindex, #map>) {
    ^bb0(%a: index, %b: index):
      linalg.yield %a : index
    }
  }
  return
}
// CHECK: [[$MAP:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: func @fuse_indexed_producer_tiled_second_dim_only
// CHECK:  scf.parallel ([[J:%.*]]) =
// CHECK:    linalg.generic
// CHECK:      [[idx0:%.*]] = linalg.index 0 : index
// CHECK:      [[idx1:%.*]] = linalg.index 1 : index
// CHECK:      [[j_new:%.*]] = affine.apply [[$MAP]]([[idx1]], [[J]])
// CHECK:      [[sum:%.*]] = addi [[idx0]], [[j_new]] : index
// CHECK:      linalg.yield [[sum]] : index
// CHECK:    linalg.generic

