// RUN: mlir-opt %s -linalg-fusion -split-input-file | FileCheck %s

#map = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#id_2d = affine_map<(d0, d1) -> (d0, d1)>
#pointwise_2d_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}
func @fuse_indexed_generic_consumer(%A: memref<?x?xf32>,
                                    %B: memref<?x?xf32>,
                                    %C: memref<?x?xf32>,
                                    %D: memref<?x?xf32>) {
  linalg.generic #pointwise_2d_trait %A, %B, %C {
  ^bb0(%e: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %2 = addf %e, %arg5 : f32
    linalg.yield %2 : f32
  }: memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c25 = constant 25 : index
  %c10 = constant 10 : index
  %0 = dim %C, %c0 : memref<?x?xf32>
  %1 = dim %C, %c1 : memref<?x?xf32>
  %2 = dim %D, %c0 : memref<?x?xf32>
  %3 = dim %D, %c1 : memref<?x?xf32>
  scf.for %arg2 = %c0 to %0 step %c10 {
    scf.for %arg3 = %c0 to %1 step %c25 {
      %4 = std.subview %C[%arg2, %arg3][%c10, %c25][%c1, %c1] :
          memref<?x?xf32> to memref<?x?xf32, #map>
      %5 = std.subview %D[%arg2, %arg3][%c10, %c25][%c1, %c1] :
          memref<?x?xf32> to memref<?x?xf32, #map>
      linalg.indexed_generic {
        indexing_maps = [#id_2d, #id_2d],
        iterator_types = ["parallel", "parallel"],
        args_in = 1,
        args_out = 1
      } %4, %5 {
      ^bb0(%arg4: index, %arg5: index, %arg6: f32, %arg7: f32):
        %6 = addi %arg4, %arg2 : index
        %7 = addi %arg5, %arg3 : index
        %8 = index_cast %6 : index to i32
        %9 = sitofp %8 : i32 to f32
        %10 = index_cast %7 : index to i32
        %11 = sitofp %10 : i32 to f32
        %12 = addf %9, %11 : f32
        linalg.yield %12 : f32
      }: memref<?x?xf32, #map>, memref<?x?xf32, #map>
    }
  }
  return
}
// CHECK-LABEL: func @fuse_indexed_generic_consumer
// CHECK:  scf.for
// CHECK:    scf.for
// CHECK-NOT:  scf.for
// CHECK:      linalg.generic
// CHECK-NOT:    addi
// CHECK:        addf
// CHECK:      linalg.indexed_generic
// CHECK:        index_cast

// -----

#map = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#id_2d = affine_map<(d0, d1) -> (d0, d1)>
#pointwise_2d_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}
func @fuse_indexed_generic_producer(%A: memref<?x?xf32>,
                                    %B: memref<?x?xf32>,
                                    %C: memref<?x?xf32>,
                                    %D: memref<?x?xf32>) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c25 = constant 25 : index
  %c10 = constant 10 : index
  linalg.indexed_generic #pointwise_2d_trait %A, %B, %C {
    ^bb0(%i: index, %j: index, %a: f32, %b: f32, %c: f32): // no predecessors
      %i_int = index_cast %i: index to i32
      %i_float = sitofp %i_int : i32 to f32
      %ab = addf %a, %b : f32
      %out = addf %ab, %i_float : f32
      linalg.yield %out : f32
  }: memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  %C_X = dim %C, %c0 : memref<?x?xf32>
  %C_Y = dim %C, %c1 : memref<?x?xf32>
  %D_X = dim %D, %c0 : memref<?x?xf32>
  %D_Y = dim %D, %c1 : memref<?x?xf32>
  scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%C_X, %C_Y) step (%c10, %c25) {
    %C_view = std.subview %C[%arg2, %arg3][%c10, %c25][%c1, %c1] :
        memref<?x?xf32> to memref<?x?xf32, #map>
    %D_view = std.subview %D[%arg2, %arg3][%c10, %c25][%c1, %c1] :
        memref<?x?xf32> to memref<?x?xf32, #map>
    linalg.generic {
      indexing_maps = [#id_2d, #id_2d],
      iterator_types = ["parallel", "parallel"],
      args_in = 1,
      args_out = 1
    } %C_view, %D_view {
    ^bb0( %a: f32, %b: f32):
      %ab = addf %a, %b : f32
      linalg.yield %ab : f32
    }: memref<?x?xf32, #map>, memref<?x?xf32, #map>
  }
  return
}
// CHECK-LABEL: func @fuse_indexed_generic_producer
// CHECK:  scf.parallel ([[I:%.*]], [[J:%.*]]) =
// CHECK-NOT:  scf.parallel
// CHECK:      linalg.indexed_generic
// CHECK:        ^bb0([[i:%.*]]: index, [[j:%.*]]: index
// CHECK:          [[i_new:%.*]] = addi [[i]], [[I]] : index
// CHECK:          [[j_new:%.*]] = addi [[j]], [[J]] : index
// CHECK:          {{.*}} = index_cast [[i_new]] : index to i32
// CHECK:      linalg.generic
// CHECK:          addf

// -----

#map = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#id_2d = affine_map<(d0, d1) -> (d0, d1)>
#pointwise_2d_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}
func @fuse_indexed_generic_producer_tile_second_dim_only(%A: memref<?x?xf32>,
                                                         %B: memref<?x?xf32>,
                                                         %C: memref<?x?xf32>,
                                                         %D: memref<?x?xf32>) {
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %c0 = constant 0 : index
  linalg.indexed_generic #pointwise_2d_trait %A, %B, %C {
    ^bb0(%i: index, %j: index, %a: f32, %b: f32, %c: f32): // no predecessors
      %j_int = index_cast %j: index to i32
      %j_float = sitofp %j_int : i32 to f32
      %ab = addf %a, %b : f32
      %out = addf %ab, %j_float : f32
      linalg.yield %out : f32
  }: memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  %C_X = dim %C, %c0 : memref<?x?xf32>
  %C_Y = dim %C, %c1 : memref<?x?xf32>
  %D_X = dim %D, %c0 : memref<?x?xf32>
  %D_Y = dim %D, %c1 : memref<?x?xf32>
  %3 = linalg.range %c0 : %C_Y : %c3 : !linalg.range
  scf.parallel (%j) = (%c0) to (%C_Y) step (%c3) {
    %0 = affine.min affine_map<(d0, d1, d2) -> (d0, d1 - d2)>(%c3, %C_Y, %j)
    %C_view = subview %C[%c0, %j] [%C_X, %0] [%c1, %c1] :
      memref<?x?xf32> to memref<?x?xf32, #map>

    %1 = affine.min affine_map<(d0, d1, d2) -> (d0, d1 - d2)>(%c3, %D_Y, %j)
    %D_view = subview %D[%c0, %j] [%D_X, %1] [%c1, %c1] :
      memref<?x?xf32> to memref<?x?xf32, #map>

    linalg.generic {
      indexing_maps = [#id_2d, #id_2d],
      iterator_types = ["parallel", "parallel"],
      args_in = 1,
      args_out = 1
    } %C_view, %D_view {
    ^bb0( %a: f32, %b: f32):
      %ab = addf %a, %b : f32
      linalg.yield %ab : f32
    }: memref<?x?xf32, #map>, memref<?x?xf32, #map>
    scf.yield
  }
  return
}
// CHECK-LABEL: func @fuse_indexed_generic_producer_tile_second_dim_only
// CHECK:  [[C0:%.*]] = constant 0 : index
// CHECK:  scf.parallel ([[J:%.*]]) =
// CHECK-NOT:  scf.parallel
// CHECK:      linalg.indexed_generic
// CHECK:        ^bb0([[i:%.*]]: index, [[j:%.*]]: index
// CHECK:          [[i_new:%.*]] = addi [[i]], [[C0]] : index
// CHECK:          [[j_new:%.*]] = addi [[j]], [[J]] : index
// CHECK:          {{.*}} = index_cast [[j_new]] : index to i32
// CHECK:      linalg.generic
// CHECK:          addf
