// RUN: mlir-opt -affine-fold-memref-alias-ops -split-input-file %s | FileCheck %s

// Tests for folding memref aliasing ops (expand/collapse_shape and subview) into
// affine memory access operations, which require specific handling (and thus
// their own pass) because of the embedded affine maps.

// CHECK-LABEL: func @fold_static_stride_subview_with_affine_load_store
func.func @fold_static_stride_subview_with_affine_load_store(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) -> f32 {
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] : memref<12x32xf32> to memref<4x4xf32, strided<[64, 3], offset: ?>>
  %1 = affine.load %0[%arg3, %arg4] : memref<4x4xf32, strided<[64, 3], offset: ?>>
  // CHECK-NEXT: affine.apply
  // CHECK-NEXT: affine.apply
  // CHECK-NEXT: affine.load
  affine.store %1, %0[%arg3, %arg4] : memref<4x4xf32, strided<[64, 3], offset: ?>>
  // CHECK-NEXT: affine.apply
  // CHECK-NEXT: affine.apply
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: return
  return %1 : f32
}

// -----

// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<12x32xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> f32 {
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1], [2]] output_shape [2, 6, 32] : memref<12x32xf32> into memref<2x6x32xf32>
  %1 = affine.load %0[%arg1, %arg2, %arg3] : memref<2x6x32xf32>
  return %1 : f32
}
// CHECK: %[[INDEX:.*]] = affine.linearize_index disjoint [%[[ARG1]], %[[ARG2]]] by (2, 6)
// CHECK-NEXT: %[[RESULT:.*]] = affine.load %[[ARG0]][%[[INDEX]], %[[ARG3]]] : memref<12x32xf32>
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-LABEL: @fold_static_stride_subview_with_affine_load_store_collapse_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<2x6x32xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @fold_static_stride_subview_with_affine_load_store_collapse_shape(%arg0 : memref<2x6x32xf32>, %arg1 : index, %arg2 : index) -> f32 {
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<2x6x32xf32> into memref<12x32xf32>
  %1 = affine.load %0[%arg1, %arg2] : memref<12x32xf32>
  return %1 : f32
}
// CHECK-NEXT: %[[MODIFIED_INDEXES:.*]]:2 = affine.delinearize_index %[[ARG1]] into (2, 6)
// CHECK-NEXT: %[[RESULT:.*]] = affine.load %[[ARG0]][%[[MODIFIED_INDEXES]]#0, %[[MODIFIED_INDEXES]]#1, %[[ARG2]]] : memref<2x6x32xf32>
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-LABEL: @fold_dynamic_size_collapse_shape_with_affine_load
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x6x32xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @fold_dynamic_size_collapse_shape_with_affine_load(%arg0 : memref<?x6x32xf32>, %arg1 : index, %arg2 : index) -> f32 {
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<?x6x32xf32> into memref<?x32xf32>
  %1 = affine.load %0[%arg1, %arg2] : memref<?x32xf32>
  return %1 : f32
}
// CHECK-NEXT: %{{.*}}, %{{.*}}, %[[SIZES:.*]]:3, %{{.*}}:3 = memref.extract_strided_metadata %[[ARG0]]
// CHECK-NEXT: %[[MODIFIED_INDEXES:.*]]:2 = affine.delinearize_index %[[ARG1]] into (%[[SIZES]]#0, 6)
// CHECK-NEXT: %[[RESULT:.*]] = affine.load %[[ARG0]][%[[MODIFIED_INDEXES]]#0, %[[MODIFIED_INDEXES]]#1, %[[ARG2]]] : memref<?x6x32xf32>
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-LABEL: @fold_fully_dynamic_size_collapse_shape_with_affine_load
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @fold_fully_dynamic_size_collapse_shape_with_affine_load(%arg0 : memref<?x?x?xf32>, %arg1 : index, %arg2 : index) -> f32 {
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
  %1 = affine.load %0[%arg1, %arg2] : memref<?x?xf32>
  return %1 : f32
}
// CHECK-NEXT: %{{.*}}, %{{.*}}, %[[SIZES:.*]]:3, %{{.*}}:3 = memref.extract_strided_metadata %[[ARG0]]
// CHECK-NEXT: %[[MODIFIED_INDEXES:.*]]:2 = affine.delinearize_index %[[ARG1]] into (%[[SIZES]]#0, %[[SIZES]]#1)
// CHECK-NEXT: %[[RESULT:.*]] = affine.load %[[ARG0]][%[[MODIFIED_INDEXES]]#0, %[[MODIFIED_INDEXES]]#1, %[[ARG2]]] : memref<?x?x?xf32>
// CHECK-NEXT: return %[[RESULT]] : f32


// -----

// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape_3d
// CHECK-SAME: (%[[ARG0:.*]]: memref<12x32xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index) -> f32 {
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape_3d(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4: index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1, 2], [3]] output_shape [2, 2, 3, 32] : memref<12x32xf32> into memref<2x2x3x32xf32>
  %1 = affine.load %0[%arg1, %arg2, %arg3, %arg4] : memref<2x2x3x32xf32>
  return %1 : f32
}
// CHECK: %[[INDEX:.*]] = affine.linearize_index disjoint [%[[ARG1]], %[[ARG2]], %[[ARG3]]] by (2, 2, 3)
// CHECK-NEXT: %[[RESULT:.*]] = affine.load %[[ARG0]][%[[INDEX]], %[[ARG4]]] : memref<12x32xf32>
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-LABEL: fold_memref_alias_expand_shape_subview_load_store_dynamic_dim
// CHECK-SAME: (%[[ARG0:.*]]: memref<2048x16xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index)
func.func @fold_memref_alias_expand_shape_subview_load_store_dynamic_dim(%alloc: memref<2048x16xf32>, %c10: index, %c5: index, %c0: index, %sz0: index) {
  %subview = memref.subview %alloc[%c5, 0] [%c10, 16] [1, 1] : memref<2048x16xf32> to memref<?x16xf32, strided<[16, 1], offset: ?>>
  %expand_shape = memref.expand_shape %subview [[0], [1, 2, 3]] output_shape [%sz0, 1, 8, 2] : memref<?x16xf32, strided<[16, 1], offset: ?>> into memref<?x1x8x2xf32, strided<[16, 16, 2, 1], offset: ?>>
  %dim = memref.dim %expand_shape, %c0 : memref<?x1x8x2xf32, strided<[16, 16, 2, 1], offset: ?>>

  affine.for %arg6 = 0 to %dim step 64 {
    affine.for %arg7 = 0 to 16 step 16 {
      %dummy_load = affine.load %expand_shape[%arg6, 0, %arg7, %arg7] : memref<?x1x8x2xf32, strided<[16, 16, 2, 1], offset: ?>>
      affine.store %dummy_load, %subview[%arg6, %arg7] : memref<?x16xf32, strided<[16, 1], offset: ?>>
    }
  }
  return
}
// CHECK-NEXT:   %[[C0:.*]] = arith.constant 0
// CHECK-NEXT:   memref.subview
// CHECK-NEXT:   %[[EXPAND_SHAPE:.*]] = memref.expand_shape
// CHECK-NEXT:   %[[DIM:.*]] = memref.dim %[[EXPAND_SHAPE]], %[[ARG3]] : memref<?x1x8x2xf32, strided<[16, 16, 2, 1], offset: ?>>
// CHECK-NEXT:   affine.for %[[ARG5:.*]] = 0 to %[[DIM]] step 64 {
// CHECK-NEXT:   affine.for %[[ARG6:.*]] = 0 to 16 step 16 {
// CHECK-NEXT:   %[[VAL0:.*]] = affine.linearize_index disjoint [%[[C0]], %[[ARG6]], %[[ARG6]]] by (1, 8, 2)
// CHECK-NEXT:   %[[VAL1:.*]] = affine.apply #[[$MAP0]](%[[ARG5]])[%[[ARG2]]]
// CHECK-NEXT:   %[[VAL2:.*]] = affine.load %[[ARG0]][%[[VAL1]], %[[VAL0]]] : memref<2048x16xf32>
// CHECK-NEXT:   %[[VAL3:.*]] = affine.apply #[[$MAP0]](%[[ARG5]])[%[[ARG2]]]
// CHECK-NEXT:   affine.store %[[VAL2]], %[[ARG0]][%[[VAL3]], %[[ARG6]]] : memref<2048x16xf32>

// -----

// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape_loops
// CHECK-SAME: (%[[ARG0:.*]]: memref<1024x1024xf32>, %[[ARG1:.*]]: memref<1xf32>, %[[ARG2:.*]]: index)
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape_loops(%arg0: memref<1024x1024xf32>, %arg1: memref<1xf32>, %arg2: index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [1, 1024, 1024, 1] : memref<1024x1024xf32> into memref<1x1024x1024x1xf32>
  affine.for %arg3 = 0 to 1 {
    affine.for %arg4 = 0 to 1024 {
      affine.for %arg5 = 0 to 1020 {
        affine.for %arg6 = 0 to 1 {
          %1 = affine.load %0[%arg3, %arg4, %arg5, %arg6] : memref<1x1024x1024x1xf32>
          affine.store %1, %arg1[%arg2] : memref<1xf32>
        }
      }
    }
  }
  %2 = affine.load %arg1[%arg2] : memref<1xf32>
  return %2 : f32
}
// CHECK-NEXT: affine.for %[[ARG3:.*]] = 0 to 1 {
// CHECK-NEXT:  affine.for %[[ARG4:.*]] = 0 to 1024 {
// CHECK-NEXT:   affine.for %[[ARG5:.*]] = 0 to 1020 {
// CHECK-NEXT:    affine.for %[[ARG6:.*]] = 0 to 1 {
// CHECK-NEXT:     %[[IDX1:.*]] = affine.linearize_index disjoint [%[[ARG3]], %[[ARG4]]] by (1, 1024)
// CHECK-NEXT:     %[[IDX2:.*]] = affine.linearize_index disjoint [%[[ARG5]], %[[ARG6]]] by (1024, 1)
// CHECK-NEXT:     affine.load %[[ARG0]][%[[IDX1]], %[[IDX2]]] : memref<1024x1024xf32>

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_expand_shape_when_access_index_is_an_expression
// CHECK-SAME: (%[[ARG0:.*]]: memref<1024x1024xf32>, %[[ARG1:.*]]: memref<1xf32>, %[[ARG2:.*]]: index)
func.func @fold_static_stride_subview_with_affine_load_store_expand_shape_when_access_index_is_an_expression(%arg0: memref<1024x1024xf32>, %arg1: memref<1xf32>, %arg2: index) -> f32 {
  %0 = memref.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [1, 1024, 1024, 1] : memref<1024x1024xf32> into memref<1x1024x1024x1xf32>
  affine.for %arg3 = 0 to 1 {
    affine.for %arg4 = 0 to 1024 {
      affine.for %arg5 = 0 to 1020 {
        affine.for %arg6 = 0 to 1 {
          %1 = affine.load %0[%arg3, %arg4 + %arg3, %arg5, %arg6] : memref<1x1024x1024x1xf32>
          affine.store %1, %arg1[%arg2] : memref<1xf32>
        }
      }
    }
  }
  %2 = affine.load %arg1[%arg2] : memref<1xf32>
  return %2 : f32
}
// CHECK-NEXT: affine.for %[[ARG3:.*]] = 0 to 1 {
// CHECK-NEXT:  affine.for %[[ARG4:.*]] = 0 to 1024 {
// CHECK-NEXT:   affine.for %[[ARG5:.*]] = 0 to 1020 {
// CHECK-NEXT:    affine.for %[[ARG6:.*]] = 0 to 1 {
// CHECK-NEXT:      %[[TMP1:.*]] = affine.apply #[[$MAP0]](%[[ARG3]], %[[ARG4]])
// CHECK-NEXT:      %[[TMP2:.*]] = affine.linearize_index disjoint [%[[ARG3]], %[[TMP1]]] by (1, 1024)
// CHECK-NEXT:      %[[TMP3:.*]] = affine.linearize_index disjoint [%[[ARG5]], %[[ARG6]]] by (1024, 1)
// CHECK-NEXT:      affine.load %[[ARG0]][%[[TMP2]], %[[TMP3]]] : memref<1024x1024xf32>

// -----

// CHECK-LABEL: fold_static_stride_subview_with_affine_load_store_collapse_shape_with_0d_result
// CHECK-SAME: (%[[ARG0:.*]]: memref<1xf32>, %[[ARG1:.*]]: memref<1xf32>)
func.func @fold_static_stride_subview_with_affine_load_store_collapse_shape_with_0d_result(%arg0: memref<1xf32>, %arg1: memref<1xf32>) -> memref<1xf32> {
  %0 = memref.collapse_shape %arg0 [] : memref<1xf32> into memref<f32>
  affine.for %arg2 = 0 to 3 {
    %1 = affine.load %0[] : memref<f32>
    affine.store %1, %arg1[0] : memref<1xf32>
  }
  return %arg1 : memref<1xf32>
}
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT: affine.for %{{.*}} = 0 to 3 {
// CHECK-NEXT:   affine.load %[[ARG0]][%[[ZERO]]] : memref<1xf32>

