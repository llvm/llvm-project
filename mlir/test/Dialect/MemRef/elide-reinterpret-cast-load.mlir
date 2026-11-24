// RUN: mlir-opt -memref-elide-reinterpret-cast %s | FileCheck %s

/// Tests for the RewriteLoadFromReinterpretCast pattern
/// to show how reinterpret_cast is elided.

//===----------------------------------------------------------------------===//
// Positive tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func private @expand_scalar(
// CHECK-SAME:    %[[SRC:.*]]: memref<1xi64>) {
func.func private @expand_scalar(%src : memref<1xi64>) {
  // CHECK:       %[[IDX:.*]] = arith.constant 0 : index
  %idx = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 1, 1], strides: [1, 1, 1]
    : memref<1xi64> to memref<1x1x1xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX]]] : memref<1xi64>
  %0 = memref.load %reinterpret_cast[%idx, %idx, %idx] : memref<1x1x1xi64>
  return
}

// CHECK-LABEL: func.func private @collapse_scalar(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x1xi64>) {
func.func private @collapse_scalar(%src : memref<1x1x1xi64>) {
  // CHECK:   %[[IDX:.*]] = arith.constant 0 : index
  %idx = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 1], strides: [1, 1]
    : memref<1x1x1xi64> to memref<1x1xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX]], %[[IDX]], %[[IDX]]] : memref<1x1x1xi64>
  %0 = memref.load %reinterpret_cast[%idx, %idx] : memref<1x1xi64>
  return
}

/// 1x999 is effectively a 1D MemRef
// CHECK-LABEL: func.func private @expand_1D(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x999xf32>) {
func.func private @expand_1D(
    %src : memref<1x999xf32>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 1, 999], strides: [999, 999, 1]
    : memref<1x999xf32> to memref<1x1x999xf32>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_1]], %[[IDX_2]]] : memref<1x999xf32>
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_1, %idx_2]
    : memref<1x1x999xf32>
  return
}

/// Same as @expand_1D, with attributes on the load.
// CHECK-LABEL: func.func private @expand_1D_load_attrs(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x999xf32>) {
func.func private @expand_1D_load_attrs(
    %src : memref<1x999xf32>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 1, 999], strides: [999, 999, 1]
    : memref<1x999xf32> to memref<1x1x999xf32>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_1]], %[[IDX_2]]] alignment(16) nontemporal(true) invariant(true) : memref<1x999xf32>
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_1, %idx_2]
    alignment(16) nontemporal(true) invariant(true) : memref<1x1x999xf32>
  return
}

// CHECK-LABEL: func.func private @collapse_1D(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x999xf32>) {
func.func private @collapse_1D(
    %src : memref<1x1x999xf32>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 999], strides: [999, 1]
    : memref<1x1x999xf32> to memref<1x999xf32>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_1]], %[[IDX_1]], %[[IDX_2]]] : memref<1x1x999xf32>
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_2] : memref<1x999xf32>
  return
}

// CHECK-LABEL: func.func private @expand_1D_dynamic_index(
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<1x999xi64>) {
func.func private @expand_1D_dynamic_index(%i : index,
    %src : memref<1x999xi64>) {
  // CHECK:       %[[IDX:.*]] = arith.constant 0 : index
  %idx = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 1, 999], strides: [999, 999, 1]
    : memref<1x999xi64> to memref<1x1x999xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX]], %[[I]]] : memref<1x999xi64>
  %0 = memref.load %reinterpret_cast[%idx, %idx, %i] : memref<1x1x999xi64>
  return
}

// CHECK-LABEL: func.func private @collapse_1D_dynamic_index(
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x999xi64>) {
func.func private @collapse_1D_dynamic_index(%i : index,
    %src : memref<1x1x999xi64>) {
  // CHECK-DAG:   %[[IDX:.*]] = arith.constant 0 : index
  %idx = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 999], strides: [999, 1]
    : memref<1x1x999xi64> to memref<1x999xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX]], %[[IDX]], %[[I]]] : memref<1x1x999xi64>
  %0 = memref.load %reinterpret_cast[%idx, %i] : memref<1x999xi64>
  return
}

// CHECK-LABEL: func.func private @expand_multiple_non_unit_dims(
// CHECK-SAME:    %[[SRC:.*]]: memref<17x100xf32>) {
func.func private @expand_multiple_non_unit_dims(
    %src : memref<17x100xf32>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [17, 1, 1, 100], strides: [100, 100, 100, 1]
    : memref<17x100xf32> to memref<17x1x1x100xf32,
      strided<[100, 100, 100, 1]>>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_2]], %[[IDX_2]]] : memref<17x100xf32>
  %0 = memref.load %reinterpret_cast[%idx_2, %idx_1, %idx_1, %idx_2]
    : memref<17x1x1x100xf32, strided<[100, 100, 100, 1]>>
  return
}

// CHECK-LABEL: func.func private @collapse_multiple_non_unit_dims(
// CHECK-SAME:    %[[SRC:.*]]: memref<17x1x1x100xf32>) {
func.func private @collapse_multiple_non_unit_dims(
    %src : memref<17x1x1x100xf32>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [17, 100], strides: [100, 1]
    : memref<17x1x1x100xf32> to memref<17x100xf32>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_2]], %[[IDX_1]], %[[IDX_1]], %[[IDX_2]]] : memref<17x1x1x100xf32>
  %0 = memref.load %reinterpret_cast[%idx, %idx] : memref<17x100xf32>
  return
}

// CHECK-LABEL: func.func private @expand_inner_non_unit_dims(
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<1x33xf32>) {
func.func private @expand_inner_non_unit_dims(%i : index,
    %src : memref<1x33xf32>) {
  // CHECK:       %[[IDX:.*]] = arith.constant 0 : index
  %idx = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 33, 1, 1], strides: [33, 1, 1, 1]
    : memref<1x33xf32> to memref<1x33x1x1xf32>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX]], %[[I]]] : memref<1x33xf32>
  %0 = memref.load %reinterpret_cast[%idx, %i, %idx, %idx]
    : memref<1x33x1x1xf32>
  return
}

// CHECK-LABEL: func.func private @collapse_inner_non_unit_dims(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x1x100xf32>) {
func.func private @collapse_inner_non_unit_dims(
    %src : memref<1x1x1x100xf32>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 100, 1], strides: [100, 1, 100]
    : memref<1x1x1x100xf32> to memref<1x100x1xf32, strided<[100, 1, 100]>>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_1]], %[[IDX_1]], %[[IDX_1]], %[[IDX_2]]] : memref<1x1x1x100xf32>
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_2, %idx_1] : memref<1x100x1xf32,
    strided<[100, 1, 100]>>
  return
}

// CHECK-LABEL: func.func private @expand_diff_non_unit_boundary(
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<1x33xf32>) {
func.func private @expand_diff_non_unit_boundary(%i : index,
    %src : memref<1x33xf32>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [33, 1, 1], strides: [1, 33, 33]
    : memref<1x33xf32> to memref<33x1x1xf32, strided<[1, 33, 33]>>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_1]], %[[IDX_2]]] : memref<1x33xf32>
  %0 = memref.load %reinterpret_cast[%idx_2, %idx_1, %idx_1]
    : memref<33x1x1xf32, strided<[1, 33, 33]>>
  return
}

// CHECK-LABEL: func.func private @collapse_diff_non_unit_boundary(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x1x100xf32>) {
func.func private @collapse_diff_non_unit_boundary(
    %src : memref<1x1x1x100xf32>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [100, 1, 1], strides: [1, 100, 100]
    : memref<1x1x1x100xf32> to memref<100x1x1xf32, strided<[1, 100, 100]>>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_1]], %[[IDX_1]], %[[IDX_1]], %[[IDX_2]]] : memref<1x1x1x100xf32>
  %0 = memref.load %reinterpret_cast[%idx_2, %idx_1, %idx_1] : memref<100x1x1xf32,
    strided<[1, 100, 100]>>
  return
}

// CHECK-LABEL: func.func private @expand_3d_moved_unit_dims(
// CHECK-SAME:    %[[I:[A-Za-z0-9_]+]]: index
// CHECK-SAME:    %[[J:[A-Za-z0-9_]+]]: index
// CHECK-SAME:    %[[K:[A-Za-z0-9_]+]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<1x3x22x3xf32>) {
func.func private @expand_3d_moved_unit_dims(%i : index, %j : index,
    %k : index, %src : memref<1x3x22x3xf32>) {
  // CHECK:       %[[IDX:.*]] = arith.constant 0 : index
  %idx = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [3, 1, 1, 22, 1, 3],
    strides: [66, 66, 66, 3, 3, 1]
    : memref<1x3x22x3xf32> to memref<3x1x1x22x1x3xf32,
      strided<[66, 66, 66, 3, 3, 1]>>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX]], %[[I]], %[[J]], %[[K]]] : memref<1x3x22x3xf32>
  %0 = memref.load %reinterpret_cast[%i, %idx, %idx, %j, %idx, %k]
    : memref<3x1x1x22x1x3xf32, strided<[66, 66, 66, 3, 3, 1]>>
  return
}

// CHECK-LABEL: func.func private @collapse_3d_moved_unit_dims(
// CHECK-SAME:    %[[I:[A-Za-z0-9_]+]]: index
// CHECK-SAME:    %[[J:[A-Za-z0-9_]+]]: index
// CHECK-SAME:    %[[K:[A-Za-z0-9_]+]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<1x3x1x1x22x1x3xf32>) {
func.func private @collapse_3d_moved_unit_dims(%i : index, %j : index,
    %k : index, %src : memref<1x3x1x1x22x1x3xf32>) {
  // CHECK:       %[[IDX:.*]] = arith.constant 0 : index
  %idx_1 = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [3, 1, 22, 3, 1, 1],
    strides: [66, 66, 3, 1, 1, 1]
    : memref<1x3x1x1x22x1x3xf32> to memref<3x1x22x3x1x1xf32,
      strided<[66, 66, 3, 1, 1, 1]>>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX]], %[[I]], %[[IDX]], %[[IDX]], %[[J]], %[[IDX]], %[[K]]] : memref<1x3x1x1x22x1x3xf32>
  %0 = memref.load %reinterpret_cast[%i, %idx_1, %j, %k, %idx_1, %idx_1]
    : memref<3x1x22x3x1x1xf32, strided<[66, 66, 3, 1, 1, 1]>>
  return
}

/// The window shrinks the source at a zero offset, so the load indices stay
/// unshifted.
// CHECK-LABEL: func.func private @window_zero_offset(
// CHECK-SAME:    %[[SRC:.*]]: memref<17x100xf32>) {
func.func private @window_zero_offset(%src : memref<17x100xf32>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 1 : index
  // CHECK-DAG:   %[[IDX_3:.*]] = arith.constant 3 : index
  %idx_1 = arith.constant 1 : index
  %idx_3 = arith.constant 3 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [4, 50], strides: [100, 1]
    : memref<17x100xf32> to memref<4x50xf32, strided<[100, 1]>>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_1]], %[[IDX_3]]] : memref<17x100xf32>
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_3]
    : memref<4x50xf32, strided<[100, 1]>>
  return
}

/// Same as @window_zero_offset, but offset along dim 0 up to the source bound.
// CHECK-LABEL: func.func private @window_single_dim_offset(
// CHECK-SAME:    %[[SRC:.*]]: memref<17x100xf32>) {
func.func private @window_single_dim_offset(%src : memref<17x100xf32>) {
  // CHECK-DAG:   %[[IDX_3:.*]] = arith.constant 3 : index
  %idx_1 = arith.constant 1 : index
  %idx_3 = arith.constant 3 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [1300], sizes: [4, 50], strides: [100, 1]
    : memref<17x100xf32> to memref<4x50xf32, strided<[100, 1], offset: 1300>>
  // CHECK:       %[[C14:.*]] = arith.constant 14 : index
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[C14]], %[[IDX_3]]] : memref<17x100xf32>
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_3]
    : memref<4x50xf32, strided<[100, 1], offset: 1300>>
  return
}

/// Same as @window_single_dim_offset, but also offset along dim 1.
// CHECK-LABEL: func.func private @window_multi_dim_offset(
// CHECK-SAME:    %[[SRC:.*]]: memref<17x100xf32>) {
func.func private @window_multi_dim_offset(%src : memref<17x100xf32>) {
  %idx_1 = arith.constant 1 : index
  %idx_3 = arith.constant 3 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [1350], sizes: [4, 50], strides: [100, 1]
    : memref<17x100xf32> to memref<4x50xf32, strided<[100, 1], offset: 1350>>
  // CHECK-DAG:   %[[C14:.*]] = arith.constant 14 : index
  // CHECK-DAG:   %[[C53:.*]] = arith.constant 53 : index
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[C14]], %[[C53]]] : memref<17x100xf32>
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_3]
    : memref<4x50xf32, strided<[100, 1], offset: 1350>>
  return
}

/// Same as @window_multi_dim_offset, but with dynamic load indices.
// CHECK-LABEL: func.func private @window_dynamic_index(
// CHECK-SAME:    %[[I:[A-Za-z0-9_]+]]: index
// CHECK-SAME:    %[[J:[A-Za-z0-9_]+]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<17x100xf32>) {
func.func private @window_dynamic_index(%i : index, %j : index,
    %src : memref<17x100xf32>) {
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [1350], sizes: [4, 50], strides: [100, 1]
    : memref<17x100xf32> to memref<4x50xf32, strided<[100, 1], offset: 1350>>
  // CHECK-DAG:   %[[C13:.*]] = arith.constant 13 : index
  // CHECK-DAG:   %[[C50:.*]] = arith.constant 50 : index
  // CHECK-DAG:   %[[I13:.*]] = arith.addi %[[I]], %[[C13]] : index
  // CHECK-DAG:   %[[J50:.*]] = arith.addi %[[J]], %[[C50]] : index
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[I13]], %[[J50]]] : memref<17x100xf32>
  %0 = memref.load %reinterpret_cast[%i, %j]
    : memref<4x50xf32, strided<[100, 1], offset: 1350>>
  return
}

/// Multi-dim offset on the source of @expand_3d_moved_unit_dims.
// CHECK-LABEL: func.func private @window_3d_multi_dim_offset(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x3x22x3xf32>) {
func.func private @window_3d_multi_dim_offset(%src : memref<1x3x22x3xf32>) {
  // CHECK-DAG:   %[[IDX_0:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 1 : index
  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  %idx_3 = arith.constant 3 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [81], sizes: [1, 2, 10, 3], strides: [198, 66, 3, 1]
    : memref<1x3x22x3xf32>
      to memref<1x2x10x3xf32, strided<[198, 66, 3, 1], offset: 81>>
  // CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_0]], %[[C2]], %[[C8]], %[[IDX_1]]] : memref<1x3x22x3xf32>
  %0 = memref.load %reinterpret_cast[%idx_0, %idx_1, %idx_3, %idx_1]
    : memref<1x2x10x3xf32, strided<[198, 66, 3, 1], offset: 81>>
  return
}

//===----------------------------------------------------------------------===//
// Negative tests (must NOT rewrite)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func private @negative_nonzero_offset(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x100xf32>) {
func.func private @negative_nonzero_offset(
    %src : memref<1x100xf32>) {
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [1], sizes: [1, 1, 100], strides: [1, 1, 1]
    : memref<1x100xf32> to memref<1x1x100xf32, strided<[1, 1, 1], offset: 1>>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_1, %idx_2]
    : memref<1x1x100xf32, strided<[1, 1, 1], offset: 1>>
  return
}

/// Same as @window_single_dim_offset, but the window ends past the source
/// bound.
// CHECK-LABEL: func.func private @negative_window_out_of_bounds(
// CHECK-SAME:    %[[SRC:.*]]: memref<17x100xf32>) {
func.func private @negative_window_out_of_bounds(%src : memref<17x100xf32>) {
  %idx_1 = arith.constant 1 : index
  %idx_3 = arith.constant 3 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [1400], sizes: [4, 50], strides: [100, 1]
    : memref<17x100xf32> to memref<4x50xf32, strided<[100, 1], offset: 1400>>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_3]
    : memref<4x50xf32, strided<[100, 1], offset: 1400>>
  return
}

/// Same as @window_zero_offset, but with non-identity strides.
// CHECK-LABEL: func.func private @negative_window_strides_mismatch(
// CHECK-SAME:    %[[SRC:.*]]: memref<17x100xf32>) {
func.func private @negative_window_strides_mismatch(%src : memref<17x100xf32>) {
  %idx_1 = arith.constant 1 : index
  %idx_3 = arith.constant 3 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [4, 50], strides: [200, 1]
    : memref<17x100xf32> to memref<4x50xf32, strided<[200, 1]>>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_3]
    : memref<4x50xf32, strided<[200, 1]>>
  return
}

/// Same as @window_single_dim_offset, but the offset starts past the source
/// end.
// CHECK-LABEL: func.func private @negative_window_offset_past_end(
// CHECK-SAME:    %[[SRC:.*]]: memref<17x100xf32>) {
func.func private @negative_window_offset_past_end(%src : memref<17x100xf32>) {
  %idx_1 = arith.constant 1 : index
  %idx_3 = arith.constant 3 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [1700], sizes: [4, 50], strides: [100, 1]
    : memref<17x100xf32> to memref<4x50xf32, strided<[100, 1], offset: 1700>>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_3]
    : memref<4x50xf32, strided<[100, 1], offset: 1700>>
  return
}

/// Same as @window_multi_dim_offset, but with a non-identity source layout.
// CHECK-LABEL: func.func private @negative_window_non_identity_source(
// CHECK-SAME:    %[[SRC:.*]]: memref<17x100xf32, strided<[100, 1], offset: 5>>) {
func.func private @negative_window_non_identity_source(
    %src : memref<17x100xf32, strided<[100, 1], offset: 5>>) {
  %idx_1 = arith.constant 1 : index
  %idx_3 = arith.constant 3 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [1350], sizes: [4, 50], strides: [100, 1]
    : memref<17x100xf32, strided<[100, 1], offset: 5>>
      to memref<4x50xf32, strided<[100, 1], offset: 1350>>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_3]
    : memref<4x50xf32, strided<[100, 1], offset: 1350>>
  return
}

// CHECK-LABEL: func.func private @negative_dynamic_shape(
// CHECK-SAME:   %[[SRC:[A-Za-z][A-Za-z0-9-]*]]: memref<?xf32>
func.func private @negative_dynamic_shape(%dim : index,
    %src : memref<?xf32>) {
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, %dim], strides: [1, 1]
    : memref<?xf32> to memref<1x?xf32, strided<[1, 1]>>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_2] :
    memref<1x?xf32, strided<[1, 1]>>
  return
}

// CHECK-LABEL: func.func private @negative_dynamic_stride(
// CHECK-SAME:   %[[SRC:[A-Za-z][A-Za-z0-9-]*]]: memref<1x108xf32>
func.func private @negative_dynamic_stride(%stride: index,
    %src : memref<1x108xf32>) {
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [108], strides: [%stride]
    : memref<1x108xf32> to memref<108xf32, strided<[?]>>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%idx_2]
    : memref<108xf32, strided<[?]>>
  return
}

// CHECK-LABEL: func.func private @negative_diff_non_unit_dims_order(
// CHECK-SAME:    %[[SRC:.*]]: memref<17x1x1x100xf32>) {
func.func private @negative_diff_non_unit_dims_order(
  %src : memref<17x1x1x100xf32>) {
  %idx = arith.constant 13 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [100, 17], strides: [1, 100]
    : memref<17x1x1x100xf32> to memref<100x17xf32, strided<[1, 100]>>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%idx, %idx] : memref<100x17xf32,
    strided<[1, 100]>>
  return
}

// CHECK-LABEL: func.func private @negative_diff_non_unit_size(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x1x100xf32>) {
func.func private @negative_diff_non_unit_size(
    %src : memref<1x1x1x100xf32>) {
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 99], strides: [99, 1]
    : memref<1x1x1x100xf32> to memref<1x99xf32>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_2] : memref<1x99xf32>
  return
}
