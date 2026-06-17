// RUN: mlir-opt -split-input-file -memref-elide-reinterpret-cast %s \
// RUN: | FileCheck %s

//===----------------------------------------------------------------------===//
// Positive tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func private @copy_to_strided_zero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x108xf32>
func.func private @copy_to_strided_zero_offset(%src : memref<1x1xf32>,
  %dst : memref<1x108xf32>) {
  /// reinterpret_cast removed
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1], strides: [1, 1]
    : memref<1x108xf32> to memref<1x1xf32>

  /// Ensure copy was replaced
  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]]] : memref<1x1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C0]]] : memref<1x108xf32>
  memref.copy %src, %reinterpret_cast
    : memref<1x1xf32> to memref<1x1xf32>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_to_strided_nonzero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x108xf32>
func.func private @copy_to_strided_nonzero_offset(%src : memref<1x1xf32>,
  %dst : memref<1x108xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [1], sizes: [1, 1], strides: [1, 1]
    : memref<1x108xf32>
      to memref<1x1xf32, strided<[1, 1], offset: 1>>

  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[C1:.*]] = arith.constant 1 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]]] : memref<1x1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C1]]] : memref<1x108xf32>
  memref.copy %src, %reinterpret_cast
    : memref<1x1xf32>
      to memref<1x1xf32, strided<[1, 1], offset: 1>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_to_strided_dynamic_offset(
// CHECK-SAME:   %[[OFF:.*]]: index
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x108xf32>
func.func private @copy_to_strided_dynamic_offset(%offset: index, %src : memref<1x1xf32>,
  %dst : memref<1x108xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [%offset], sizes: [1, 1], strides: [1, 1]
    : memref<1x108xf32>
      to memref<1x1xf32, strided<[1, 1], offset: ?>>

  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]]]
  // CHECK-SAME: : memref<1x1xf32>
  /// Dynamic offset used in store
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[OFF]]] : memref<1x108xf32>
  memref.copy %src, %reinterpret_cast
    : memref<1x1xf32>
      to memref<1x1xf32, strided<[1, 1], offset: ?>>
  // CHECK-NOT:  memref.copy
  return
}

// Dynamic strides are irrelevant because all strided memref indices are zero.
// CHECK-LABEL: func.func private @copy_to_strided_dynamic_stride(
// CHECK-SAME:   %[[STR0:[A-Za-z][A-Za-z0-9-]*]]: index
// CHECK-SAME:   %[[STR1:[A-Za-z][A-Za-z0-9-]*]]: index
// CHECK-SAME:   %[[SRC:[A-Za-z][A-Za-z0-9-]*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:[A-Za-z][A-Za-z0-9-]*]]: memref<1x108xf32>
func.func private @copy_to_strided_dynamic_stride(%stride0: index,
  %stride1: index, %src : memref<1x1xf32>, %dst : memref<1x108xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1], strides: [%stride0, %stride1]
    : memref<1x108xf32>
      to memref<1x1xf32, strided<[?, ?]>>

  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]]] : memref<1x1xf32>
  /// Dynamic offset used in store
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C0]]] : memref<1x108xf32>
  memref.copy %src, %reinterpret_cast
    : memref<1x1xf32>
      to memref<1x1xf32, strided<[?, ?]>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_to_strided_rank0(
// CHECK-SAME:   %[[SRC:.*]]: memref<f32>, %[[DST:.*]]: memref<f32>
func.func private @copy_to_strided_rank0(%src : memref<f32>, %dst : memref<f32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [], strides: []
    : memref<f32> to memref<f32>

  // CHECK-NOT:  memref.copy
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][] : memref<f32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][] : memref<f32>
  memref.copy %src, %reinterpret_cast : memref<f32> to memref<f32>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_to_strided_0d_2d_base(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x33x42xf32>
func.func private @copy_to_strided_0d_2d_base(
  %src : memref<1x1x1xf32>, %dst : memref<1x33x42xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1, 1], strides: [1, 1, 1]
    : memref<1x33x42xf32>
      to memref<1x1x1xf32>
  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x1x1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x33x42xf32>
  memref.copy %src, %reinterpret_cast
    : memref<1x1x1xf32> to memref<1x1x1xf32>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_to_strided_1d_vector_zero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x3x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x3x11xf32>
func.func private @copy_to_strided_1d_vector_zero_offset(
  %src : memref<1x3x1xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 3, 1], strides: [33, 11, 1]
    : memref<1x3x11xf32>
      to memref<1x3x1xf32, strided<[33, 11, 1]>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C3:.*]] = arith.constant 3 : index
  // CHECK:      scf.for %[[IDX:.*]] = %[[C0]] to %[[C3]] step %[[C1]] {
  // CHECK:        %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[IDX]], %[[C0]]] : memref<1x3x1xf32>
  // CHECK:        memref.store %[[VAL]], %[[DST]][%[[C0]], %[[IDX]], %[[C0]]] : memref<1x3x11xf32>
  // CHECK:      }
  memref.copy %src, %reinterpret_cast
    : memref<1x3x1xf32>
      to memref<1x3x1xf32, strided<[33, 11, 1]>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_to_strided_1d_vector_nonzero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x3x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x3x11xf32>
func.func private @copy_to_strided_1d_vector_nonzero_offset(
  %src : memref<1x3x1xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [10], sizes: [1, 3, 1], strides: [33, 11, 1]
    : memref<1x3x11xf32>
      to memref<1x3x1xf32, strided<[33, 11, 1], offset: 10>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG:  %[[C10:.*]] = arith.constant 10 : index
  // CHECK:      scf.for %[[IDX:.*]] = %[[C0]] to %[[C3]] step %[[C1]] {
  // CHECK:        %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[IDX]], %[[C0]]] : memref<1x3x1xf32>
  // CHECK:        memref.store %[[VAL]], %[[DST]][%[[C0]], %[[IDX]], %[[C10]]] : memref<1x3x11xf32>
  // CHECK:      }
  memref.copy %src, %reinterpret_cast
    : memref<1x3x1xf32>
      to memref<1x3x1xf32, strided<[33, 11, 1], offset: 10>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_to_strided_1d_vector_dynamic_offset_in_loop_dim(
// CHECK-SAME:   %[[OFF:.*]]: index
// CHECK-SAME:   %[[SRC:.*]]: memref<4xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<42xf32>
func.func private @copy_to_strided_1d_vector_dynamic_offset_in_loop_dim(
  %offset : index, %src : memref<4xf32>, %dst : memref<42xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [%offset], sizes: [4], strides: [1]
    : memref<42xf32> to memref<4xf32, strided<[1], offset: ?>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
  // CHECK:      scf.for %[[IDX:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
  // CHECK:        %[[DST_IDX:.*]] = arith.addi %[[OFF]], %[[IDX]] : index
  // CHECK:        %[[VAL:.*]] = memref.load %[[SRC]][%[[IDX]]] : memref<4xf32>
  // CHECK:        memref.store %[[VAL]], %[[DST]][%[[DST_IDX]]] : memref<42xf32>
  // CHECK:      }
  memref.copy %src, %reinterpret_cast
    : memref<4xf32> to memref<4xf32, strided<[1], offset: ?>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_to_strided_1d_vector_dynamic_offset_not_in_loop_dim(
// CHECK-SAME:   %[[OFF:.*]]: index
// CHECK-SAME:   %[[SRC:.*]]: memref<1x3x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x3x11xf32>
func.func private @copy_to_strided_1d_vector_dynamic_offset_not_in_loop_dim(
  %offset : index, %src : memref<1x3x1xf32>,
  %dst : memref<1x3x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [%offset], sizes: [1, 3, 1], strides: [33, 11, 1]
    : memref<1x3x11xf32>
      to memref<1x3x1xf32, strided<[33, 11, 1], offset: ?>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C3:.*]] = arith.constant 3 : index
  // CHECK:      scf.for %[[IDX:.*]] = %[[C0]] to %[[C3]] step %[[C1]] {
  // CHECK:        %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[IDX]], %[[C0]]] : memref<1x3x1xf32>
  // CHECK:        memref.store %[[VAL]], %[[DST]][%[[C0]], %[[IDX]], %[[OFF]]] : memref<1x3x11xf32>
  // CHECK:      }
  memref.copy %src, %reinterpret_cast
    : memref<1x3x1xf32>
      to memref<1x3x1xf32, strided<[33, 11, 1], offset: ?>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_to_strided_2d_vector_nonzero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x3x4xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x3x11xf32>
func.func private @copy_to_strided_2d_vector_nonzero_offset(
  %src : memref<1x3x4xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [7], sizes: [1, 3, 4], strides: [33, 11, 1]
    : memref<1x3x11xf32>
      to memref<1x3x4xf32, strided<[33, 11, 1], offset: 7>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG:  %[[C7:.*]] = arith.constant 7 : index
  // CHECK:      scf.for %[[IDX0:.*]] = %[[C0]] to %[[C3]] step %[[C1]] {
  // CHECK:        scf.for %[[IDX1:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
  // CHECK:          %[[DST_IDX:.*]] = arith.addi %[[C7]], %[[IDX1]] : index
  // CHECK:          %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[IDX0]], %[[IDX1]]] : memref<1x3x4xf32>
  // CHECK:          memref.store %[[VAL]], %[[DST]][%[[C0]], %[[IDX0]], %[[DST_IDX]]] : memref<1x3x11xf32>
  // CHECK:        }
  // CHECK:      }
  memref.copy %src, %reinterpret_cast
    : memref<1x3x4xf32>
      to memref<1x3x4xf32, strided<[33, 11, 1], offset: 7>>
  // CHECK-NOT:  memref.copy
  return
}

//===----------------------------------------------------------------------===//
// Negative tests (must NOT rewrite)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func private @negative_copy_to_strided_non_identity_base(
func.func private @negative_copy_to_strided_non_identity_base(%src: memref<1x1xf32>,
  %dst: memref<8x1xf32, strided<[10, 2]>>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg1
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [6], sizes: [1, 1], strides: [11, 80]
    : memref<8x1xf32, strided<[10, 2]>>
      to memref<1x1xf32, strided<[11, 80], offset: 6>>

  // CHECK:      memref.copy %arg0, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %reinterpret_cast
    : memref<1x1xf32> to memref<1x1xf32, strided<[11, 80], offset: 6>>

  return
}

// CHECK-LABEL: func.func private @negative_copy_to_strided_rank_change(
func.func private @negative_copy_to_strided_rank_change(%src : memref<2x3xf32>,
  %dst : memref<6xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg1
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [2, 3], strides: [3, 1]
    : memref<6xf32> to memref<2x3xf32>

  // CHECK:      memref.copy %arg0, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %reinterpret_cast
    : memref<2x3xf32> to memref<2x3xf32>
  return
}

// CHECK-LABEL: func.func private @negative_copy_to_strided_dynamic_copy_source_shape(
func.func private @negative_copy_to_strided_dynamic_copy_source_shape(%src : memref<?xf32>,
  %dst : memref<4xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg1
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [4], strides: [1]
    : memref<4xf32> to memref<4xf32>

  // CHECK:      memref.copy %arg0, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %reinterpret_cast
    : memref<?xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func private @negative_copy_to_strided_dynamic_rc_shapes(
func.func private @negative_copy_to_strided_dynamic_rc_shapes(%dim : index,
  %src : memref<4xf32>, %dst : memref<?xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg2
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [%dim], strides: [1]
    : memref<?xf32> to memref<?xf32, strided<[1]>>

  // CHECK:      memref.copy %arg1, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %reinterpret_cast
    : memref<4xf32> to memref<?xf32, strided<[1]>>
  return
}

// CHECK-LABEL: func.func private @negative_copy_to_strided_dynamic_offset_multi_dim_base(
func.func private @negative_copy_to_strided_dynamic_offset_multi_dim_base(
  %offset : index, %src : memref<1x1xf32>, %dst : memref<4x8xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg2
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [%offset], sizes: [1, 1], strides: [8, 1]
    : memref<4x8xf32> to memref<1x1xf32, strided<[8, 1], offset: ?>>

  // CHECK:      memref.copy %arg1, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %reinterpret_cast
    : memref<1x1xf32>
      to memref<1x1xf32, strided<[8, 1], offset: ?>>
  return
}

// CHECK-LABEL: func.func private @negative_copy_to_strided_2d_dynamic_offset(
func.func private @negative_copy_to_strided_2d_dynamic_offset(
  %offset : index, %src : memref<1x3x4xf32>,
  %dst : memref<1x3x11xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg2
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [%offset], sizes: [1, 3, 4], strides: [33, 11, 1]
    : memref<1x3x11xf32>
      to memref<1x3x4xf32, strided<[33, 11, 1], offset: ?>>

  // CHECK:      memref.copy %arg1, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %reinterpret_cast
    : memref<1x3x4xf32>
      to memref<1x3x4xf32, strided<[33, 11, 1], offset: ?>>
  return
}

/// Non-unit copied dimension needs stride-based address computation.
// CHECK-LABEL: func.func private @negative_copy_to_strided_dynamic_rc_stride(
func.func private @negative_copy_to_strided_dynamic_rc_stride(%stride : index,
  %src : memref<1x3x1xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg2
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 3, 1], strides: [33, %stride, 1]
    : memref<1x3x11xf32>
      to memref<1x3x1xf32, strided<[33, ?, 1]>>

  // CHECK:      memref.copy %arg1, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %reinterpret_cast
    : memref<1x3x1xf32>
      to memref<1x3x1xf32, strided<[33, ?, 1]>>
  return
}

// CHECK-LABEL: func.func private @negative_plain_copy(
func.func private @negative_plain_copy(%src : memref<1x1xf32>,
  %dst : memref<1x1xf32>) {
  // CHECK:      memref.copy %arg0, %arg1
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %dst
  : memref<1x1xf32> to memref<1x1xf32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Positive tests
//===----------------------------------------------------------------------===//

/// For rank-1 MemRefs, expansion/collapsing may be considered on either side.

// CHECK-LABEL: func.func private @expand_scalar(
// CHECK-SAME:    %[[SRC:.*]]: memref<1xi64>) {
func.func private @expand_scalar(%src : memref<1xi64>) {
  // CHECK:       %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 1, 1], strides: [1, 1, 1]
    : memref<1xi64> to memref<1x1x1xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[C0]]] : memref<1xi64>
  %0 = memref.load %reinterpret_cast[%c0, %c0, %c0] : memref<1x1x1xi64>
  return
}

// CHECK-LABEL: func.func private @collapse_scalar(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x1xi64>) {
func.func private @collapse_scalar(%src : memref<1x1x1xi64>) {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 1], strides: [1, 1]
    : memref<1x1x1xi64> to memref<1x1xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[C0_0]], %[[C0]], %[[C0]]] : memref<1x1x1xi64>
  %0 = memref.load %reinterpret_cast[%c0, %c0] : memref<1x1xi64>
  return
}

// CHECK-LABEL: func.func private @expand_left_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<999xi64>) {
func.func private @expand_left_vector(%src : memref<999xi64>) {
  // CHECK:       %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 1, 999], strides: [999, 999, 1]
    : memref<999xi64> to memref<1x1x999xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[C0]]] : memref<999xi64>
  %0 = memref.load %reinterpret_cast[%c0, %c0, %c0] : memref<1x1x999xi64>
  return
}

// CHECK-LABEL: func.func private @expand_left_vector_dynamic_index(
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<999xi64>) {
func.func private @expand_left_vector_dynamic_index(%i : index,
    %src : memref<999xi64>) {
  // CHECK:       %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 1, 999], strides: [999, 999, 1]
    : memref<999xi64> to memref<1x1x999xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[I]]] : memref<999xi64>
  %0 = memref.load %reinterpret_cast[%c0, %c0, %i] : memref<1x1x999xi64>
  return
}

// CHECK-LABEL: func.func private @collapse_left_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x999xi64>) {
func.func private @collapse_left_vector(%src : memref<1x1x999xi64>) {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [999], strides: [1]
    : memref<1x1x999xi64> to memref<999xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]], %[[C1]]] : memref<1x1x999xi64>
  %0 = memref.load %reinterpret_cast[%c1] : memref<999xi64>
  return
}

// CHECK-LABEL: func.func private @partial_expand_left_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x999xf32>) {
func.func private @partial_expand_left_vector(
    %src : memref<1x999xf32>) {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 1, 999], strides: [999, 999, 1]
    : memref<1x999xf32> to memref<1x1x999xf32>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[C0]], %[[C1]]] : memref<1x999xf32>
  %0 = memref.load %reinterpret_cast[%c0, %c0, %c1]
    : memref<1x1x999xf32>
  return
}

// CHECK-LABEL: func.func private @partial_collapse_left_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x999xf32>) {
func.func private @partial_collapse_left_vector(
    %src : memref<1x1x999xf32>) {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 999], strides: [999, 1]
    : memref<1x1x999xf32> to memref<1x999xf32>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[C0_0]], %[[C0]], %[[C1]]] : memref<1x1x999xf32>
  %0 = memref.load %reinterpret_cast[%c0, %c1] : memref<1x999xf32>
  return
}

// CHECK-LABEL: func.func private @expand_right_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<999xi64>) {
func.func private @expand_right_vector(%src : memref<999xi64>) {
  // CHECK:       %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [999, 1, 1], strides: [1, 999, 999]
    : memref<999xi64> to memref<999x1x1xi64, strided<[1, 999, 999]>>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[C0]]] : memref<999xi64>
  %0 = memref.load %reinterpret_cast[%c0, %c0, %c0] : memref<999x1x1xi64,
    strided<[1, 999, 999]>>
  return
}

// CHECK-LABEL: func.func private @collapse_right_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<999x1x1xi64>) {
func.func private @collapse_right_vector(%src : memref<999x1x1xi64>) {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [999], strides: [1]
      : memref<999x1x1xi64> to memref<999xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[C1]], %[[C0]], %[[C0]]] : memref<999x1x1xi64>
  %0 = memref.load %reinterpret_cast[%c1] : memref<999xi64>
  return
}

// CHECK-LABEL: func.func private @collapse_right_vector_dynamic_index(
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<999x1x1xi64>) {
func.func private @collapse_right_vector_dynamic_index(%i : index,
    %src : memref<999x1x1xi64>) {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [999], strides: [1]
    : memref<999x1x1xi64> to memref<999xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[I]], %[[C0]], %[[C0]]] : memref<999x1x1xi64>
  %0 = memref.load %reinterpret_cast[%i] : memref<999xi64>
  return
}

// CHECK-LABEL: func.func private @partial_expand_right_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<999x1xf32>) {
func.func private @partial_expand_right_vector(
    %src : memref<999x1xf32>) {
  // CHECK:       %[[C0:.*]] = arith.constant 0 : index
  // CHECK:       %[[C1:.*]] = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [999, 1, 1], strides: [1, 999, 999]
    : memref<999x1xf32> to memref<999x1x1xf32, strided<[1, 999, 999]>>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[C1]], %[[C0]]] : memref<999x1xf32>
  %0 = memref.load %reinterpret_cast[%c1, %c0, %c0]
    : memref<999x1x1xf32, strided<[1, 999, 999]>>
  return
}

// CHECK-LABEL: func.func private @partial_collapse_right_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<999x1x1xf32>) {
func.func private @partial_collapse_right_vector(
    %src : memref<999x1x1xf32>) {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [999, 1], strides: [1, 999]
    : memref<999x1x1xf32> to memref<999x1xf32, strided<[1, 999]>>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[C1]], %[[C0]], %[[C0_0]]] : memref<999x1x1xf32>
  %0 = memref.load %reinterpret_cast[%c1, %c0] : memref<999x1xf32,
    strided<[1, 999]>>
  return
}

//===----------------------------------------------------------------------===//
// Negative tests (must NOT rewrite)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func private @negative_nonzero_offset(
// CHECK-SAME:    %[[SRC:.*]]: memref<1xi64>) {
func.func private @negative_nonzero_offset(
    %src : memref<1xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [1], sizes: [1, 1, 1], strides: [1, 1, 1]
    : memref<1xi64> to memref<1x1x1xi64, strided<[1, 1, 1], offset: 1>>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%c0, %c0, %c1]
    : memref<1x1x1xi64, strided<[1, 1, 1], offset: 1>>
  return
}

// CHECK-LABEL: func.func private @negative_dynamic_shape(
// CHECK-SAME:   %[[SRC:[A-Za-z][A-Za-z0-9-]*]]: memref<?xi64>
func.func private @negative_dynamic_shape(%dim : index, %i : index,
    %src : memref<?xi64>) {
  %c0 = arith.constant 0 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, %dim], strides: [1, 1]
    : memref<?xi64> to memref<1x?xi64>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%c0, %i] : memref<1x?xi64>
  return
}

// CHECK-LABEL: func.func private @negative_dynamic_stride(
// CHECK-SAME:   %[[SRC:[A-Za-z][A-Za-z0-9-]*]]: memref<1x108xi64>
func.func private @negative_dynamic_stride(%stride0: index,
    %stride1: index, %src : memref<1x108xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 1], strides: [%stride0, %stride1]
    : memref<1x108xi64> to memref<1x1xi64, strided<[?, ?]>>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%c0, %c1]
    : memref<1x1xi64, strided<[?, ?]>>
  return
}

// CHECK-LABEL: func.func private @negative_multiple_non_unit_dims(
// CHECK-SAME:    %[[SRC:.*]]: memref<2x1x1x100xf32>) {
func.func private @negative_multiple_non_unit_dims(
  %src : memref<2x1x1x100xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [2, 100], strides: [100, 1]
    : memref<2x1x1x100xf32> to memref<2x100xf32>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%c0, %c1] : memref<2x100xf32>
  return
}

// CHECK-LABEL: func.func private @negative_inner_non_unit_dims(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x1x100xf32>) {
func.func private @negative_inner_non_unit_dims(
    %src : memref<1x1x1x100xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 100, 1], strides: [100, 1, 100]
    : memref<1x1x1x100xf32> to memref<1x100x1xf32, strided<[100, 1, 100]>>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%c0, %c1, %c0] : memref<1x100x1xf32,
    strided<[100, 1, 100]>>
  return
}

// CHECK-LABEL: func.func private @negative_diff_non_unit_boundary(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x1x100xf32>) {
func.func private @negative_diff_non_unit_boundary(
    %src : memref<1x1x1x100xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [100, 1, 1], strides: [1, 100, 100]
    : memref<1x1x1x100xf32> to memref<100x1x1xf32, strided<[1, 100, 100]>>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%c1, %c0, %c0] : memref<100x1x1xf32,
    strided<[1, 100, 100]>>
  return
}

// CHECK-LABEL: func.func private @negative_diff_non_unit_size(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x1x100xf32>) {
func.func private @negative_diff_non_unit_size(
    %src : memref<1x1x1x100xf32>) {
  %c0 = arith.constant 0 : index
  %c98 = arith.constant 98 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 99], strides: [99, 1]
    : memref<1x1x1x100xf32> to memref<1x99xf32>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%c0, %c98] : memref<1x99xf32>
  return
}
