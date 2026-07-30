// RUN: mlir-opt -memref-elide-reinterpret-cast %s | FileCheck %s

/// Tests for the CopyToLoadAndStore pattern
/// to show how reinterpret_cast is elided.

//===----------------------------------------------------------------------===//
// Scalar (0D) copy
//
// All RC result dimensions are unit (1) =>
//   RC result strides do not affect copy destination address and are ignored.
//===----------------------------------------------------------------------===//

// The destination is effectively a scalar within a MemRef with rank == 0 
// CHECK-LABEL: func.func private @copy_scalar_into_0D_strided_zero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<f32>, %[[DST:.*]]: memref<f32>
func.func private @copy_scalar_into_0D_strided_zero_offset(%src : memref<f32>, %dst : memref<f32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [], strides: []
    : memref<f32> to memref<f32>

  // CHECK-NOT:  memref.copy
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][] : memref<f32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][] : memref<f32>
  memref.copy %src, %rc : memref<f32> to memref<f32>
  // CHECK-NOT:  memref.copy
  return
}

/// The destination is effectively a 1D array within a MemRef with rank >= 1 
// CHECK-LABEL: func.func private @copy_scalar_into_1D_strided_zero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x108xf32>
func.func private @copy_scalar_into_1D_strided_zero_offset(%src : memref<1x1xf32>,
  %dst : memref<1x108xf32>) {
  /// reinterpret_cast removed
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1], strides: [1, 1]
    : memref<1x108xf32> to memref<1x1xf32>

  /// Ensure copy was replaced
  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]]] : memref<1x1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C0]]] : memref<1x108xf32>
  memref.copy %src, %rc
    : memref<1x1xf32> to memref<1x1xf32>
  // CHECK-NOT:  memref.copy
  return
}

/// Reject non-identity layout rc source strides
// CHECK-LABEL: func.func private @negative_copy_scalar_into_1D_strided_zero_offset_base_non_identity_layout(
func.func private @negative_copy_scalar_into_1D_strided_zero_offset_base_non_identity_layout(
  %src: memref<1x1xf32>, %dst: memref<1x108xf32, strided<[54, 2]>>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg1
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1], strides: [54, 2]
    : memref<1x108xf32, strided<[54, 2]>>
      to memref<1x1xf32, strided<[54, 2]>>

  // CHECK:      memref.copy %arg0, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %rc
    : memref<1x1xf32> to memref<1x1xf32, strided<[54, 2]>>

  return
}

// CHECK-LABEL: func.func private @copy_scalar_into_1D_strided_nonzero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x108xf32>
func.func private @copy_scalar_into_1D_strided_nonzero_offset(%src : memref<1x1xf32>,
  %dst : memref<1x108xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [1], sizes: [1, 1], strides: [1, 1]
    : memref<1x108xf32>
      to memref<1x1xf32, strided<[1, 1], offset: 1>>

  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[OFF:.*]] = arith.constant 1 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]]] : memref<1x1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[OFF]]] : memref<1x108xf32>
  memref.copy %src, %rc
    : memref<1x1xf32>
      to memref<1x1xf32, strided<[1, 1], offset: 1>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_scalar_into_1D_strided_dynamic_offset(
// CHECK-SAME:   %[[OFF:.*]]: index
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x108xf32>
func.func private @copy_scalar_into_1D_strided_dynamic_offset(%offset: index, %src : memref<1x1xf32>,
  %dst : memref<1x108xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [%offset], sizes: [1, 1], strides: [1, 1]
    : memref<1x108xf32>
      to memref<1x1xf32, strided<[1, 1], offset: ?>>

  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]]]
  // CHECK-SAME: : memref<1x1xf32>
  /// Dynamic offset used in store
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[OFF]]] : memref<1x108xf32>
  memref.copy %src, %rc
    : memref<1x1xf32>
      to memref<1x1xf32, strided<[1, 1], offset: ?>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_scalar_into_1D_strided_zero_offset_non_identity_stride(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x108xf32>
func.func private @copy_scalar_into_1D_strided_zero_offset_non_identity_stride(
  %src : memref<1x1xf32>, %dst : memref<1x108xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1], strides: [54, 2]
    : memref<1x108xf32> to memref<1x1xf32, strided<[54, 2]>>

  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]]] : memref<1x1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C0]]] : memref<1x108xf32>
  memref.copy %src, %reinterpret_cast
    : memref<1x1xf32> to memref<1x1xf32, strided<[54, 2]>>
  return
}

// CHECK-LABEL: func.func private @copy_scalar_into_1D_strided_zero_offset_dynamic_stride(
// CHECK-SAME:   %[[STR0:[A-Za-z][A-Za-z0-9-]*]]: index
// CHECK-SAME:   %[[STR1:[A-Za-z][A-Za-z0-9-]*]]: index
// CHECK-SAME:   %[[SRC:[A-Za-z][A-Za-z0-9-]*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:[A-Za-z][A-Za-z0-9-]*]]: memref<1x108xf32>
func.func private @copy_scalar_into_1D_strided_zero_offset_dynamic_stride(%stride0: index,
  %stride1: index, %src : memref<1x1xf32>, %dst : memref<1x108xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1], strides: [%stride0, %stride1]
    : memref<1x108xf32>
      to memref<1x1xf32, strided<[?, ?]>>

  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]]] : memref<1x1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C0]]] : memref<1x108xf32>
  memref.copy %src, %rc
    : memref<1x1xf32>
      to memref<1x1xf32, strided<[?, ?]>>
  // CHECK-NOT:  memref.copy
  return
}

/// The destination is effectively a 2D array within a MemRef with rank >= 2 
// CHECK-LABEL: func.func private @copy_scalar_into_2D_strided_zero_offset_non_identity_stride(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x3x11xf32>
func.func private @copy_scalar_into_2D_strided_zero_offset_non_identity_stride(
  %src : memref<1x1x1xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1, 1], strides: [1, 1, 1]
    : memref<1x3x11xf32>
      to memref<1x1x1xf32>
  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x1x1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x3x11xf32>
  memref.copy %src, %rc
    : memref<1x1x1xf32> to memref<1x1x1xf32>
  // CHECK-NOT:  memref.copy
  return
}

/// %dst has identity strides [33, 11, 1].
/// Offset 10 delinearizes as:
///   dim 0: 10 / 33 = 0, remainder 10
///   dim 1: 10 / 11 = 0, remainder 10
///   dim 2: 10 /  1 = 10, remainder 0
/// Therefore the scalar is stored at %dst[0, 0, 10].
// CHECK-LABEL: func.func private @copy_scalar_into_2D_scalar_strided_nonzero_offset_delinearized_v1(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x3x11xf32>
func.func private @copy_scalar_into_2D_scalar_strided_nonzero_offset_delinearized_v1(
    %src : memref<1x1x1xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [10], sizes: [1, 1, 1], strides: [1, 1, 1]
    : memref<1x3x11xf32>
      to memref<1x1x1xf32, strided<[1, 1, 1], offset: 10>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[OFF:.*]] = arith.constant 10 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x1x1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C0]], %[[OFF]]] : memref<1x3x11xf32>
  memref.copy %src, %rc
    : memref<1x1x1xf32>
      to memref<1x1x1xf32, strided<[1, 1, 1], offset: 10>>
  // CHECK-NOT:  memref.copy
  return
}

/// %dst has identity strides [33, 11, 1].
/// Offset 23 delinearizes as:
///   dim 0: 23 / 33 = 0, remainder 23
///   dim 1: 23 / 11 = 2, remainder 1
///   dim 2:  1 /  1 = 1, remainder 0
/// Therefore the scalar is stored at %dst[0, 2, 1].
// CHECK-LABEL: func.func private @copy_scalar_into_2D_scalar_strided_nonzero_offset_delinearized_v2(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x3x11xf32>
func.func private @copy_scalar_into_2D_scalar_strided_nonzero_offset_delinearized_v2(
    %src : memref<1x1x1xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [23], sizes: [1, 1, 1], strides: [1, 1, 1]
    : memref<1x3x11xf32>
      to memref<1x1x1xf32, strided<[1, 1, 1], offset: 23>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C2:.*]] = arith.constant 2 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x1x1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C2]], %[[C1]]] : memref<1x3x11xf32>
  memref.copy %src, %rc
    : memref<1x1x1xf32>
      to memref<1x1x1xf32, strided<[1, 1, 1], offset: 23>>
  // CHECK-NOT:  memref.copy
  return
}

/// rc result dynamic offset:
///    supported only for effectively-1D rc source
///    (runtime delinearization not implemented)
// CHECK-LABEL: func.func private @negative_copy_scalar_into_2D_strided_dynamic_offset(
func.func private @negative_copy_scalar_into_2D_strided_dynamic_offset(
  %offset : index, %src : memref<1x1x1xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg2
  %rc = memref.reinterpret_cast %dst
    to offset: [%offset], sizes: [1, 1, 1], strides: [33, 11, 1]
    : memref<1x3x11xf32> to memref<1x1x1xf32, strided<[33, 11, 1], offset: ?>>

  // CHECK:      memref.copy %arg1, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %rc
    : memref<1x1x1xf32>
      to memref<1x1x1xf32, strided<[33, 11, 1], offset: ?>>
  return
}

//===----------------------------------------------------------------------===//
// Non-scalar (ND) copy
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func private @copy_1D_into_1D_strided_zero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<4xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<108xf32>
func.func private @copy_1D_into_1D_strided_zero_offset(
  %src : memref<4xf32>, %dst : memref<108xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [4], strides: [1]
    : memref<108xf32> to memref<4xf32, strided<[1]>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[UB:.*]] = arith.constant 4 : index
  // CHECK:      scf.for %[[IDX:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
  // CHECK:        %[[VAL:.*]] = memref.load %[[SRC]][%[[IDX]]] : memref<4xf32>
  // CHECK:        memref.store %[[VAL]], %[[DST]][%[[IDX]]] : memref<108xf32>
  // CHECK:      }
  memref.copy %src, %rc
    : memref<4xf32> to memref<4xf32, strided<[1]>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_1D_into_2D_strided_zero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x3x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x3x11xf32>
func.func private @copy_1D_into_2D_strided_zero_offset(
  %src : memref<1x3x1xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 3, 1], strides: [33, 11, 1]
    : memref<1x3x11xf32>
      to memref<1x3x1xf32, strided<[33, 11, 1]>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[UB:.*]] = arith.constant 3 : index
  // CHECK:      scf.for %[[IDX:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
  // CHECK:        %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[IDX]], %[[C0]]] : memref<1x3x1xf32>
  // CHECK:        memref.store %[[VAL]], %[[DST]][%[[C0]], %[[IDX]], %[[C0]]] : memref<1x3x11xf32>
  // CHECK:      }
  memref.copy %src, %rc
    : memref<1x3x1xf32>
      to memref<1x3x1xf32, strided<[33, 11, 1]>>
  // CHECK-NOT:  memref.copy
  return
}

/// Copied non-unit dimension is the right-most dimension, therefore the loop indices are placed there.
// CHECK-LABEL: func.func private @copy_1D_into_2D_strided_zero_offset_loop_trailing_dim(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1x11xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x3x11xf32>
func.func private @copy_1D_into_2D_strided_zero_offset_loop_trailing_dim(
  %src : memref<1x1x11xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1, 11], strides: [33, 11, 1]
    : memref<1x3x11xf32>
      to memref<1x1x11xf32, strided<[33, 11, 1]>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[UB:.*]] = arith.constant 11 : index
  // CHECK:      scf.for %[[IDX:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
  // CHECK:        %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]], %[[IDX]]] : memref<1x1x11xf32>
  // CHECK:        memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C0]], %[[IDX]]] : memref<1x3x11xf32>
  // CHECK:      }
  memref.copy %src, %rc
    : memref<1x1x11xf32>
      to memref<1x1x11xf32, strided<[33, 11, 1]>>
  // CHECK-NOT:  memref.copy
  return
}

/// Offset delinearized to [0, 0, 10], therefore is only
/// added to the trailing source dimension.
// CHECK-LABEL: func.func private @copy_1D_into_2D_strided_nonzero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x3x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x3x11xf32>
func.func private @copy_1D_into_2D_strided_nonzero_offset(
  %src : memref<1x3x1xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [10], sizes: [1, 3, 1], strides: [33, 11, 1]
    : memref<1x3x11xf32>
      to memref<1x3x1xf32, strided<[33, 11, 1], offset: 10>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[UB:.*]] = arith.constant 3 : index
  // CHECK-DAG:  %[[OFF:.*]] = arith.constant 10 : index
  // CHECK:      scf.for %[[IDX:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
  // CHECK:        %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[IDX]], %[[C0]]] : memref<1x3x1xf32>
  // CHECK:        memref.store %[[VAL]], %[[DST]][%[[C0]], %[[IDX]], %[[OFF]]] : memref<1x3x11xf32>
  // CHECK:      }
  memref.copy %src, %rc
    : memref<1x3x1xf32>
      to memref<1x3x1xf32, strided<[33, 11, 1], offset: 10>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @negative_copy_1D_into_2D_strided_dynamic_offset(
func.func private @negative_copy_1D_into_2D_strided_dynamic_offset(
  %offset : index, %src : memref<1x3x1xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg2
  %rc = memref.reinterpret_cast %dst
    to offset: [%offset], sizes: [1, 3, 1], strides: [33, 11, 1]
    : memref<1x3x11xf32>
      to memref<1x3x1xf32, strided<[33, 11, 1], offset: ?>>

  // CHECK:      memref.copy %arg1, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %rc
    : memref<1x3x1xf32>
      to memref<1x3x1xf32, strided<[33, 11, 1], offset: ?>>
  return
}

/// Reject rc result strides that are not equal to rc source identity strides.
/// (non-unit copied dimension needs stride-based address computation)
// CHECK-LABEL: func.func private @negative_copy_1D_into_2D_strided_zero_offset_non_identity_strides(
func.func private @negative_copy_1D_into_2D_strided_zero_offset_non_identity_strides(
  %src : memref<1x3x1xf32>, %dst : memref<1x3x4xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg1
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 3, 1], strides: [12, 4, 4]
    : memref<1x3x4xf32>
      to memref<1x3x1xf32, strided<[12, 4, 4]>>

  // CHECK:      memref.copy %arg0, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %rc
    : memref<1x3x1xf32>
      to memref<1x3x1xf32, strided<[12, 4, 4]>>
  return
}

/// Reject dynamic rc result strides.
// CHECK-LABEL: func.func private @negative_copy_1D_into_2D_strided_zero_offset_dynamic_stride(
func.func private @negative_copy_1D_into_2D_strided_zero_offset_dynamic_stride(%stride : index,
  %src : memref<1x3x1xf32>, %dst : memref<1x3x4xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg2
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 3, 1], strides: [12, %stride, 1]
    : memref<1x3x4xf32>
      to memref<1x3x1xf32, strided<[12, ?, 1]>>

  // CHECK:      memref.copy %arg1, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %rc
    : memref<1x3x1xf32>
      to memref<1x3x1xf32, strided<[12, ?, 1]>>
  return
}

// CHECK-LABEL: func.func private @negative_copy_1D_into_2D_multiple_truncated_dims(
func.func private @negative_copy_1D_into_2D_multiple_truncated_dims(
  %src : memref<1x3x1xf32>, %dst : memref<1x4x11xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg1
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 3, 1], strides: [44, 11, 1]
    : memref<1x4x11xf32>
      to memref<1x3x1xf32, strided<[44, 11, 1]>>

  // CHECK:      memref.copy %arg0, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %rc
    : memref<1x3x1xf32>
      to memref<1x3x1xf32, strided<[44, 11, 1]>>
  return
}

// CHECK-LABEL: func.func private @negative_copy_into_strided_no_truncated_dims(
func.func private @negative_copy_into_strided_no_truncated_dims(%src : memref<3x4xf32>,
  %dst : memref<3x4xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg1
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [3, 4], strides: [12, 1]
    : memref<3x4xf32> to memref<3x4xf32, strided<[12, 1]>>

  // CHECK:      memref.copy %arg0, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %rc
    : memref<3x4xf32> to memref<3x4xf32, strided<[12, 1]>>
  return
}

// CHECK-LABEL: func.func private @copy_2D_into_2D_strided_zero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x3x4xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x3x11xf32>
func.func private @copy_2D_into_2D_strided_zero_offset(
  %src : memref<1x3x4xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 3, 4], strides: [33, 11, 1]
    : memref<1x3x11xf32>
      to memref<1x3x4xf32, strided<[33, 11, 1]>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[UB0:.*]] = arith.constant 3 : index
  // CHECK-DAG:  %[[UB1:.*]] = arith.constant 4 : index
  // CHECK:      scf.for %[[IDX0:.*]] = %[[C0]] to %[[UB0]] step %[[C1]] {
  // CHECK:        scf.for %[[IDX1:.*]] = %[[C0]] to %[[UB1]] step %[[C1]] {
  // CHECK:          %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[IDX0]], %[[IDX1]]] : memref<1x3x4xf32>
  // CHECK:          memref.store %[[VAL]], %[[DST]][%[[C0]], %[[IDX0]], %[[IDX1]]] : memref<1x3x11xf32>
  // CHECK:        }
  // CHECK:      }
  memref.copy %src, %rc
    : memref<1x3x4xf32>
      to memref<1x3x4xf32, strided<[33, 11, 1]>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_2D_into_2D_strided_nonzero_offset_delinearized_v1(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x3x4xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x3x11xf32>
func.func private @copy_2D_into_2D_strided_nonzero_offset_delinearized_v1(
  %src : memref<1x3x4xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [6], sizes: [1, 3, 4], strides: [33, 11, 1]
    : memref<1x3x11xf32>
      to memref<1x3x4xf32, strided<[33, 11, 1], offset: 6>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[UB0:.*]] = arith.constant 3 : index
  // CHECK-DAG:  %[[UB1:.*]] = arith.constant 4 : index
  // CHECK-DAG:  %[[OFF:.*]] = arith.constant 6 : index
  // CHECK:      scf.for %[[IDX0:.*]] = %[[C0]] to %[[UB0]] step %[[C1]] {
  // CHECK:        scf.for %[[IDX1:.*]] = %[[C0]] to %[[UB1]] step %[[C1]] {
  // CHECK:          %[[DST_IDX:.*]] = arith.addi %[[OFF]], %[[IDX1]] : index
  // CHECK:          %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[IDX0]], %[[IDX1]]] : memref<1x3x4xf32>
  // CHECK:          memref.store %[[VAL]], %[[DST]][%[[C0]], %[[IDX0]], %[[DST_IDX]]] : memref<1x3x11xf32>
  // CHECK:        }
  // CHECK:      }
  memref.copy %src, %rc
    : memref<1x3x4xf32>
      to memref<1x3x4xf32, strided<[33, 11, 1], offset: 6>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_2D_into_2D_strided_nonzero_offset_delinearized_v2(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x3x11xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x10x11xf32>
func.func private @copy_2D_into_2D_strided_nonzero_offset_delinearized_v2(
  %src : memref<1x3x11xf32>, %dst : memref<1x10x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [44], sizes: [1, 3, 11], strides: [110, 11, 1]
    : memref<1x10x11xf32>
      to memref<1x3x11xf32, strided<[110, 11, 1], offset: 44>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[UB0:.*]] = arith.constant 3 : index
  // CHECK-DAG:  %[[UB1:.*]] = arith.constant 11 : index
  // CHECK-DAG:  %[[OFF:.*]] = arith.constant 4 : index
  // CHECK:      scf.for %[[IDX0:.*]] = %[[C0]] to %[[UB0]] step %[[C1]] {
  // CHECK:        %[[DST_IDX:.*]] = arith.addi %[[OFF]], %[[IDX0]] : index
  // CHECK:        scf.for %[[IDX1:.*]] = %[[C0]] to %[[UB1]] step %[[C1]] {
  // CHECK:          %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[IDX0]], %[[IDX1]]] : memref<1x3x11xf32>
  // CHECK:          memref.store %[[VAL]], %[[DST]][%[[C0]], %[[DST_IDX]], %[[IDX1]]] : memref<1x10x11xf32>
  // CHECK:        }
  // CHECK:      }
  memref.copy %src, %rc
    : memref<1x3x11xf32>
      to memref<1x3x11xf32, strided<[110, 11, 1], offset: 44>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_2D_into_3D_strided_zero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<3x1x4x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<3x1x4x11xf32>
func.func private @copy_2D_into_3D_strided_zero_offset(
  %src : memref<3x1x4x1xf32>, %dst : memref<3x1x4x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [3, 1, 4, 1], strides: [44, 44, 11, 1]
    : memref<3x1x4x11xf32>
      to memref<3x1x4x1xf32, strided<[44, 44, 11, 1]>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[UB0:.*]] = arith.constant 3 : index
  // CHECK-DAG:  %[[UB1:.*]] = arith.constant 4 : index
  // CHECK:      scf.for %[[IDX0:.*]] = %[[C0]] to %[[UB0]] step %[[C1]] {
  // CHECK:        scf.for %[[IDX1:.*]] = %[[C0]] to %[[UB1]] step %[[C1]] {
  // CHECK:          %[[VAL:.*]] = memref.load %[[SRC]][%[[IDX0]], %[[C0]], %[[IDX1]], %[[C0]]] : memref<3x1x4x1xf32>
  // CHECK:          memref.store %[[VAL]], %[[DST]][%[[IDX0]], %[[C0]], %[[IDX1]], %[[C0]]] : memref<3x1x4x11xf32>
  // CHECK:        }
  // CHECK:      }
  memref.copy %src, %rc
    : memref<3x1x4x1xf32>
      to memref<3x1x4x1xf32, strided<[44, 44, 11, 1]>>
  // CHECK-NOT:  memref.copy
  return
}

// CHECK-LABEL: func.func private @copy_2D_into_3D_strided_nonzero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<3x1x4x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<3x1x4x11xf32>
func.func private @copy_2D_into_3D_strided_nonzero_offset(
  %src : memref<3x1x4x1xf32>, %dst : memref<3x1x4x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [10], sizes: [3, 1, 4, 1], strides: [44, 44, 11, 1]
    : memref<3x1x4x11xf32>
      to memref<3x1x4x1xf32, strided<[44, 44, 11, 1], offset: 10>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[UB0:.*]] = arith.constant 3 : index
  // CHECK-DAG:  %[[UB1:.*]] = arith.constant 4 : index
  // CHECK-DAG:  %[[OFF:.*]] = arith.constant 10 : index
  // CHECK:      scf.for %[[IDX0:.*]] = %[[C0]] to %[[UB0]] step %[[C1]] {
  // CHECK:        scf.for %[[IDX1:.*]] = %[[C0]] to %[[UB1]] step %[[C1]] {
  // CHECK:          %[[VAL:.*]] = memref.load %[[SRC]][%[[IDX0]], %[[C0]], %[[IDX1]], %[[C0]]] : memref<3x1x4x1xf32>
  // CHECK:          memref.store %[[VAL]], %[[DST]][%[[IDX0]], %[[C0]], %[[IDX1]], %[[OFF]]] : memref<3x1x4x11xf32>
  // CHECK:        }
  // CHECK:      }
  memref.copy %src, %rc
    : memref<3x1x4x1xf32>
      to memref<3x1x4x1xf32, strided<[44, 44, 11, 1], offset: 10>>
  // CHECK-NOT:  memref.copy
  return
}

//===----------------------------------------------------------------------===// 
// Either scalar (0D) OR non-scalar (ND) copy
//===----------------------------------------------------------------------===//

/// Reject copies that don't target a reinterpret_cast result
// CHECK-LABEL: func.func private @negative_no_rc(
func.func private @negative_no_rc(%src : memref<1x1xf32>,
  %dst : memref<1x1xf32>) {
  // CHECK:      memref.copy %arg0, %arg1
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %dst
  : memref<1x1xf32> to memref<1x1xf32>
  return
}

/// Reject unranked memref operands
// CHECK-LABEL: func.func private @negative_copy_into_strided_unranked_rc_base(
func.func private @negative_copy_into_strided_unranked_rc_base(
  %src : memref<4xf32>, %dst : memref<*xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg1
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [4], strides: [1]
    : memref<*xf32> to memref<4xf32>

  // CHECK:      memref.copy %arg0, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %rc
    : memref<4xf32> to memref<4xf32>
  return
}

/// Reject rank-changing reinterpet_casts
// CHECK-LABEL: func.func private @negative_copy_into_strided_rank_change(
func.func private @negative_copy_into_strided_rank_change(%src : memref<3x4xf32>,
  %dst : memref<12xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg1
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [3, 4], strides: [1, 1]
    : memref<12xf32> to memref<3x4xf32, strided<[1, 1]>>

  // CHECK:      memref.copy %arg0, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %rc
    : memref<3x4xf32> to memref<3x4xf32, strided<[1, 1]>>
  return
}

/// Reject dynamic shapes
// CHECK-LABEL: func.func private @negative_copy_into_strided_dynamic_copy_source_shape(
func.func private @negative_copy_into_strided_dynamic_copy_source_shape(%src : memref<?xf32>,
  %dst : memref<4xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg1
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [4], strides: [1]
    : memref<4xf32> to memref<4xf32>

  // CHECK:      memref.copy %arg0, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %rc
    : memref<?xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func private @negative_copy_into_strided_dynamic_rc_source_shape(
func.func private @negative_copy_into_strided_dynamic_rc_source_shape(
  %src : memref<4xf32>, %dst : memref<?xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg1
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [4], strides: [1]
    : memref<?xf32> to memref<4xf32, strided<[1]>>

  // CHECK:      memref.copy %arg0, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %rc
    : memref<4xf32> to memref<4xf32, strided<[1]>>
  return
}

// CHECK-LABEL: func.func private @negative_copy_into_strided_dynamic_rc_result_shape(
func.func private @negative_copy_into_strided_dynamic_rc_result_shape(%dim : index,
  %src : memref<4xf32>, %dst : memref<12xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg2
  %rc = memref.reinterpret_cast %dst
    to offset: [0], sizes: [%dim], strides: [1]
    : memref<12xf32> to memref<?xf32, strided<[1]>>

  // CHECK:      memref.copy %arg1, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %rc
    : memref<4xf32> to memref<?xf32, strided<[1]>>
  return
}
