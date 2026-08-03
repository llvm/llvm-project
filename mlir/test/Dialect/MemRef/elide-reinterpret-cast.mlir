// RUN: mlir-opt -split-input-file -memref-elide-reinterpret-cast %s \
// RUN: | FileCheck %s

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

// -----

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

// CHECK-LABEL: func.func private @expand_left_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<999xi64>) {
func.func private @expand_left_vector(%src : memref<999xi64>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 1, 999], strides: [999, 999, 1]
    : memref<999xi64> to memref<1x1x999xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_2]]] : memref<999xi64>
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_1, %idx_2] : memref<1x1x999xi64>
  return
}

// CHECK-LABEL: func.func private @expand_left_vector_dynamic_index(
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<999xi64>) {
func.func private @expand_left_vector_dynamic_index(%i : index,
    %src : memref<999xi64>) {
  // CHECK:       %[[IDX:.*]] = arith.constant 0 : index
  %idx = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, 1, 999], strides: [999, 999, 1]
    : memref<999xi64> to memref<1x1x999xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[I]]] : memref<999xi64>
  %0 = memref.load %reinterpret_cast[%idx, %idx, %i] : memref<1x1x999xi64>
  return
}

// CHECK-LABEL: func.func private @collapse_left_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x999xi64>) {
func.func private @collapse_left_vector(%src : memref<1x1x999xi64>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [999], strides: [1]
    : memref<1x1x999xi64> to memref<999xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_1]], %[[IDX_1]], %[[IDX_2]]] : memref<1x1x999xi64>
  %0 = memref.load %reinterpret_cast[%idx] : memref<999xi64>
  return
}

// CHECK-LABEL: func.func private @partial_expand_left_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x999xf32>) {
func.func private @partial_expand_left_vector(
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

// CHECK-LABEL: func.func private @partial_collapse_left_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<1x1x999xf32>) {
func.func private @partial_collapse_left_vector(
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

// CHECK-LABEL: func.func private @expand_right_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<999xi64>) {
func.func private @expand_right_vector(%src : memref<999xi64>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [999, 1, 1], strides: [1, 999, 999]
    : memref<999xi64> to memref<999x1x1xi64, strided<[1, 999, 999]>>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_2]]] : memref<999xi64>
  %0 = memref.load %reinterpret_cast[%idx_2, %idx_1, %idx_1] : memref<999x1x1xi64,
    strided<[1, 999, 999]>>
  return
}

// CHECK-LABEL: func.func private @collapse_right_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<999x1x1xi64>) {
func.func private @collapse_right_vector(%src : memref<999x1x1xi64>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [999], strides: [1]
      : memref<999x1x1xi64> to memref<999xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_2]], %[[IDX_1]], %[[IDX_1]]] : memref<999x1x1xi64>
  %0 = memref.load %reinterpret_cast[%idx] : memref<999xi64>
  return
}

// CHECK-LABEL: func.func private @collapse_right_vector_dynamic_index(
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<999x1x1xi64>) {
func.func private @collapse_right_vector_dynamic_index(%i : index,
    %src : memref<999x1x1xi64>) {
  // CHECK-DAG:   %[[IDX:.*]] = arith.constant 0 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [999], strides: [1]
    : memref<999x1x1xi64> to memref<999xi64>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[I]], %[[IDX]], %[[IDX]]] : memref<999x1x1xi64>
  %0 = memref.load %reinterpret_cast[%i] : memref<999xi64>
  return
}

// CHECK-LABEL: func.func private @partial_expand_right_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<999x1xf32>) {
func.func private @partial_expand_right_vector(
    %src : memref<999x1xf32>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [999, 1, 1], strides: [1, 999, 999]
    : memref<999x1xf32> to memref<999x1x1xf32, strided<[1, 999, 999]>>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_2]], %[[IDX_1]]] : memref<999x1xf32>
  %0 = memref.load %reinterpret_cast[%idx_2, %idx_1, %idx_1]
    : memref<999x1x1xf32, strided<[1, 999, 999]>>
  return
}

// CHECK-LABEL: func.func private @partial_collapse_right_vector(
// CHECK-SAME:    %[[SRC:.*]]: memref<999x1x1xf32>) {
func.func private @partial_collapse_right_vector(
    %src : memref<999x1x1xf32>) {
  // CHECK-DAG:   %[[IDX_1:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[IDX_2:.*]] = arith.constant 13 : index
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK-NOT:   memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [999, 1], strides: [1, 999]
    : memref<999x1x1xf32> to memref<999x1xf32, strided<[1, 999]>>
  // CHECK:       %[[LOAD:.*]] = memref.load %[[SRC]][%[[IDX_2]], %[[IDX_1]], %[[IDX_1]]] : memref<999x1x1xf32>
  %0 = memref.load %reinterpret_cast[%idx_2, %idx_1] : memref<999x1xf32,
    strided<[1, 999]>>
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

// CHECK-LABEL: func.func private @negative_dynamic_shape(
// CHECK-SAME:   %[[SRC:[A-Za-z][A-Za-z0-9-]*]]: memref<?xf32>
func.func private @negative_dynamic_shape(%dim : index,
    %src : memref<?xf32>) {
  %idx_1 = arith.constant 0 : index
  %idx_2 = arith.constant 13 : index
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast %[[SRC]]
  %reinterpret_cast = memref.reinterpret_cast %src
    to offset: [0], sizes: [1, %dim], strides: [1, 1]
    : memref<?xf32> to memref<1x?xf32>
  // CHECK:       memref.load %[[RC]]
  %0 = memref.load %reinterpret_cast[%idx_1, %idx_2] : memref<1x?xf32>
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
