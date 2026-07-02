// RUN: mlir-opt -split-input-file -memref-elide-reinterpret-cast %s \
// RUN: | FileCheck %s

//===----------------------------------------------------------------------===//
// Scalar (0D) copy
//
// No varying RC result dimensions =>
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
// CHECK-LABEL: func.func private @negative_copy_scalar_into_1D_strided_zero_offset_non_identity_layout(
func.func private @negative_copy_scalar_into_1D_strided_zero_offset_non_identity_layout(
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

/// Reject dynamic offsets for rc sources with > 1 non-unit dimension -
/// runtime delinearization of these offsets is TODO.
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

// CHECK-LABEL: func.func private @copy_1D_into_2D_strided_delinearized_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x3x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x4x11xf32>
func.func private @copy_1D_into_2D_strided_delinearized_offset(
  %src : memref<1x3x1xf32>, %dst : memref<1x4x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [12], sizes: [1, 3, 1], strides: [44, 11, 1]
    : memref<1x4x11xf32>
      to memref<1x3x1xf32, strided<[44, 11, 1], offset: 12>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[UB:.*]] = arith.constant 3 : index
  // CHECK:      scf.for %[[IDX:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
  // CHECK:        %[[DST_IDX:.*]] = arith.addi %[[C1]], %[[IDX]] : index
  // CHECK:        %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[IDX]], %[[C0]]] : memref<1x3x1xf32>
  // CHECK:        memref.store %[[VAL]], %[[DST]][%[[C0]], %[[DST_IDX]], %[[C1]]] : memref<1x4x11xf32>
  // CHECK:      }
  memref.copy %src, %rc
    : memref<1x3x1xf32>
      to memref<1x3x1xf32, strided<[44, 11, 1], offset: 12>>
  // CHECK-NOT:  memref.copy
  return
}

/// Reject rc result strides that not equal to rc source identity strides.
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

// CHECK-LABEL: func.func private @copy_2D_into_2D_strided_nonzero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x3x4xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x3x11xf32>
func.func private @copy_2D_into_2D_strided_nonzero_offset(
  %src : memref<1x3x4xf32>, %dst : memref<1x3x11xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [7], sizes: [1, 3, 4], strides: [33, 11, 1]
    : memref<1x3x11xf32>
      to memref<1x3x4xf32, strided<[33, 11, 1], offset: 7>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[UB0:.*]] = arith.constant 3 : index
  // CHECK-DAG:  %[[UB1:.*]] = arith.constant 4 : index
  // CHECK-DAG:  %[[OFF:.*]] = arith.constant 7 : index
  // CHECK:      scf.for %[[IDX0:.*]] = %[[C0]] to %[[UB0]] step %[[C1]] {
  // CHECK:        scf.for %[[IDX1:.*]] = %[[C0]] to %[[UB1]] step %[[C1]] {
  // CHECK:          %[[DST_IDX:.*]] = arith.addi %[[OFF]], %[[IDX1]] : index
  // CHECK:          %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[IDX0]], %[[IDX1]]] : memref<1x3x4xf32>
  // CHECK:          memref.store %[[VAL]], %[[DST]][%[[C0]], %[[IDX0]], %[[DST_IDX]]] : memref<1x3x11xf32>
  // CHECK:        }
  // CHECK:      }
  memref.copy %src, %rc
    : memref<1x3x4xf32>
      to memref<1x3x4xf32, strided<[33, 11, 1], offset: 7>>
  // CHECK-NOT:  memref.copy
  return
}

/// rc result dynamic offset:
///    supported only for effectively-1D rc source
///    (runtime delinearization not implemented)
// CHECK-LABEL: func.func private @copy_1D_into_1D_strided_dynamic_offset(
// CHECK-SAME:   %[[OFF:.*]]: index
// CHECK-SAME:   %[[SRC:.*]]: memref<4xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<108xf32>
func.func private @copy_1D_into_1D_strided_dynamic_offset(
  %offset : index, %src : memref<4xf32>, %dst : memref<108xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %rc = memref.reinterpret_cast %dst
    to offset: [%offset], sizes: [4], strides: [1]
    : memref<108xf32> to memref<4xf32, strided<[1], offset: ?>>

  // CHECK-NOT:  memref.copy
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[UB:.*]] = arith.constant 4 : index
  // CHECK:      scf.for %[[IDX:.*]] = %[[C0]] to %[[UB]] step %[[C1]] {
  // CHECK:        %[[DST_IDX:.*]] = arith.addi %[[OFF]], %[[IDX]] : index
  // CHECK:        %[[VAL:.*]] = memref.load %[[SRC]][%[[IDX]]] : memref<4xf32>
  // CHECK:        memref.store %[[VAL]], %[[DST]][%[[DST_IDX]]] : memref<108xf32>
  // CHECK:      }
  memref.copy %src, %rc
    : memref<4xf32> to memref<4xf32, strided<[1], offset: ?>>
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
