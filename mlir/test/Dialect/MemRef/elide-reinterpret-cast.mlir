// RUN: mlir-opt -split-input-file -memref-elide-reinterpret-cast %s \
// RUN: | FileCheck %s

//===----------------------------------------------------------------------===//
// Positive tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func private @concat_zero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x108xf32>
func.func private @concat_zero_offset(%src : memref<1x1xf32>,
  %dst : memref<1x108xf32>) {
  /// reinterpret_cast removed
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1], strides: [1, 1]
    : memref<1x108xf32> to memref<1x1xf32>

  /// Ensure copy was replaced
  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]]] : memref<1x1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C0_0]]] : memref<1x108xf32>
  memref.copy %src, %reinterpret_cast
    : memref<1x1xf32> to memref<1x1xf32>
  return
}

// CHECK-LABEL: func.func private @concat_nonzero_offset(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x108xf32>
func.func private @concat_nonzero_offset(%src : memref<1x1xf32>,
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
  return
}

// CHECK-LABEL: func.func private @concat_dynamic_offset(
// CHECK-SAME:   %[[OFF:.*]]: index
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x108xf32>
func.func private @concat_dynamic_offset(%offset: index, %src : memref<1x1xf32>,
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
  return
}

// CHECK-LABEL: func.func private @concat_strided(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x108xf32>
func.func private @concat_strided(%src : memref<1x1xf32>,
  %dst : memref<1x108xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1], strides: [107, 2]
    : memref<1x108xf32> to memref<1x1xf32, strided<[107, 2]>>

  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]]] : memref<1x1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C0_0]]] : memref<1x108xf32>
  memref.copy %src, %reinterpret_cast
    : memref<1x1xf32> to memref<1x1xf32, strided<[107, 2]>>
  return
}

// CHECK-LABEL: func.func private @concat_dynamic_stride(
// CHECK-SAME:   %[[STR0:[A-Za-z][A-Za-z0-9-]*]]: index
// CHECK-SAME:   %[[STR1:[A-Za-z][A-Za-z0-9-]*]]: index
// CHECK-SAME:   %[[SRC:[A-Za-z][A-Za-z0-9-]*]]: memref<1x1xf32>
// CHECK-SAME:   %[[DST:[A-Za-z][A-Za-z0-9-]*]]: memref<1x108xf32>
func.func private @concat_dynamic_stride(%stride0: index,
  %stride1: index, %src : memref<1x1xf32>, %dst : memref<1x108xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1], strides: [%stride0, %stride1]
    : memref<1x108xf32>
      to memref<1x1xf32, strided<[?, ?]>>

  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]]] : memref<1x1xf32>
  /// Dynamic offset used in store
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C0_0]]] : memref<1x108xf32>
  memref.copy %src, %reinterpret_cast
    : memref<1x1xf32>
      to memref<1x1xf32, strided<[?, ?]>>
  return
}

// CHECK-LABEL: func.func private @concat_rank1(
// CHECK-SAME:   %[[SRC:.*]]: memref<1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<108xf32>
func.func private @concat_rank1(%src : memref<1xf32>, %dst : memref<108xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1], strides: [1]
    : memref<108xf32> to memref<1xf32>

  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]]] : memref<1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0_0]]] : memref<108xf32>
  memref.copy %src, %reinterpret_cast
    : memref<1xf32> to memref<1xf32>
  return
}

// CHECK-LABEL: func.func private @concat_rank3(
// CHECK-SAME:   %[[SRC:.*]]: memref<1x1x1xf32>
// CHECK-SAME:   %[[DST:.*]]: memref<1x1x108xf32>
func.func private @concat_rank3(%src : memref<1x1x1xf32>,
  %dst : memref<1x1x108xf32>) {
  // CHECK-NOT:  memref.reinterpret_cast
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1, 1], strides: [1, 1, 1]
    : memref<1x1x108xf32> to memref<1x1x1xf32>

  // CHECK-NOT:  memref.copy
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK:      %[[VAL:.*]] = memref.load %[[SRC]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x1x1xf32>
  // CHECK:      memref.store %[[VAL]], %[[DST]][%[[C0]], %[[C0]], %[[C0_0]]] : memref<1x1x108xf32>
  memref.copy %src, %reinterpret_cast
    : memref<1x1x1xf32> to memref<1x1x1xf32>
  return
}

//===----------------------------------------------------------------------===//
// Negative tests (must NOT rewrite)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func private @negative_concat_strided_base(
func.func private @negative_concat_strided_base(%src: memref<1x1xf32>,
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

// CHECK-LABEL: func.func private @negative_rank_change(
func.func private @negative_rank_change(%src : memref<2x3xf32>,
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

// CHECK-LABEL: func.func private @negative_concat_multiple_non_unit_dims(
func.func private @negative_concat_multiple_non_unit_dims(
  %src : memref<1x1xf32>, %dst : memref<2x108xf32>) {
  // CHECK:      %reinterpret_cast = memref.reinterpret_cast %arg1
  %reinterpret_cast = memref.reinterpret_cast %dst
    to offset: [0], sizes: [1, 1], strides: [1, 1]
    : memref<2x108xf32>
      to memref<1x1xf32>
  // CHECK:      memref.copy %arg0, %reinterpret_cast
  // CHECK-NOT:  memref.load
  // CHECK-NOT:  memref.store
  memref.copy %src, %reinterpret_cast
    : memref<1x1xf32> to memref<1x1xf32>
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

// -----

//===----------------------------------------------------------------------===//
// Positive tests for offset-shift reinterpret_cast
//
// `RewriteLoadFromOffsetShiftReinterpretCast` folds a load through a
// reinterpret_cast that differs from its source only by offset (rank-1, same
// element type / memory space / strides, innermost stride == 1). The cast
// offset is absorbed into the consumer load index:
//   load %rc[%idx]  ->  load %src[%idx + rcOff - srcOff]
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @offset_shift_static_offsets(
// CHECK-SAME:    %[[SRC:.*]]: memref<16xi8>) -> i8
func.func @offset_shift_static_offsets(%src: memref<16xi8>) -> i8 {
  // CHECK-NOT:   memref.reinterpret_cast
  %rc = memref.reinterpret_cast %src to offset: [4], sizes: [8], strides: [1]
    : memref<16xi8> to memref<8xi8, strided<[1], offset: 4>>
  %c2 = arith.constant 2 : index
  // %adj = 2 + 4 - 0 = 6
  // CHECK:       %[[C6:.*]] = arith.constant 6 : index
  // CHECK:       %[[V:.*]] = memref.load %[[SRC]][%[[C6]]] : memref<16xi8>
  %v = memref.load %rc[%c2] : memref<8xi8, strided<[1], offset: 4>>
  // CHECK:       return %[[V]]
  return %v : i8
}

// -----

// CHECK-LABEL: func.func @offset_shift_dynamic_rc_offset(
// CHECK-SAME:    %[[OFF:.*]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<?xi8>) -> i8
func.func @offset_shift_dynamic_rc_offset(%off: index, %src: memref<?xi8>)
    -> i8 {
  // CHECK-NOT:   memref.reinterpret_cast
  %rc = memref.reinterpret_cast %src to offset: [%off], sizes: [8], strides: [1]
    : memref<?xi8> to memref<8xi8, strided<[1], offset: ?>>
  %c0 = arith.constant 0 : index
  // %adj = 0 + %off - 0 = %off
  // CHECK:       %[[V:.*]] = memref.load %[[SRC]][%[[OFF]]] : memref<?xi8>
  %v = memref.load %rc[%c0] : memref<8xi8, strided<[1], offset: ?>>
  // CHECK:       return %[[V]]
  return %v : i8
}

// -----

// CHECK-LABEL: func.func @offset_shift_dynamic_load_index(
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<16xi8>) -> i8
func.func @offset_shift_dynamic_load_index(%i: index, %src: memref<16xi8>)
    -> i8 {
  // CHECK-NOT:   memref.reinterpret_cast
  %rc = memref.reinterpret_cast %src to offset: [3], sizes: [8], strides: [1]
    : memref<16xi8> to memref<8xi8, strided<[1], offset: 3>>
  // %adj = %i + 3 - 0 = %i + 3
  // CHECK:       %[[ADJ:.*]] = affine.apply {{.*}}[%[[I]]]
  // CHECK:       %[[V:.*]] = memref.load %[[SRC]][%[[ADJ]]] : memref<16xi8>
  %v = memref.load %rc[%i] : memref<8xi8, strided<[1], offset: 3>>
  // CHECK:       return %[[V]]
  return %v : i8
}

// -----

// Dynamic source offset: must materialize the source offset via
// memref.extract_strided_metadata before computing the adjusted index.
//
// CHECK-LABEL: func.func @offset_shift_dynamic_src_offset(
// CHECK-SAME:    %[[OFF:.*]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<?xi8, strided<[1], offset: ?>>) -> i8
func.func @offset_shift_dynamic_src_offset(%off: index,
    %src: memref<?xi8, strided<[1], offset: ?>>) -> i8 {
  // CHECK:       %{{.*}}, %[[SRCOFF:[a-zA-Z0-9_]+]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[SRC]]
  // CHECK-NOT:   memref.reinterpret_cast
  %rc = memref.reinterpret_cast %src to offset: [%off], sizes: [8], strides: [1]
    : memref<?xi8, strided<[1], offset: ?>>
      to memref<8xi8, strided<[1], offset: ?>>
  %c1 = arith.constant 1 : index
  // %adj = 1 + %off - %srcOff
  // CHECK:       %[[ADJ:.*]] = affine.apply {{.*}}[%[[OFF]], %[[SRCOFF]]]
  // CHECK:       %[[V:.*]] = memref.load %[[SRC]][%[[ADJ]]]
  %v = memref.load %rc[%c1] : memref<8xi8, strided<[1], offset: ?>>
  // CHECK:       return %[[V]]
  return %v : i8
}

// -----

// Same-offset cast (rc offset equals src offset). The shift folds to 0,
// so the rewritten load index equals the original index.
//
// CHECK-LABEL: func.func @offset_shift_same_offset(
// CHECK-SAME:    %[[I:.*]]: index
// CHECK-SAME:    %[[SRC:.*]]: memref<16xi8>) -> i8
func.func @offset_shift_same_offset(%i: index, %src: memref<16xi8>) -> i8 {
  // CHECK-NOT:   memref.reinterpret_cast
  %rc = memref.reinterpret_cast %src to offset: [0], sizes: [8], strides: [1]
    : memref<16xi8> to memref<8xi8, strided<[1]>>
  // CHECK:       %[[V:.*]] = memref.load %[[SRC]][%[[I]]] : memref<16xi8>
  %v = memref.load %rc[%i] : memref<8xi8, strided<[1]>>
  // CHECK:       return %[[V]]
  return %v : i8
}

// -----

//===----------------------------------------------------------------------===//
// Negative tests for offset-shift reinterpret_cast (must NOT rewrite)
//===----------------------------------------------------------------------===//

// Rank-2 source/result: pattern is restricted to rank-1.
//
// CHECK-LABEL: func.func @negative_offset_shift_rank2(
func.func @negative_offset_shift_rank2(%src: memref<4x4xi8>) -> i8 {
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast
  %rc = memref.reinterpret_cast %src to offset: [4], sizes: [2, 2], strides: [4, 1]
    : memref<4x4xi8> to memref<2x2xi8, strided<[4, 1], offset: 4>>
  %c0 = arith.constant 0 : index
  // CHECK:       memref.load %[[RC]]
  %v = memref.load %rc[%c0, %c0] : memref<2x2xi8, strided<[4, 1], offset: 4>>
  return %v : i8
}

// -----

// Element type mismatch is invalid IR for reinterpret_cast in general; the
// allowed case the pattern must reject is a *stride* mismatch.
//
// CHECK-LABEL: func.func @negative_offset_shift_diff_stride(
func.func @negative_offset_shift_diff_stride(
    %src: memref<16xi8, strided<[1]>>) -> i8 {
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast
  %rc = memref.reinterpret_cast %src to offset: [4], sizes: [4], strides: [2]
    : memref<16xi8, strided<[1]>>
      to memref<4xi8, strided<[2], offset: 4>>
  %c0 = arith.constant 0 : index
  // CHECK:       memref.load %[[RC]]
  %v = memref.load %rc[%c0] : memref<4xi8, strided<[2], offset: 4>>
  return %v : i8
}

// -----

// Innermost stride != 1: offset shift cannot be absorbed into a single index
// addition without scaling.
//
// CHECK-LABEL: func.func @negative_offset_shift_inner_stride_ne_one(
func.func @negative_offset_shift_inner_stride_ne_one(
    %src: memref<16xi8, strided<[2]>>) -> i8 {
  // CHECK:       %[[RC:.*]] = memref.reinterpret_cast
  %rc = memref.reinterpret_cast %src to offset: [4], sizes: [4], strides: [2]
    : memref<16xi8, strided<[2]>>
      to memref<4xi8, strided<[2], offset: 4>>
  %c0 = arith.constant 0 : index
  // CHECK:       memref.load %[[RC]]
  %v = memref.load %rc[%c0] : memref<4xi8, strided<[2], offset: 4>>
  return %v : i8
}

