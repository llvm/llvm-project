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
