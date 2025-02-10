// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

///----------------------------------------------------------------------------------------
/// [Pattern: PadOpVectorizationWithTransferReadPattern]
///----------------------------------------------------------------------------------------
// CHECK-LABEL: func @pad_and_transfer_read
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<5x6xf32>
//   CHECK-NOT:   tensor.pad
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C5:.*]] = arith.constant 5.0
//       CHECK:   %[[RESULT:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], %[[C5]] : tensor<5x6xf32>, vector<7x9xf32>
//       CHECK:   return %[[RESULT]]
func.func @pad_and_transfer_read(%arg0: tensor<5x6xf32>) -> vector<7x9xf32> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5.0 : f32
  %c6 = arith.constant 6.0 : f32
  %0 = tensor.pad %arg0 low[0, 0] high[5, 7] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %c5 : f32
  } : tensor<5x6xf32> to tensor<10x13xf32>
  %1 = vector.transfer_read %0[%c0, %c0], %c6
      : tensor<10x13xf32>, vector<7x9xf32>
  return %1 : vector<7x9xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.op<"func.func">

    transform.apply_patterns to %func_op {
      transform.apply_patterns.linalg.pad_vectorization
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

///----------------------------------------------------------------------------------------
/// [Pattern: PadOpVectorizationWithTransferWritePattern]
///----------------------------------------------------------------------------------------
func.func private @make_vector() -> vector<7x9xf32>

// CHECK-LABEL: func @pad_and_transfer_write_static_low_and_high
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<5x6xf32>
//   CHECK-NOT:   tensor.pad
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[VEC0:.*]] = call @make_vector() : () -> vector<7x9xf32>
//       CHECK:   %[[RESULT:.*]] = vector.transfer_write %[[VEC0]], %[[ARG0]][%[[C0]], %[[C0]]] : vector<7x9xf32>, tensor<5x6xf32>
//       CHECK:   return %[[RESULT]]
func.func @pad_and_transfer_write_static_low_and_high(
    %arg0: tensor<5x6xf32>) -> tensor<5x6xf32> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5.0 : f32
  %0 = tensor.pad %arg0 low[0, 0] high[5, 7] {
    ^bb0(%arg2: index, %arg3: index):
      tensor.yield %c5 : f32
  } : tensor<5x6xf32> to tensor<10x13xf32>
  %1 = call @make_vector() : () -> vector<7x9xf32>
  %2 = vector.transfer_write %1, %0[%c0, %c0]
      : vector<7x9xf32>, tensor<10x13xf32>
  %3 = tensor.extract_slice %2[0, 0] [5, 6] [1, 1] : tensor<10x13xf32> to tensor<5x6xf32>
  return %3 : tensor<5x6xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.op<"func.func">

    transform.apply_patterns to %func_op {
      transform.apply_patterns.linalg.pad_vectorization
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

func.func private @make_vector() -> vector<7x9xf32>

// CHECK-LABEL: func @pad_and_transfer_write_static_low_dynamic_high
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<?x?xf32>, %[[SIZE:.*]]: index, %[[PADDING:.*]]: index
//   CHECK-NOT:   tensor.pad
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[SUB:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [%[[SIZE]], 6] [1, 1] : tensor<?x?xf32> to tensor<?x6xf32>
//       CHECK:   %[[VEC0:.*]] = call @make_vector() : () -> vector<7x9xf32>
//       CHECK:   %[[RESULT:.*]] = vector.transfer_write %[[VEC0]], %[[SUB]][%[[C0]], %[[C0]]] : vector<7x9xf32>, tensor<?x6xf32>
//       CHECK:   return %[[RESULT]]
func.func @pad_and_transfer_write_static_low_dynamic_high(
    %arg0: tensor<?x?xf32>, %size: index, %padding: index) -> tensor<?x6xf32> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5.0 : f32
  %s = tensor.extract_slice %arg0[0, 0] [%size, 6] [1, 1]
      : tensor<?x?xf32> to tensor<?x6xf32>
  %0 = tensor.pad %s low[0, 0] high[%padding, 7] {
    ^bb0(%arg2: index, %arg3: index):
      tensor.yield %c5 : f32
  } : tensor<?x6xf32> to tensor<?x13xf32>
  %1 = call @make_vector() : () -> vector<7x9xf32>
  %2 = vector.transfer_write %1, %0[%c0, %c0]
      : vector<7x9xf32>, tensor<?x13xf32>
  %3 = tensor.extract_slice %2[0, 0] [%size, 6] [1, 1] : tensor<?x13xf32> to tensor<?x6xf32>
  return %3 : tensor<?x6xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.op<"func.func">

    transform.apply_patterns to %func_op {
      transform.apply_patterns.linalg.pad_vectorization
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

func.func private @make_vector() -> vector<7x9xf32>

// Negative test - low pad is non-zero

// CHECK-LABEL: func @pad_and_transfer_write_static_non_zero_low_pad
//   CHECK:   tensor.pad
func.func @pad_and_transfer_write_static_non_zero_low_pad(
    %arg0: tensor<5x6xf32>) -> tensor<5x6xf32> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5.0 : f32
  %0 = tensor.pad %arg0 low[0, 1] high[5, 6] {
    ^bb0(%arg2: index, %arg3: index):
      tensor.yield %c5 : f32
  } : tensor<5x6xf32> to tensor<10x13xf32>
  %1 = call @make_vector() : () -> vector<7x9xf32>
  %2 = vector.transfer_write %1, %0[%c0, %c0]
      : vector<7x9xf32>, tensor<10x13xf32>
  %3 = tensor.extract_slice %2[0, 0] [5, 6] [1, 1] : tensor<10x13xf32> to tensor<5x6xf32>
  return %3 : tensor<5x6xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.op<"func.func">

    transform.apply_patterns to %func_op {
      transform.apply_patterns.linalg.pad_vectorization
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

// Negative test - TransferWriteOp result is not _directly_ consumed by an
// ExtractSliceOp (noet the non-zero offset).

func.func private @make_vector() -> vector<7x9xf32>

// CHECK-LABEL: func @pad_and_transfer_write_static_non_zero_offset
//   CHECK:   tensor.pad
func.func @pad_and_transfer_write_static_non_zero_offset(
    %arg0: tensor<5x6xf32>) -> tensor<5x6xf32> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5.0 : f32
  %0 = tensor.pad %arg0 low[0, 0] high[5, 7] {
    ^bb0(%arg2: index, %arg3: index):
      tensor.yield %c5 : f32
  } : tensor<5x6xf32> to tensor<10x13xf32>
  %1 = call @make_vector() : () -> vector<7x9xf32>
  %2 = vector.transfer_write %1, %0[%c0, %c0]
      : vector<7x9xf32>, tensor<10x13xf32>
  %3 = tensor.extract_slice %2[0, 1] [5, 6] [1, 1] : tensor<10x13xf32> to tensor<5x6xf32>
  return %3 : tensor<5x6xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.op<"func.func">

    transform.apply_patterns to %func_op {
      transform.apply_patterns.linalg.pad_vectorization
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

///----------------------------------------------------------------------------------------
/// [Pattern: PadOpVectorizationWithInsertSlicePattern]
///----------------------------------------------------------------------------------------

func.func private @make_vector() -> tensor<12x13xf32>

// CHECK-LABEL: func @pad_and_insert_slice_source
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<5x6xf32>
//   CHECK-NOT:   tensor.pad
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C5:.*]] = arith.constant 5.0
//       CHECK:   %[[VEC0:.*]] = call @make_vector() : () -> tensor<12x13xf32>
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], %[[C5]] : tensor<5x6xf32>, vector<7x9xf32>
//       CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[READ]], %[[VEC0]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<7x9xf32>, tensor<12x13xf32>
//       CHECK:   return %[[WRITE]]
func.func @pad_and_insert_slice_source(
    %arg0: tensor<5x6xf32>) -> tensor<12x13xf32> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5.0 : f32
  %0 = tensor.pad %arg0 low[0, 0] high[2, 3] {
    ^bb0(%arg2: index, %arg3: index):
      tensor.yield %c5 : f32
  } : tensor<5x6xf32> to tensor<7x9xf32>
  %1 = call @make_vector() : () -> tensor<12x13xf32>
  %r = tensor.insert_slice %0 into %1[0, 0][7, 9][1, 1] : tensor<7x9xf32> into tensor<12x13xf32>
  return %r : tensor<12x13xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.op<"func.func">

    transform.apply_patterns to %func_op {
      transform.apply_patterns.linalg.pad_vectorization
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

func.func private @make_vector() -> tensor<12x13xf32>

// The destination of tensor.insert_slice matches the result of tensor.pad -
// not supported.

// CHECK-LABEL:   func.func @pad_and_insert_slice_dest(
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write

func.func @pad_and_insert_slice_dest(
    %arg0: tensor<1x5x6xf32>) -> tensor<1x12x13xf32> {
  %c5 = arith.constant 5.0 : f32
  %0 = tensor.pad %arg0 low[0, 0, 0] high[0, 7, 7] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index):
      tensor.yield %c5 : f32
  } : tensor<1x5x6xf32> to tensor<1x12x13xf32>
  %1 = call @make_vector() : () -> tensor<12x13xf32>
  %r = tensor.insert_slice %1 into %0[0, 0, 0][1, 12, 13][1, 1, 1] : tensor<12x13xf32> into tensor<1x12x13xf32>
  return %r : tensor<1x12x13xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.op<"func.func">

    transform.apply_patterns to %func_op {
      transform.apply_patterns.linalg.pad_vectorization
    } : !transform.op<"func.func">
    transform.yield
  }
}
