// RUN: mlir-opt %s -test-tensor-transform-patterns="test-bubble-up-extract-slice-alignment" | FileCheck %s

// Power-of-two alignment through integer expressions and loop bounds.

// CHECK-LABEL: func.func @bubble_collapse_muli_offset(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 8] [1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_collapse_muli_offset(%src: tensor<4x8xf32>, %row: index) -> tensor<8xf32> {
  %c8 = arith.constant 8 : index
  %offset = arith.muli %row, %c8 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
  return %slice : tensor<8xf32>
}

// CHECK-LABEL: func.func @bubble_collapse_addi_offset(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 8] [1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_collapse_addi_offset(%src: tensor<4x8xf32>, %row: index) -> tensor<8xf32> {
  %c8 = arith.constant 8 : index
  %base = arith.muli %row, %c8 : index
  %offset = arith.addi %base, %c8 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
  return %slice : tensor<8xf32>
}

// CHECK-LABEL: func.func @bubble_collapse_subi_offset(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 8] [1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_collapse_subi_offset(%src: tensor<4x8xf32>, %row: index) -> tensor<8xf32> {
  %c8 = arith.constant 8 : index
  %base = arith.muli %row, %c8 : index
  %offset = arith.subi %base, %c8 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
  return %slice : tensor<8xf32>
}

// CHECK-LABEL: func.func @bubble_collapse_select_offset(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 8] [1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_collapse_select_offset(%src: tensor<4x8xf32>, %row: index, %cond: i1) -> tensor<8xf32> {
  %c8 = arith.constant 8 : index
  %a = arith.muli %row, %c8 : index
  %b = arith.addi %a, %c8 : index
  %offset = arith.select %cond, %a, %b : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
  return %slice : tensor<8xf32>
}

// CHECK-LABEL: func.func @no_bubble_collapse_misaligned_add(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_collapse_misaligned_add(%src: tensor<4x8xf32>, %row: index) -> tensor<8xf32> {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %base = arith.muli %row, %c8 : index
  %offset = arith.addi %base, %c1 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
  return %slice : tensor<8xf32>
}

// CHECK-LABEL: func.func @no_bubble_collapse_misaligned_select(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_collapse_misaligned_select(%src: tensor<4x8xf32>, %row: index, %cond: i1) -> tensor<8xf32> {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %a = arith.muli %row, %c8 : index
  %b = arith.addi %a, %c1 : index
  %offset = arith.select %cond, %a, %b : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
  return %slice : tensor<8xf32>
}

// CHECK-LABEL: func.func @no_bubble_collapse_non_power_of_two_muli(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x6xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_collapse_non_power_of_two_muli(%src: tensor<4x6xf32>, %row: index) -> tensor<6xf32> {
  %c6 = arith.constant 6 : index
  %offset = arith.muli %row, %c6 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x6xf32> into tensor<24xf32>
  %slice = tensor.extract_slice %flat[%offset] [6] [1] : tensor<24xf32> to tensor<6xf32>
  return %slice : tensor<6xf32>
}

// CHECK-LABEL: func.func @no_bubble_collapse_unsupported_division(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_collapse_unsupported_division(%src: tensor<4x8xf32>, %row: index) -> tensor<8xf32> {
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  %base = arith.muli %row, %c8 : index
  %offset = arith.divui %base, %c2 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
  return %slice : tensor<8xf32>
}

// CHECK-LABEL: func.func @bubble_collapse_shared_expression_dag(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 8] [1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_collapse_shared_expression_dag(%src: tensor<4x8xf32>, %row: index) -> tensor<8xf32> {
  %c8 = arith.constant 8 : index
  %v0 = arith.muli %row, %c8 : index
  %v1 = arith.addi %v0, %v0 : index
  %v2 = arith.addi %v1, %v1 : index
  %v3 = arith.addi %v2, %v2 : index
  %v4 = arith.addi %v3, %v3 : index
  %v5 = arith.addi %v4, %v4 : index
  %v6 = arith.addi %v5, %v5 : index
  %offset = arith.addi %v6, %v6 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
  return %slice : tensor<8xf32>
}

// CHECK-LABEL: func.func @bubble_collapse_multiple_groups(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8x2x4xf32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 8, 1, 4] [1, 1, 1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_collapse_multiple_groups(%src: tensor<4x8x2x4xf32>, %row: index, %col: index) -> tensor<8x4xf32> {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %a = arith.muli %row, %c8 : index
  %b = arith.muli %col, %c4 : index
  %flat = tensor.collapse_shape %src [[0, 1], [2, 3]] : tensor<4x8x2x4xf32> into tensor<32x8xf32>
  %slice = tensor.extract_slice %flat[%a, %b] [8, 4] [1, 1] : tensor<32x8xf32> to tensor<8x4xf32>
  return %slice : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @bubble_collapse_loop_iv(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: scf.for
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 8] [1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_collapse_loop_iv(%src: tensor<4x8xf32>, %dst: tensor<32xf32>) -> tensor<32xf32> {
  %lb = arith.constant 0 : index
  %ub = arith.constant 25 : index
  %step = arith.constant 8 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %r = scf.for %iv = %lb to %ub step %step iter_args(%out = %dst) -> tensor<32xf32> {
    %slice = tensor.extract_slice %flat[%iv] [8] [1] : tensor<32xf32> to tensor<8xf32>
    %insert = tensor.insert_slice %slice into %out[%iv] [8] [1] : tensor<8xf32> into tensor<32xf32>
    scf.yield %insert : tensor<32xf32>
  }
  return %r : tensor<32xf32>
}

// CHECK-LABEL: func.func @bubble_collapse_aligned_loop_lb(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: scf.for
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 8] [1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_collapse_aligned_loop_lb(%src: tensor<4x8xf32>, %dst: tensor<32xf32>) -> tensor<32xf32> {
  %lb = arith.constant 8 : index
  %ub = arith.constant 25 : index
  %step = arith.constant 8 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %r = scf.for %iv = %lb to %ub step %step iter_args(%out = %dst) -> tensor<32xf32> {
    %slice = tensor.extract_slice %flat[%iv] [8] [1] : tensor<32xf32> to tensor<8xf32>
    %insert = tensor.insert_slice %slice into %out[%iv] [8] [1] : tensor<8xf32> into tensor<32xf32>
    scf.yield %insert : tensor<32xf32>
  }
  return %r : tensor<32xf32>
}

// CHECK-LABEL: func.func @no_bubble_collapse_misaligned_loop_lb(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: scf.for
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_collapse_misaligned_loop_lb(%src: tensor<4x8xf32>, %dst: tensor<32xf32>) -> tensor<32xf32> {
  %lb = arith.constant 1 : index
  %ub = arith.constant 25 : index
  %step = arith.constant 8 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %r = scf.for %iv = %lb to %ub step %step iter_args(%out = %dst) -> tensor<32xf32> {
    %slice = tensor.extract_slice %flat[%iv] [8] [1] : tensor<32xf32> to tensor<8xf32>
    %insert = tensor.insert_slice %slice into %out[%iv] [8] [1] : tensor<8xf32> into tensor<32xf32>
    scf.yield %insert : tensor<32xf32>
  }
  return %r : tensor<32xf32>
}

// CHECK-LABEL: func.func @no_bubble_collapse_misaligned_loop_step(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: scf.for
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_collapse_misaligned_loop_step(%src: tensor<4x8xf32>, %dst: tensor<32xf32>) -> tensor<32xf32> {
  %lb = arith.constant 0 : index
  %ub = arith.constant 25 : index
  %step = arith.constant 4 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %r = scf.for %iv = %lb to %ub step %step iter_args(%out = %dst) -> tensor<32xf32> {
    %slice = tensor.extract_slice %flat[%iv] [8] [1] : tensor<32xf32> to tensor<8xf32>
    %insert = tensor.insert_slice %slice into %out[%iv] [8] [1] : tensor<8xf32> into tensor<32xf32>
    scf.yield %insert : tensor<32xf32>
  }
  return %r : tensor<32xf32>
}

// The data-flow fixed point also proves alignment of a loop-carried offset.
// CHECK-LABEL: func.func @bubble_collapse_iter_arg(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: scf.for
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 8] [1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_collapse_iter_arg(%src: tensor<4x8xf32>, %dst: tensor<32xf32>) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %r:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%offset = %c0, %out = %dst) -> (index, tensor<32xf32>) {
    %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
    %insert = tensor.insert_slice %slice into %out[%offset] [8] [1] : tensor<8xf32> into tensor<32xf32>
    %next = arith.addi %offset, %c8 : index
    scf.yield %next, %insert : index, tensor<32xf32>
  }
  return %r#1 : tensor<32xf32>
}

// CHECK-LABEL: func.func @bubble_collapse_negative_multiplier(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 8] [1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_collapse_negative_multiplier(%src: tensor<4x8xf32>, %row: index) -> tensor<8xf32> {
  %cneg8 = arith.constant -8 : index
  %offset = arith.muli %row, %cneg8 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
  return %slice : tensor<8xf32>
}

// CHECK-LABEL: func.func @bubble_collapse_divisor_product_overflow(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 8] [1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_collapse_divisor_product_overflow(%src: tensor<4x8xf32>, %row: index) -> tensor<8xf32> {
  %c = arith.constant -9223372036854775808 : index
  %c2 = arith.constant 2 : index
  %a = arith.muli %row, %c : index
  %offset = arith.muli %a, %c2 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
  return %slice : tensor<8xf32>
}

// Loop-carried facts must be weakened when the recurrence loses alignment.
// CHECK-LABEL: func.func @no_bubble_collapse_variant_iter_arg(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: scf.for
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_collapse_variant_iter_arg(%src: tensor<4x8xf32>, %dst: tensor<32xf32>) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %r:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%offset = %c0, %out = %dst) -> (index, tensor<32xf32>) {
    %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
    %insert = tensor.insert_slice %slice into %out[%offset] [8] [1] : tensor<8xf32> into tensor<32xf32>
    %next = arith.addi %offset, %c1 : index
    scf.yield %next, %insert : index, tensor<32xf32>
  }
  return %r#1 : tensor<32xf32>
}

// A general divisor cannot be propagated through division after wraparound.
// CHECK-LABEL: func.func @no_bubble_collapse_wrapping_division(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_collapse_wrapping_division(%src: tensor<4x8xf32>, %row: index) -> tensor<8xf32> {
  %c24 = arith.constant 24 : index
  %c3 = arith.constant 3 : index
  %base = arith.muli %row, %c24 : index
  %offset = arith.divui %base, %c3 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
  return %slice : tensor<8xf32>
}

// Analysis of the enclosing scope must tolerate unrelated integer operations.
// CHECK-LABEL: func.func @bubble_collapse_with_unrelated_integer_ops(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 8] [1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_collapse_with_unrelated_integer_ops(%src: tensor<4x8xf32>, %row: index, %wide: i128) -> (tensor<8xf32>, i128, index) {
  %c = arith.constant 170141183460469231731687303715884105728 : i128
  %wide_product = arith.muli %wide, %c : i128
  %zero = arith.constant 0 : index
  %undefined = arith.divui %row, %zero : index
  %c8 = arith.constant 8 : index
  %offset = arith.muli %row, %c8 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %slice = tensor.extract_slice %flat[%offset] [8] [1] : tensor<32xf32> to tensor<8xf32>
  return %slice, %wide_product, %undefined : tensor<8xf32>, i128, index
}

// The first batch cannot prove alignment. Replacing the loop creates a new IV
// whose step is eight; the second batch must analyze it rather than reuse facts.
// CHECK-LABEL: func.func @bubble_collapse_replaced_loop(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: scf.for %[[IV:.*]] = {{.*}} step %[[C8]]
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 8] [1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_collapse_replaced_loop(%src: tensor<4x8xf32>, %out: tensor<32xf32>) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %r = scf.for %iv = %c0 to %c32 step %c1 iter_args(%acc = %out) -> tensor<32xf32> {
    %slice = tensor.extract_slice %flat[%iv] [8] [1] : tensor<32xf32> to tensor<8xf32>
    %next = tensor.insert_slice %slice into %acc[%iv] [8] [1] : tensor<8xf32> into tensor<32xf32>
    scf.yield %next : tensor<32xf32>
  } {test.replacement_step = 8 : i64}
  return %r : tensor<32xf32>
}

// Refreshing analysis must not invent alignment for a misaligned lower bound.
// CHECK-LABEL: func.func @no_bubble_collapse_replaced_loop_misaligned_lb(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: scf.for
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_collapse_replaced_loop_misaligned_lb(%src: tensor<4x8xf32>, %out: tensor<32xf32>) -> tensor<32xf32> {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %r = scf.for %iv = %c1 to %c32 step %c1 iter_args(%acc = %out) -> tensor<32xf32> {
    %slice = tensor.extract_slice %flat[%iv] [8] [1] : tensor<32xf32> to tensor<8xf32>
    %next = tensor.insert_slice %slice into %acc[%iv] [8] [1] : tensor<8xf32> into tensor<32xf32>
    scf.yield %next : tensor<32xf32>
  } {test.replacement_step = 8 : i64}
  return %r : tensor<32xf32>
}
