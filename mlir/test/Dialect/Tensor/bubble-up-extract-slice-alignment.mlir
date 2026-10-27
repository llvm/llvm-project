// RUN: mlir-opt %s -transform-interpreter | FileCheck %s

// Non-power-of-two factors are valid for SCF IVs, including dynamic upper bounds.
// CHECK-LABEL: func.func @bubble_loop_iv(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x6xf32>
// CHECK: scf.for
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 6] [1, 1]
// CHECK: tensor.collapse_shape %[[SLICE]]
func.func @bubble_loop_iv(%src: tensor<4x6xf32>, %out: tensor<24xf32>, %ub: index) -> tensor<24xf32> {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x6xf32> into tensor<24xf32>
  %r = scf.for %iv = %c0 to %ub step %c6 iter_args(%acc = %out) -> tensor<24xf32> {
    %tile = tensor.extract_slice %flat[%iv] [6] [1] : tensor<24xf32> to tensor<6xf32>
    %next = tensor.insert_slice %tile into %acc[%iv] [6] [1] : tensor<6xf32> into tensor<24xf32>
    scf.yield %next : tensor<24xf32>
  }
  return %r : tensor<24xf32>
}

// CHECK-LABEL: func.func @bubble_nonzero_lb(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x6xf32>
// CHECK: scf.for
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 6] [1, 1]
func.func @bubble_nonzero_lb(%src: tensor<4x6xf32>, %out: tensor<24xf32>) -> tensor<24xf32> {
  %c6 = arith.constant 6 : index
  %c24 = arith.constant 24 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x6xf32> into tensor<24xf32>
  %r = scf.for %iv = %c6 to %c24 step %c6 iter_args(%acc = %out) -> tensor<24xf32> {
    %tile = tensor.extract_slice %flat[%iv] [6] [1] : tensor<24xf32> to tensor<6xf32>
    %next = tensor.insert_slice %tile into %acc[%iv] [6] [1] : tensor<6xf32> into tensor<24xf32>
    scf.yield %next : tensor<24xf32>
  }
  return %r : tensor<24xf32>
}

// CHECK-LABEL: func.func @no_bubble_misaligned_lb(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x6xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: scf.for
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_misaligned_lb(%src: tensor<4x6xf32>, %out: tensor<24xf32>) -> tensor<24xf32> {
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c19 = arith.constant 19 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x6xf32> into tensor<24xf32>
  %r = scf.for %iv = %c1 to %c19 step %c6 iter_args(%acc = %out) -> tensor<24xf32> {
    %tile = tensor.extract_slice %flat[%iv] [6] [1] : tensor<24xf32> to tensor<6xf32>
    %next = tensor.insert_slice %tile into %acc[%iv] [6] [1] : tensor<6xf32> into tensor<24xf32>
    scf.yield %next : tensor<24xf32>
  }
  return %r : tensor<24xf32>
}

// CHECK-LABEL: func.func @no_bubble_misaligned_step(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x6xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: scf.for
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_misaligned_step(%src: tensor<4x6xf32>, %out: tensor<24xf32>) -> tensor<24xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c19 = arith.constant 19 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x6xf32> into tensor<24xf32>
  %r = scf.for %iv = %c0 to %c19 step %c4 iter_args(%acc = %out) -> tensor<24xf32> {
    %tile = tensor.extract_slice %flat[%iv] [6] [1] : tensor<24xf32> to tensor<6xf32>
    %next = tensor.insert_slice %tile into %acc[%iv] [6] [1] : tensor<6xf32> into tensor<24xf32>
    scf.yield %next : tensor<24xf32>
  }
  return %r : tensor<24xf32>
}

// CHECK-LABEL: func.func @no_bubble_dynamic_lb(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x6xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: scf.for
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_dynamic_lb(%src: tensor<4x6xf32>, %out: tensor<24xf32>, %lb: index) -> tensor<24xf32> {
  %c6 = arith.constant 6 : index
  %c24 = arith.constant 24 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x6xf32> into tensor<24xf32>
  %r = scf.for %iv = %lb to %c24 step %c6 iter_args(%acc = %out) -> tensor<24xf32> {
    %tile = tensor.extract_slice %flat[%iv] [6] [1] : tensor<24xf32> to tensor<6xf32>
    %next = tensor.insert_slice %tile into %acc[%iv] [6] [1] : tensor<6xf32> into tensor<24xf32>
    scf.yield %next : tensor<24xf32>
  }
  return %r : tensor<24xf32>
}

// CHECK-LABEL: func.func @no_bubble_dynamic_step(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x6xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: scf.for
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_dynamic_step(%src: tensor<4x6xf32>, %out: tensor<24xf32>, %step: index) -> tensor<24xf32> {
  %c0 = arith.constant 0 : index
  %c24 = arith.constant 24 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x6xf32> into tensor<24xf32>
  %r = scf.for %iv = %c0 to %c24 step %step iter_args(%acc = %out) -> tensor<24xf32> {
    %tile = tensor.extract_slice %flat[%iv] [6] [1] : tensor<24xf32> to tensor<6xf32>
    %next = tensor.insert_slice %tile into %acc[%iv] [6] [1] : tensor<6xf32> into tensor<24xf32>
    scf.yield %next : tensor<24xf32>
  }
  return %r : tensor<24xf32>
}

// The recurrence happens to be aligned, but this rule only handles IVs.
// CHECK-LABEL: func.func @no_bubble_iter_arg(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x6xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: scf.for
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_iter_arg(%src: tensor<4x6xf32>, %out: tensor<24xf32>) -> tensor<24xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x6xf32> into tensor<24xf32>
  %r:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%offset = %c0, %acc = %out) -> (index, tensor<24xf32>) {
    %tile = tensor.extract_slice %flat[%offset] [6] [1] : tensor<24xf32> to tensor<6xf32>
    %next = tensor.insert_slice %tile into %acc[%offset] [6] [1] : tensor<6xf32> into tensor<24xf32>
    %next_offset = arith.addi %offset, %c6 : index
    scf.yield %next_offset, %next : index, tensor<24xf32>
  }
  return %r#1 : tensor<24xf32>
}

// CHECK-LABEL: func.func @no_bubble_arithmetic(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x6xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK: tensor.extract_slice %[[FLAT]]
func.func @no_bubble_arithmetic(%src: tensor<4x6xf32>, %row: index) -> tensor<6xf32> {
  %c6 = arith.constant 6 : index
  %offset = arith.muli %row, %c6 : index
  %flat = tensor.collapse_shape %src [[0, 1]] : tensor<4x6xf32> into tensor<24xf32>
  %tile = tensor.extract_slice %flat[%offset] [6] [1] : tensor<24xf32> to tensor<6xf32>
  return %tile : tensor<6xf32>
}

func.func private @consume(tensor<6x6xf32>)

// CHECK-LABEL: func.func @bubble_parallel_ivs(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x6x4x6xf32>
// CHECK: scf.parallel
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME: [1, 6, 1, 6] [1, 1, 1, 1]
func.func @bubble_parallel_ivs(%src: tensor<4x6x4x6xf32>) {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c24 = arith.constant 24 : index
  %flat = tensor.collapse_shape %src [[0, 1], [2, 3]] : tensor<4x6x4x6xf32> into tensor<24x24xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c24, %c24) step (%c6, %c6) {
    %tile = tensor.extract_slice %flat[%i, %j] [6, 6] [1, 1] : tensor<24x24xf32> to tensor<6x6xf32>
    func.call @consume(%tile) : (tensor<6x6xf32>) -> ()
    scf.reduce
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.tensor.bubble_up_extract_slice
    } : !transform.any_op
    transform.yield
  }
}
