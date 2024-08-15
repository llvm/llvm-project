// RUN: mlir-opt -split-input-file -canonicalize -cse %s | FileCheck %s

// This test verifies the simplification of IR patterns that emerge when
// lowering high-level element-wise ops with unranked tensor inputs. Consider
// the following function incrementing and doubling the value of an input
// unranked tensor using ops in a hypothetical high-level dialect called 'hl':
//
//  func.func @f(%input: tensor<*xf32>) -> tensor<*xf32> {
//    %0 = hl.inc %input : tensor<*xf32>
//    %1 = hl.double %0 : tensor<*xf32>
//    return %1 : tensor<*xf32>
//  }
//
// A possible strategy to lower 'hl.inc' consists in reshaping its operand into
// a 1D tensor, creating a 1D tensor splat with the same total size as the input
// operand and with value 1.0, adding both 1D tensors using 'arith.addf', and
// reshaping the result back into the original input shape. A similar process
// applies for 'hl.double', except with a tensor splat with value 2.0 and an
// 'arith.mulf' op. The body of the function in the test below contains the full
// sequence.
//
// Since such lowering process would operate on individual 'hl' ops in a
// context-oblivious manner, the emitted code produces a redundant IR pattern
// where the result of 'arith.addf' is reshaped into an unranked tensor, just
// for it to be immediately reshaped back into the 1D tensor consumed by
// 'arith.mulf'. This entails the overhead of re-computing the unranked tensor
// shape ('shape.shape_of') and size ('shape.num_elements').
//
// This test verifies that the consecutive application of a canonicalization and
// a CSE pass successfully simplifies this emerging pattern, leading to a
// version of the code in which the result of the emitted 'arith.addf' op
// associated with 'hl.inc' is directly consumed by the 'arith.mulf' op
// associated with 'hl.double', as observed in the FileCheck directives. The
// main rewrite patterns at play are 'shape.shape_of' canonicalization,
// 'tensor.reshape' canonicalization, and 'shape.num_elements' subexpression
// elimination.
//

// CHECK-LABEL: @unranked_tensor_lowering
// CHECK-SAME: %[[INPUT:.*]]: tensor<*xf32>

// CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2.000000e+00 : f32

// CHECK: %[[INPUT_SHAPE:.*]] = shape.shape_of %[[INPUT]] : tensor<*xf32> -> tensor<?xindex>
// CHECK: %[[INPUT_SIZE:.*]] = shape.num_elements %[[INPUT_SHAPE]] : tensor<?xindex> -> index
// CHECK: %[[INPUT_COLLAPSED_SHAPE:.*]] = tensor.from_elements %[[INPUT_SIZE]] : tensor<1xindex>
// CHECK: %[[INPUT_COLLAPSED:.*]] = tensor.reshape %[[INPUT]](%[[INPUT_COLLAPSED_SHAPE]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>

// CHECK: %[[ONE_SPLAT:.*]] = tensor.splat %[[ONE]]{{\[}}%[[INPUT_SIZE]]] : tensor<?xf32>
// CHECK: %[[SUM_COLLAPSED:.*]] = arith.addf %[[INPUT_COLLAPSED]], %[[ONE_SPLAT]] : tensor<?xf32>

// CHECK: %[[TWO_SPLAT:.*]] = tensor.splat %[[TWO]]{{\[}}%[[INPUT_SIZE]]] : tensor<?xf32>
// CHECK: %[[PRODUCT_COLLAPSED:.*]] = arith.mulf %[[SUM_COLLAPSED]], %[[TWO_SPLAT]] : tensor<?xf32>

// CHECK: %[[PRODUCT:.*]] = tensor.reshape %[[PRODUCT_COLLAPSED]](%[[INPUT_SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK: return %[[PRODUCT]] : tensor<*xf32>

func.func @unranked_tensor_lowering(%input: tensor<*xf32>) -> tensor<*xf32> {

  // Collapse input
  %input_shape = shape.shape_of %input : tensor<*xf32> -> tensor<?xindex>
  %input_size = shape.num_elements %input_shape : tensor<?xindex> -> index
  %input_collapsed_shape = tensor.from_elements %input_size : tensor<1xindex>
  %input_collapsed = tensor.reshape %input(%input_collapsed_shape) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>

  // Second operand for sum
  %one = arith.constant 1.0 : f32
  %one_splat = tensor.splat %one[%input_size] : tensor<?xf32>

  // Compute sum and expand it
  %sum_collapsed = arith.addf %input_collapsed, %one_splat : tensor<?xf32>
  %sum = tensor.reshape %sum_collapsed(%input_shape) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>

  // Collapse sum
  %sum_shape = shape.shape_of %sum : tensor<*xf32> -> tensor<?xindex>
  %sum_size = shape.num_elements %sum_shape : tensor<?xindex> -> index
  %sum_collapsed_shape = tensor.from_elements %sum_size : tensor<1xindex>
  %sum_collapsed_0 = tensor.reshape %sum(%sum_collapsed_shape) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>

  // Second operand for product
  %two = arith.constant 2.0 : f32
  %two_splat = tensor.splat %two[%sum_size] : tensor<?xf32>

  // Compute product and expand it
  %product_collapsed = arith.mulf %sum_collapsed_0, %two_splat : tensor<?xf32>
  %product = tensor.reshape %product_collapsed(%sum_shape) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>

  return %product : tensor<*xf32>
}
