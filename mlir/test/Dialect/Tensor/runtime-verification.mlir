// RUN: mlir-opt %s -generate-runtime-verification -cse -split-input-file | FileCheck %s

// CHECK-LABEL: func @expand_shape(
//  CHECK-SAME:     %[[m:.*]]: tensor<?xf32>
//  CHECK-SAME:     %[[sz0:.*]]: index
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = tensor.dim %[[m]], %[[c0]]
//       CHECK:   %[[c2:.*]] = arith.constant 2 : index
//       CHECK:   %[[prod:.*]] = arith.muli %[[sz0]], %[[c2]]
//       CHECK:   %[[cmpi:.*]] = arith.cmpi eq, %[[prod]], %[[dim]]
//       CHECK:   cf.assert %[[cmpi]], "ERROR: Runtime op verification failed
func.func @expand_shape(%m: tensor<?xf32>, %sz0: index) -> tensor<?x2xf32> {
  %0 = tensor.expand_shape %m [[0, 1]] output_shape [%sz0, 2] : tensor<?xf32> into tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: func @concat(
//  CHECK-SAME:     %[[a:.*]]: tensor<?x?xf32>,
//  CHECK-SAME:     %[[b:.*]]: tensor<?x?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[da:.*]] = tensor.dim %[[a]], %[[c0]]
//       CHECK:   %[[db:.*]] = tensor.dim %[[b]], %[[c0]]
//       CHECK:   %[[cmp:.*]] = arith.cmpi eq, %[[db]], %[[da]]
//       CHECK:   cf.assert %[[cmp]], "ERROR: Runtime op verification failed
func.func @concat(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensor.concat dim(1) %a, %b : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @concat_sum(
//  CHECK-SAME:     %[[a:.*]]: tensor<?x3xf32>,
//  CHECK-SAME:     %[[b:.*]]: tensor<?x3xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[da:.*]] = tensor.dim %[[a]], %[[c0]]
//       CHECK:   %[[db:.*]] = tensor.dim %[[b]], %[[c0]]
//       CHECK:   %[[sum:.*]] = arith.addi %[[da]], %[[db]]
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   %[[cmp:.*]] = arith.cmpi eq, %[[sum]], %[[c5]]
//       CHECK:   cf.assert %[[cmp]], "ERROR: Runtime op verification failed
func.func @concat_sum(%a: tensor<?x3xf32>, %b: tensor<?x3xf32>) -> tensor<5x3xf32> {
  %0 = tensor.concat dim(0) %a, %b : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
}

// -----

// CHECK-LABEL: func @reshape(
//  CHECK-SAME:     %[[x:.*]]: tensor<?xf32>
//  CHECK-SAME:     %[[shape:.*]]: tensor<2xindex>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[e0:.*]] = tensor.extract %[[shape]][%[[c0]]]
//       CHECK:   %[[c1:.*]] = arith.constant 1 : index
//       CHECK:   %[[e1:.*]] = tensor.extract %[[shape]][%[[c1]]]
//       CHECK:   %[[dim:.*]] = tensor.dim %[[x]], %[[c0]]
//       CHECK:   %[[prod:.*]] = arith.muli %[[e0]], %[[e1]]
//       CHECK:   %[[cmp:.*]] = arith.cmpi eq, %[[prod]], %[[dim]]
//       CHECK:   cf.assert %[[cmp]], "ERROR: Runtime op verification failed
func.func @reshape(%x: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  %0 = tensor.reshape %x(%shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// Check that shape operand values are compared against static result sizes.

// CHECK-LABEL: func @reshape_static_result(
//  CHECK-SAME:     %[[x:.*]]: tensor<?xf32>
//  CHECK-SAME:     %[[shape:.*]]: tensor<2xindex>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[e0:.*]] = tensor.extract %[[shape]][%[[c0]]]
//       CHECK:   %[[c1:.*]] = arith.constant 1 : index
//       CHECK:   %[[e1:.*]] = tensor.extract %[[shape]][%[[c1]]]
//       CHECK:   %[[c6:.*]] = arith.constant 6 : index
//       CHECK:   %[[cmp0:.*]] = arith.cmpi eq, %[[e0]], %[[c6]]
//       CHECK:   cf.assert %[[cmp0]], "ERROR: Runtime op verification failed
//       CHECK:   %[[dim:.*]] = tensor.dim %[[x]], %[[c0]]
//       CHECK:   %[[prod:.*]] = arith.muli %[[e0]], %[[e1]]
//       CHECK:   %[[cmp1:.*]] = arith.cmpi eq, %[[prod]], %[[dim]]
//       CHECK:   cf.assert %[[cmp1]], "ERROR: Runtime op verification failed
func.func @reshape_static_result(%x: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<6x?xf32> {
  %0 = tensor.reshape %x(%shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<6x?xf32>
  return %0 : tensor<6x?xf32>
}