// RUN: mlir-opt %s -test-affine-reify-value-bounds -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

// CHECK-LABEL: func @scf_for(
//  CHECK-SAME:     %[[a:.*]]: index, %[[b:.*]]: index, %[[c:.*]]: index
//       CHECK:   "test.some_use"(%[[a]], %[[b]])
func.func @scf_for(%a: index, %b: index, %c: index) {
  scf.for %iv = %a to %b step %c {
    %0 = "test.reify_bound"(%iv) {type = "LB"} : (index) -> (index)
    %1 = "test.reify_bound"(%iv) {type = "UB"} : (index) -> (index)
    "test.some_use"(%0, %1) : (index, index) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @scf_for_index_result_small(
//  CHECK-SAME:     %[[i:.*]]: index, %[[a:.*]]: index, %[[b:.*]]: index, %[[c:.*]]: index
//       CHECK:   "test.some_use"(%[[i]])
//       CHECK:   "test.some_use"(%[[i]])
func.func @scf_for_index_result_small(%i: index, %a: index, %b: index, %c: index) {
  %0 = scf.for %iv = %a to %b step %c iter_args(%arg = %i) -> index {
    %1 = "test.reify_bound"(%arg) {type = "EQ"} : (index) -> (index)
    "test.some_use"(%1) : (index) -> ()
    scf.yield %arg : index
  }
  %2 = "test.reify_bound"(%0) {type = "EQ"} : (index) -> (index)
  "test.some_use"(%2) : (index) -> ()
  return
}

// -----

// CHECK-LABEL: func @scf_for_index_result(
//  CHECK-SAME:     %[[i:.*]]: index, %[[a:.*]]: index, %[[b:.*]]: index, %[[c:.*]]: index
//       CHECK:   "test.some_use"(%[[i]])
//       CHECK:   "test.some_use"(%[[i]])
func.func @scf_for_index_result(%i: index, %a: index, %b: index, %c: index) {
  %0 = scf.for %iv = %a to %b step %c iter_args(%arg = %i) -> index {
    %add = arith.addi %arg, %a : index
    %sub = arith.subi %add, %a : index

    %1 = "test.reify_bound"(%arg) {type = "EQ"} : (index) -> (index)
    "test.some_use"(%1) : (index) -> ()
    scf.yield %sub : index
  }
  %2 = "test.reify_bound"(%0) {type = "EQ"} : (index) -> (index)
  "test.some_use"(%2) : (index) -> ()
  return
}

// -----

// CHECK-LABEL: func @scf_for_tensor_result_small(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>, %[[a:.*]]: index, %[[b:.*]]: index, %[[c:.*]]: index
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]]
//       CHECK:   "test.some_use"(%[[dim]])
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]]
//       CHECK:   "test.some_use"(%[[dim]])
func.func @scf_for_tensor_result_small(%t: tensor<?xf32>, %a: index, %b: index, %c: index) {
  %0 = scf.for %iv = %a to %b step %c iter_args(%arg = %t) -> tensor<?xf32> {
    %1 = "test.reify_bound"(%arg) {type = "EQ", dim = 0} : (tensor<?xf32>) -> (index)
    "test.some_use"(%1) : (index) -> ()
    scf.yield %arg : tensor<?xf32>
  }
  %2 = "test.reify_bound"(%0) {type = "EQ", dim = 0} : (tensor<?xf32>) -> (index)
  "test.some_use"(%2) : (index) -> ()
  return
}

// -----

// CHECK-LABEL: func @scf_for_tensor_result(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>, %[[a:.*]]: index, %[[b:.*]]: index, %[[c:.*]]: index
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]]
//       CHECK:   "test.some_use"(%[[dim]])
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]]
//       CHECK:   "test.some_use"(%[[dim]])
func.func @scf_for_tensor_result(%t: tensor<?xf32>, %a: index, %b: index, %c: index) {
  %cst = arith.constant 5.0 : f32
  %0 = scf.for %iv = %a to %b step %c iter_args(%arg = %t) -> tensor<?xf32> {
    %filled = linalg.fill ins(%cst : f32) outs(%arg : tensor<?xf32>) -> tensor<?xf32>
    %1 = "test.reify_bound"(%arg) {type = "EQ", dim = 0} : (tensor<?xf32>) -> (index)
    "test.some_use"(%1) : (index) -> ()
    scf.yield %filled : tensor<?xf32>
  }
  %2 = "test.reify_bound"(%0) {type = "EQ", dim = 0} : (tensor<?xf32>) -> (index)
  "test.some_use"(%2) : (index) -> ()
  return
}

// -----

func.func @scf_for_swapping_yield(%t1: tensor<?xf32>, %t2: tensor<?xf32>, %a: index, %b: index, %c: index) {
  %cst = arith.constant 5.0 : f32
  %r1, %r2 = scf.for %iv = %a to %b step %c iter_args(%arg1 = %t1, %arg2 = %t2) -> (tensor<?xf32>, tensor<?xf32>) {
    %filled1 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<?xf32>) -> tensor<?xf32>
    %filled2 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
    scf.yield %filled2, %filled1 : tensor<?xf32>, tensor<?xf32>
  }
  // expected-error @below{{could not reify bound}}
  %reify1 = "test.reify_bound"(%r1) {type = "EQ", dim = 0} : (tensor<?xf32>) -> (index)
  "test.some_use"(%reify1) : (index) -> ()
  return
}
