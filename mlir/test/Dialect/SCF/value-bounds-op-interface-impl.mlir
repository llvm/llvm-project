// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-affine-reify-value-bounds{reify-to-func-args}))' \
// RUN:     -verify-diagnostics -split-input-file | FileCheck %s

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

// -----

// CHECK-LABEL: func @scf_forall(
//  CHECK-SAME:     %[[a:.*]]: index, %[[b:.*]]: index, %[[c:.*]]: index
//       CHECK:   "test.some_use"(%[[a]], %[[b]])
func.func @scf_forall(%a: index, %b: index, %c: index) {
  scf.forall (%iv) = (%a) to (%b) step (%c) {
    %0 = "test.reify_bound"(%iv) {type = "LB"} : (index) -> (index)
    %1 = "test.reify_bound"(%iv) {type = "UB"} : (index) -> (index)
    "test.some_use"(%0, %1) : (index, index) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @scf_forall_tensor_result(
//  CHECK-SAME:     %[[size:.*]]: index, %[[a:.*]]: index, %[[b:.*]]: index, %[[c:.*]]: index
//       CHECK:   "test.some_use"(%[[size]])
//       CHECK:   "test.some_use"(%[[size]])
func.func @scf_forall_tensor_result(%size: index, %a: index, %b: index, %c: index) {
  %cst = arith.constant 5.0 : f32
  %empty = tensor.empty(%size) : tensor<?xf32>
  %0 = scf.forall (%iv) = (%a) to (%b) step (%c) shared_outs(%arg = %empty) -> tensor<?xf32> {
    %filled = linalg.fill ins(%cst : f32) outs(%arg : tensor<?xf32>) -> tensor<?xf32>
    %1 = "test.reify_bound"(%arg) {type = "EQ", dim = 0} : (tensor<?xf32>) -> (index)
    "test.some_use"(%1) : (index) -> ()
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %filled into %arg[0][%size][1] : tensor<?xf32> into tensor<?xf32>
    }
  }
  %2 = "test.reify_bound"(%0) {type = "EQ", dim = 0} : (tensor<?xf32>) -> (index)
  "test.some_use"(%2) : (index) -> ()
  return
}

// -----

// CHECK-LABEL: func @scf_if_constant(
func.func @scf_if_constant(%c : i1) {
  // CHECK: arith.constant 4 : index
  // CHECK: arith.constant 9 : index
  %c4 = arith.constant 4 : index
  %c9 = arith.constant 9 : index
  %r = scf.if %c -> index {
    scf.yield %c4 : index
  } else {
    scf.yield %c9 : index
  }

  // CHECK: %[[c4:.*]] = arith.constant 4 : index
  // CHECK: %[[c10:.*]] = arith.constant 10 : index
  %reify1 = "test.reify_bound"(%r) {type = "LB"} : (index) -> (index)
  %reify2 = "test.reify_bound"(%r) {type = "UB"} : (index) -> (index)
  // CHECK: "test.some_use"(%[[c4]], %[[c10]])
  "test.some_use"(%reify1, %reify2) : (index, index) -> ()
  return
}

// -----

// CHECK: #[[$map:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK: #[[$map1:.*]] = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK-LABEL: func @scf_if_dynamic(
//  CHECK-SAME:     %[[a:.*]]: index, %[[b:.*]]: index, %{{.*}}: i1)
func.func @scf_if_dynamic(%a: index, %b: index, %c : i1) {
  %c4 = arith.constant 4 : index
  %r = scf.if %c -> index {
    %add1 = arith.addi %a, %b : index
    scf.yield %add1 : index
  } else {
    %add2 = arith.addi %b, %c4 : index
    %add3 = arith.addi %add2, %a : index
    scf.yield %add3 : index
  }

  // CHECK: %[[lb:.*]] = affine.apply #[[$map]]()[%[[a]], %[[b]]]
  // CHECK: %[[ub:.*]] = affine.apply #[[$map1]]()[%[[a]], %[[b]]]
  %reify1 = "test.reify_bound"(%r) {type = "LB"} : (index) -> (index)
  %reify2 = "test.reify_bound"(%r) {type = "UB"} : (index) -> (index)
  // CHECK: "test.some_use"(%[[lb]], %[[ub]])
  "test.some_use"(%reify1, %reify2) : (index, index) -> ()
  return
}

// -----

func.func @scf_if_no_affine_bound(%a: index, %b: index, %c : i1) {
  %r = scf.if %c -> index {
    scf.yield %a : index
  } else {
    scf.yield %b : index
  }
  // The reified bound would be min(%a, %b). min/max expressions are not
  // supported in reified bounds.
  // expected-error @below{{could not reify bound}}
  %reify1 = "test.reify_bound"(%r) {type = "LB"} : (index) -> (index)
  "test.some_use"(%reify1) : (index) -> ()
  return
}

// -----

// CHECK-LABEL: func @scf_if_tensor_dim(
func.func @scf_if_tensor_dim(%c : i1) {
  // CHECK: arith.constant 4 : index
  // CHECK: arith.constant 9 : index
  %c4 = arith.constant 4 : index
  %c9 = arith.constant 9 : index
  %t1 = tensor.empty(%c4) : tensor<?xf32>
  %t2 = tensor.empty(%c9) : tensor<?xf32>
  %r = scf.if %c -> tensor<?xf32> {
    scf.yield %t1 : tensor<?xf32>
  } else {
    scf.yield %t2 : tensor<?xf32>
  }

  // CHECK: %[[c4:.*]] = arith.constant 4 : index
  // CHECK: %[[c10:.*]] = arith.constant 10 : index
  %reify1 = "test.reify_bound"(%r) {type = "LB", dim = 0}
      : (tensor<?xf32>) -> (index)
  %reify2 = "test.reify_bound"(%r) {type = "UB", dim = 0}
      : (tensor<?xf32>) -> (index)
  // CHECK: "test.some_use"(%[[c4]], %[[c10]])
  "test.some_use"(%reify1, %reify2) : (index, index) -> ()
  return
}

// -----

// CHECK: #[[$map:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL: func @scf_if_eq(
//  CHECK-SAME:     %[[a:.*]]: index, %[[b:.*]]: index, %{{.*}}: i1)
func.func @scf_if_eq(%a: index, %b: index, %c : i1) {
  %c0 = arith.constant 0 : index
  %r = scf.if %c -> index {
    %add1 = arith.addi %a, %b : index
    scf.yield %add1 : index
  } else {
    %add2 = arith.addi %b, %c0 : index
    %add3 = arith.addi %add2, %a : index
    scf.yield %add3 : index
  }

  // CHECK: %[[eq:.*]] = affine.apply #[[$map]]()[%[[a]], %[[b]]]
  %reify1 = "test.reify_bound"(%r) {type = "EQ"} : (index) -> (index)
  // CHECK: "test.some_use"(%[[eq]])
  "test.some_use"(%reify1) : (index) -> ()
  return
}

// -----

func.func @compare_scf_for(%a: index, %b: index, %c: index) {
  scf.for %iv = %a to %b step %c {
    // expected-remark @below{{true}}
    "test.compare"(%iv, %a) {cmp = "GE"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare"(%iv, %b) {cmp = "LT"} : (index, index) -> ()
  }
  return
}
