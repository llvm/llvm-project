// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(scf-forall-to-for,canonicalize))' -split-input-file | FileCheck %s

func.func private @callee(%i: index, %j: index)

// CHECK-LABEL: @two_iters
// CHECK-SAME: %[[UB1:.+]]: index, %[[UB2:.+]]: index
func.func @two_iters(%ub1: index, %ub2: index) {
  scf.forall (%i, %j) in (%ub1, %ub2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }
  // CHECK: scf.for %[[IV1:.+]] = %{{.*}} to %[[UB1]]
  // CHECK:   scf.for %[[IV2:.+]] = %{{.*}} to %[[UB2]]
  // CHECK:     func.call @callee(%[[IV1]], %[[IV2]])
  return
}

// -----

func.func private @callee(%i: index, %j: index)

// CHECK-LABEL: @repeated
// CHECK-SAME: %[[UB1:.+]]: index, %[[UB2:.+]]: index
func.func @repeated(%ub1: index, %ub2: index) {
  scf.forall (%i, %j) in (%ub1, %ub2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }
  // CHECK: scf.for %[[IV1:.+]] = %{{.*}} to %[[UB1]]
  // CHECK:   scf.for %[[IV2:.+]] = %{{.*}} to %[[UB2]]
  // CHECK:     func.call @callee(%[[IV1]], %[[IV2]])
  scf.forall (%i, %j) in (%ub1, %ub2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }
  // CHECK: scf.for %[[IV1:.+]] = %{{.*}} to %[[UB1]]
  // CHECK:   scf.for %[[IV2:.+]] = %{{.*}} to %[[UB2]]
  // CHECK:     func.call @callee(%[[IV1]], %[[IV2]])
  return
}

// -----

func.func private @callee(%i: index, %j: index, %k: index, %l: index)

// CHECK-LABEL: @nested
// CHECK-SAME: %[[UB1:.+]]: index, %[[UB2:.+]]: index, %[[UB3:.+]]: index, %[[UB4:.+]]: index
func.func @nested(%ub1: index, %ub2: index, %ub3: index, %ub4: index) {
  // CHECK: scf.for %[[IV1:.+]] = %{{.*}} to %[[UB1]]
  // CHECK:   scf.for %[[IV2:.+]] = %{{.*}} to %[[UB2]]
  // CHECK:     scf.for %[[IV3:.+]] = %{{.*}} to %[[UB3]]
  // CHECK:       scf.for %[[IV4:.+]] = %{{.*}} to %[[UB4]]
  // CHECK:         func.call @callee(%[[IV1]], %[[IV2]], %[[IV3]], %[[IV4]])
  scf.forall (%i, %j) in (%ub1, %ub2) {
    scf.forall (%k, %l) in (%ub3, %ub4) {
      func.call @callee(%i, %j, %k, %l) : (index, index, index, index) -> ()
    }
  }
  return
}

// -----

func.func @nested_with_result() -> tensor<4x2xf32> {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<4x2xf32>
  %res = scf.forall (%arg0, %arg1) in (%c4, %c2) shared_outs(%o = %0) -> (tensor<4x2xf32>) {
    %1 = tensor.empty() : tensor<1x1xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x1xf32>) -> tensor<1x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %2 into %o[%arg0, %arg1] [1, 1] [1, 1] :
        tensor<1x1xf32> into tensor<4x2xf32>
    }
  }
  return %res: tensor<4x2xf32>
}

// CHECK-LABEL:   func.func @nested_with_result() -> tensor<4x2xf32> {
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[FILL:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[REDUCED_RES:.*]] = tensor.empty() : tensor<4x2xf32>
// CHECK:           %[[OUTER:.*]] = scf.for %[[IV_OUTER:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[OUTER_RES:.*]] = %[[REDUCED_RES]]) -> (tensor<4x2xf32>) {
// CHECK:             %[[INNER:.*]] = scf.for %[[IV_INNER:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[INNER_RES:.*]] = %[[OUTER_RES]]) -> (tensor<4x2xf32>) {
// CHECK:               %[[ITERATION_TENS:.*]] = tensor.empty() : tensor<1x1xf32>
// CHECK:               %[[ITERATION_RES:.*]] = linalg.fill ins(%[[FILL]] : f32) outs(%[[ITERATION_TENS]] : tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:               %[[UPDATED_RES:.*]] = tensor.insert_slice %[[ITERATION_RES]] into %[[INNER_RES]]{{\[}}%[[IV_OUTER]], %[[IV_INNER]]] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<4x2xf32>
// CHECK:               scf.yield %[[UPDATED_RES]] : tensor<4x2xf32>
// CHECK:             }
// CHECK:             scf.yield %[[INNER]] : tensor<4x2xf32>
// CHECK:           }
// CHECK:           return %[[OUTER]] : tensor<4x2xf32>
// CHECK:         }