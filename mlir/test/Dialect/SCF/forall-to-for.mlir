// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(scf-forall-to-for))' -split-input-file | FileCheck %s

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

// The pass should bail out cleanly and not crash here. `scf.forall` with outputs
// is not supported, but we should still handle the `forall` op with no results
// present in the same function.

func.func private @callee(%i: index)

// CHECK-LABEL: @shared_outs
func.func @shared_outs(%arg0: tensor<1xf32>, %ub: index) -> tensor<1xf32> {
  // CHECK: %{{.*}} = scf.forall
  %0 = scf.forall (%i) in (1) shared_outs(%out = %arg0) -> (tensor<1xf32>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %out into %out[%i] [1] [1] : tensor<1xf32> into tensor<1xf32>
    }
  }

  // CHECK: scf.for
  scf.forall (%i) in (%ub) {
    func.call @callee(%i) : (index) -> ()
  }
  return %0 : tensor<1xf32>
}
