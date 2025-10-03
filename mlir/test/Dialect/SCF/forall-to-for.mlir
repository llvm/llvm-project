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

  func.func @parallel_insert_slice(%arg0: tensor<100xf32>) -> tensor<100xf32> {
    %c100 = arith.constant 100 : index
    %res = scf.forall (%i) in (%c100) shared_outs(%s = %arg0) -> (tensor<100xf32>) {
      %t = "test.foo"() : () -> tensor<100xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %t into %s[%i] [100] [1] : tensor<100xf32> into tensor<100xf32>
      }
    }
    return %res : tensor<100xf32>
  }
// CHECK-LABEL:   func.func @parallel_insert_slice(
// CHECK-SAME:      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<100xf32>) -> tensor<100xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_4:.*]] = %[[VAL_2]] to %[[VAL_1]] step %[[VAL_3]] {
// CHECK:             %[[VAL_5:.*]] = "test.foo"() : () -> tensor<100xf32>
// CHECK:           }
// CHECK:           return %[[VAL_0]] : tensor<100xf32>
// CHECK:         }