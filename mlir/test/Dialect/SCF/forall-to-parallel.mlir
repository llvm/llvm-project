// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(scf-forall-to-parallel))' -split-input-file | FileCheck %s

func.func private @callee(%i: index, %j: index)

// CHECK-LABEL: @two_iters
// CHECK-SAME: %[[UB1:.+]]: index, %[[UB2:.+]]: index
func.func @two_iters(%ub1: index, %ub2: index) {
  scf.forall (%i, %j) in (%ub1, %ub2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }

  // CHECK: scf.parallel (%[[IV1:.+]], %[[IV2:.+]]) = (%{{.*}}, %{{.*}}) to (%[[UB1]], %[[UB2]])
  // CHECK:   func.call @callee(%[[IV1]], %[[IV2]]) : (index, index) -> ()
  // CHECK:   scf.reduce
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

  // CHECK: scf.parallel (%[[IV1:.+]], %[[IV2:.+]]) = (%{{.*}}, %{{.*}}) to (%[[UB1]], %[[UB2]])
  // CHECK:   func.call @callee(%[[IV1]], %[[IV2]]) : (index, index) -> ()
  // CHECK:   scf.reduce
  scf.forall (%i, %j) in (%ub1, %ub2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }

  // CHECK: scf.parallel (%[[IV3:.+]], %[[IV4:.+]]) = (%{{.*}}, %{{.*}}) to (%[[UB1]], %[[UB2]])
  // CHECK:   func.call @callee(%[[IV3]], %[[IV4]])
  // CHECK:   scf.reduce
  return
}

// -----

func.func private @callee(%i: index, %j: index, %k: index, %l: index)

// CHECK-LABEL: @nested
// CHECK-SAME: %[[UB1:.+]]: index, %[[UB2:.+]]: index, %[[UB3:.+]]: index, %[[UB4:.+]]: index
func.func @nested(%ub1: index, %ub2: index, %ub3: index, %ub4: index) {
  // CHECK: scf.parallel (%[[IV1:.+]], %[[IV2:.+]]) = (%{{.*}}, %{{.*}}) to (%[[UB1]], %[[UB2]]) step (%{{.*}}, %{{.*}}) {
  // CHECK:   scf.parallel (%[[IV3:.+]], %[[IV4:.+]]) = (%{{.*}}, %{{.*}}) to (%[[UB3]], %[[UB4]]) step (%{{.*}}, %{{.*}}) {
  // CHECK:     func.call @callee(%[[IV1]], %[[IV2]], %[[IV3]], %[[IV4]])
  // CHECK:     scf.reduce
  // CHECK:   }
  // CHECK:   scf.reduce
  // CHECK: }
  scf.forall (%i, %j) in (%ub1, %ub2) {
    scf.forall (%k, %l) in (%ub3, %ub4) {
      func.call @callee(%i, %j, %k, %l) : (index, index, index, index) -> ()
    }
  }
  return
}

// -----

// CHECK-LABEL: @mapping_attr
func.func @mapping_attr() -> () {
  // CHECK: scf.parallel
  // CHECK:   scf.reduce
  // CHECK: {mapping = [#gpu.thread<x>]}

  %num_threads = arith.constant 100 : index

  scf.forall (%thread_idx) in (%num_threads) {
    scf.forall.in_parallel {
    }
  } {mapping = [#gpu.thread<x>]}
  return

}

// -----

// CHECK-LABEL: @forall_with_outputs
// CHECK-SAME: %[[ARG0:.+]]: tensor<32x32xf32>
func.func @forall_with_outputs(%arg0: tensor<32x32xf32>) -> tensor<8x112x32x32xf32> {
  // CHECK-NOT: scf.parallel
  // CHECK: %[[RES:.+]] = scf.forall{{.*}}shared_outs
  // CHECK: return %[[RES]] : tensor<8x112x32x32xf32>

  %0 = tensor.empty() : tensor<8x112x32x32xf32>
  %1 = scf.forall (%arg1, %arg2) in (8, 112) shared_outs(%arg3 = %0) -> (tensor<8x112x32x32xf32>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg0 into %arg3[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<8x112x32x32xf32>
    }
  }
  return %1 : tensor<8x112x32x32xf32>
}
