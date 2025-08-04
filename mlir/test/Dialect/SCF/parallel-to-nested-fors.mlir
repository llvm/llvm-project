// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(scf-parallel-for-to-nested-fors))' -split-input-file -verify-diagnostics | FileCheck %s

func.func private @callee(%i: index, %j: index)

func.func @two_iters(%lb1: index, %lb2: index, %ub1: index, %ub2: index, %step1: index, %step2: index) {
  scf.parallel (%i, %j) = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }
  // CHECK:           scf.for %[[VAL_0:.*]] = %[[ARG0:.*]] to %[[ARG2:.*]] step %[[ARG4:.*]] {
  // CHECK:             scf.for %[[VAL_1:.*]] = %[[ARG1:.*]] to %[[ARG3:.*]] step %[[ARG5:.*]] {
  // CHECK:               func.call @callee(%[[VAL_0]], %[[VAL_1]]) : (index, index) -> ()
  // CHECK:             }
  // CHECK:           }
  return
}

// -----

func.func private @callee(%i: index, %j: index)

func.func @repeated(%lb1: index, %lb2: index, %ub1: index, %ub2: index, %step1: index, %step2: index) {
  scf.parallel (%i, %j) = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }

  scf.parallel (%i, %j) = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }
  // CHECK:           scf.for %[[VAL_0:.*]] = %[[ARG0:.*]] to %[[ARG2:.*]] step %[[ARG4:.*]] {
  // CHECK:             scf.for %[[VAL_1:.*]] = %[[ARG1:.*]] to %[[ARG3:.*]] step %[[ARG5:.*]] {
  // CHECK:               func.call @callee(%[[VAL_0]], %[[VAL_1]]) : (index, index) -> ()
  // CHECK:             }
  // CHECK:           }
  // CHECK:           scf.for %[[VAL_2:.*]] = %[[ARG0]] to %[[ARG2]] step %[[ARG4]] {
  // CHECK:             scf.for %[[VAL_3:.*]] = %[[ARG1]] to %[[ARG3]] step %[[ARG5]] {
  // CHECK:               func.call @callee(%[[VAL_2]], %[[VAL_3]]) : (index, index) -> ()
  // CHECK:             }
  // CHECK:           }

  return
}

// -----

func.func private @callee(%i: index, %j: index, %k: index, %l: index)

func.func @nested(%lb1: index, %lb2: index, %lb3: index, %lb4: index, %ub1: index, %ub2: index, %ub3: index, %ub4: index, %step1: index, %step2: index, %step3: index, %step4: index) {
  scf.parallel (%i, %j) = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    scf.parallel (%k, %l) = (%lb3, %lb4) to (%ub3, %ub4) step (%step3, %step4) {
      func.call @callee(%i, %j, %k, %l) : (index, index, index, index) -> ()
    }
  }
  // CHECK:           scf.for %[[VAL_0:.*]] = %[[ARG0:.*]] to %[[ARG4:.*]] step %[[ARG8:.*]] {
  // CHECK:             scf.for %[[VAL_1:.*]] = %[[ARG1:.*]] to %[[ARG5:.*]] step %[[ARG9:.*]] {
  // CHECK:               scf.for %[[VAL_2:.*]] = %[[ARG2:.*]] to %[[ARG6:.*]] step %[[ARG10:.*]] {
  // CHECK:                 scf.for %[[VAL_3:.*]] = %[[ARG3:.*]] to %[[ARG7:.*]] step %[[ARG11:.*]] {
  // CHECK:                   func.call @callee(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) : (index, index, index, index) -> ()
  // CHECK:                 }
  // CHECK:               }
  // CHECK:             }
  // CHECK:           }
  return
}

// -----
func.func private @callee(%i: index, %j: index) -> i32

func.func @two_iters_with_reduce(%lb1: index, %lb2: index, %ub1: index, %ub2: index, %step1: index, %step2: index) -> i32 {
  %c0 = arith.constant 0 : i32
  // CHECK: scf.parallel
  %0 = scf.parallel (%i, %j) = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) init (%c0) -> i32 {
    %curr = func.call @callee(%i, %j) : (index, index) -> i32
    scf.reduce(%curr : i32) {
      ^bb0(%arg3: i32, %arg4: i32):
        %3 = arith.addi %arg3, %arg4 : i32
        scf.reduce.return %3 : i32
    }
  }
  return %0 : i32
}
