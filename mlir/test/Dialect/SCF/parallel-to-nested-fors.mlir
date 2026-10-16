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

// -----

func.func private @callee(%i: index, %j: index)

func.func @unsigned(%lb1: index, %lb2: index, %ub1: index, %ub2: index, %step1: index, %step2: index) {
  scf.parallel unsigned (%i, %j) = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  }
  // CHECK:           scf.for unsigned %[[VAL_0:.*]] = %[[ARG0:.*]] to %[[ARG2:.*]] step %[[ARG4:.*]] {
  // CHECK:             scf.for unsigned %[[VAL_1:.*]] = %[[ARG1:.*]] to %[[ARG3:.*]] step %[[ARG5:.*]] {
  // CHECK:               func.call @callee(%[[VAL_0]], %[[VAL_1]]) : (index, index) -> ()
  return
}

// -----

// CHECK: #[[$LOOP_UNROLL_DISABLE:.*]] = #llvm.loop_unroll<disable = true>
// CHECK: #[[$NO_UNROLL:.*]] = #llvm.loop_annotation<unroll = #[[$LOOP_UNROLL_DISABLE]]>
// CHECK-LABEL: func.func @discardable_attrs_1d
// CHECK:         scf.for %[[IV:.*]] = {{.*}} {
// CHECK:           func.call @callee1(%[[IV]]) : (index) -> ()
// CHECK:         } {llvm.loop_annotation = #[[$NO_UNROLL]], test.attr = "foo"}
#no_unroll = #llvm.loop_annotation<unroll = <disable = true>>
func.func private @callee1(%i: index)
func.func @discardable_attrs_1d(%lb: index, %ub: index, %step: index) {
  scf.parallel (%i) = (%lb) to (%ub) step (%step) {
    func.call @callee1(%i) : (index) -> ()
  } {llvm.loop_annotation = #no_unroll, test.attr = "foo"}
  return
}

// -----

// A multi-dimensional scf.parallel lowers to a loop nest, and discardable
// attributes go to the innermost loop only.
// CHECK: #[[$LOOP_UNROLL_DISABLE_2D:.*]] = #llvm.loop_unroll<disable = true>
// CHECK: #[[$NO_UNROLL_2D:.*]] = #llvm.loop_annotation<unroll = #[[$LOOP_UNROLL_DISABLE_2D]]>
// CHECK-LABEL: func.func @discardable_attrs_2d
// CHECK:         scf.for %[[IV0:.*]] = {{.*}} {
// CHECK:           scf.for %[[IV1:.*]] = {{.*}} {
// CHECK:             func.call @callee(%[[IV0]], %[[IV1]]) : (index, index) -> ()
// CHECK:           } {llvm.loop_annotation = #[[$NO_UNROLL_2D]]}
// CHECK-NEXT:    }
// CHECK-NEXT:    return
#no_unroll = #llvm.loop_annotation<unroll = <disable = true>>
func.func private @callee(%i: index, %j: index)
func.func @discardable_attrs_2d(%lb1: index, %lb2: index, %ub1: index, %ub2: index, %step1: index, %step2: index) {
  scf.parallel (%i, %j) = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    func.call @callee(%i, %j) : (index, index) -> ()
  } {llvm.loop_annotation = #no_unroll}
  return
}
