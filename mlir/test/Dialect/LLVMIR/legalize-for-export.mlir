// RUN: mlir-opt -llvm-legalize-for-export %s | FileCheck %s

// Verifies that duplicate successor with different arguments are deduplicated
// by introducing a new block that forwards its arguments to the original
// successor through an unconditional branch.
// CHECK-LABEL: @repeated_successor_different_args
llvm.func @repeated_successor_different_args(%arg0: i1, %arg1: i32, %arg2: i32) {
  // CHECK: llvm.cond_br %{{.*}}, ^[[BB1:.*]]({{.*}}), ^[[BB2:.*]]({{.*}})
  llvm.cond_br %arg0, ^bb1(%arg1: i32), ^bb1(%arg2: i32)

// CHECK: ^[[BB1]]({{.*}}):
^bb1(%arg3: i32):
  llvm.return

// CHECK: ^[[BB2]](%[[ARG:.*]]: i32):
// CHECK:  llvm.br ^[[BB1]](%[[ARG]] : i32)
}

// Verifies that duplicate successors without arguments do not lead to the
// introduction of new blocks during legalization.
// CHECK-LABEL: @repeated_successor_no_args
llvm.func @repeated_successor_no_args(%arg0: i1) {
  // CHECK: llvm.cond_br
  llvm.cond_br %arg0, ^bb1, ^bb1

// CHECK: ^{{.*}}:
^bb1:
  llvm.return

// CHECK-NOT: ^{{.*}}:
}

// CHECK: @repeated_successor_openmp
llvm.func @repeated_successor_openmp(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i1) {
  omp.wsloop (%arg4) : i64 = (%arg0) to (%arg1) step (%arg2)  {
    // CHECK: llvm.cond_br %{{.*}}, ^[[BB1:.*]]({{.*}}), ^[[BB2:.*]]({{.*}})
    llvm.cond_br %arg3, ^bb1(%arg0 : i64), ^bb1(%arg1 : i64)
  // CHECK: ^[[BB1]]
  ^bb1(%0: i64):  // 2 preds: ^bb0, ^bb0
    omp.yield
  // CHECK: ^[[BB2]](%[[ARG:.*]]: i64):
  // CHECK:  llvm.br ^[[BB1]](%[[ARG]] : i64)
  }
  llvm.return
}
