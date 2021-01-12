// RUN: mlir-opt -convert-openmp-to-llvm %s  -split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @branch_loop
func @branch_loop() {
  %start = constant 0 : index
  %end = constant 0 : index
  // CHECK: omp.parallel
  omp.parallel {
    // CHECK-NEXT: llvm.br ^[[BB1:.*]](%{{[0-9]+}}, %{{[0-9]+}} : i64, i64
    br ^bb1(%start, %end : index, index)
  // CHECK-NEXT: ^[[BB1]](%[[ARG1:[0-9]+]]: i64, %[[ARG2:[0-9]+]]: i64):{{.*}}
  ^bb1(%0: index, %1: index):
    // CHECK-NEXT: %[[CMP:[0-9]+]] = llvm.icmp "slt" %[[ARG1]], %[[ARG2]] : i64
    %2 = cmpi "slt", %0, %1 : index
    // CHECK-NEXT: llvm.cond_br %[[CMP]], ^[[BB2:.*]](%{{[0-9]+}}, %{{[0-9]+}} : i64, i64), ^[[BB3:.*]]
    cond_br %2, ^bb2(%end, %end : index, index), ^bb3
  // CHECK-NEXT: ^[[BB2]](%[[ARG3:[0-9]+]]: i64, %[[ARG4:[0-9]+]]: i64):
  ^bb2(%3: index, %4: index):
    // CHECK-NEXT: llvm.br ^[[BB1]](%[[ARG3]], %[[ARG4]] : i64, i64)
    br ^bb1(%3, %4 : index, index)
  // CHECK-NEXT: ^[[BB3]]:
  ^bb3:
    omp.flush
    omp.barrier
    omp.taskwait
    omp.taskyield
    omp.terminator
  }
  return
}

// CHECK-LABEL: @wsloop
// CHECK: (%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64, %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: i64)
func @wsloop(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) {
  // CHECK: omp.parallel
  omp.parallel {
    // CHECK: omp.wsloop
    // CHECK: (%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]])
    "omp.wsloop"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) ( {
    // CHECK: ^{{.*}}(%[[ARG6:.*]]: i64, %[[ARG7:.*]]: i64):
    ^bb0(%arg6: index, %arg7: index):  // no predecessors
      // CHECK: "test.payload"(%[[ARG6]], %[[ARG7]]) : (i64, i64) -> ()
      "test.payload"(%arg6, %arg7) : (index, index) -> ()
      omp.yield
    }) {operand_segment_sizes = dense<[2, 2, 2, 0, 0, 0, 0, 0, 0]> : vector<9xi32>} : (index, index, index, index, index, index) -> ()
    omp.terminator
  }
  return
}
