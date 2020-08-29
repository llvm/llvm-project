// RUN: mlir-opt -convert-openmp-to-llvm %s  -split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @branch_loop
func @branch_loop() {
  %start = constant 0 : index
  %end = constant 0 : index
  // CHECK: omp.parallel
  omp.parallel {
    // CHECK-NEXT: llvm.br ^[[BB1:.*]](%{{[0-9]+}}, %{{[0-9]+}} : !llvm.i64, !llvm.i64
    br ^bb1(%start, %end : index, index)
  // CHECK-NEXT: ^[[BB1]](%[[ARG1:[0-9]+]]: !llvm.i64, %[[ARG2:[0-9]+]]: !llvm.i64):{{.*}}
  ^bb1(%0: index, %1: index):
    // CHECK-NEXT: %[[CMP:[0-9]+]] = llvm.icmp "slt" %[[ARG1]], %[[ARG2]] : !llvm.i64
    %2 = cmpi "slt", %0, %1 : index
    // CHECK-NEXT: llvm.cond_br %[[CMP]], ^[[BB2:.*]](%{{[0-9]+}}, %{{[0-9]+}} : !llvm.i64, !llvm.i64), ^[[BB3:.*]]
    cond_br %2, ^bb2(%end, %end : index, index), ^bb3
  // CHECK-NEXT: ^[[BB2]](%[[ARG3:[0-9]+]]: !llvm.i64, %[[ARG4:[0-9]+]]: !llvm.i64):
  ^bb2(%3: index, %4: index):
    // CHECK-NEXT: llvm.br ^[[BB1]](%[[ARG3]], %[[ARG4]] : !llvm.i64, !llvm.i64)
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
