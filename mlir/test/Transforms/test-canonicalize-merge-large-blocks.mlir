 // RUN: mlir-opt -pass-pipeline='builtin.module(llvm.func(canonicalize{region-simplify=aggressive}))' %s | FileCheck %s

llvm.func @foo(%arg0: i64)

llvm.func @rand() -> i1

// CHECK-LABEL: func @large_merge_block(
llvm.func @large_merge_block(%arg0: i64) {
  // CHECK:  %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK:  %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:  %[[C2:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK:  %[[C3:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK:  %[[C4:.*]] = llvm.mlir.constant(4 : i64) : i64

  // CHECK:  llvm.cond_br %5, ^bb1(%[[C1]], %[[C3]], %[[C4]], %[[C2]] : i64, i64, i64, i64), ^bb1(%[[C4]], %[[C2]], %[[C1]], %[[C3]] : i64, i64, i64, i64)
  // CHECK: ^bb{{.*}}(%[[arg0:.*]]: i64, %[[arg1:.*]]: i64, %[[arg2:.*]]: i64, %[[arg3:.*]]: i64):
  // CHECK:    llvm.cond_br %{{.*}}, ^bb2(%[[arg0]] : i64), ^bb2(%[[arg3]] : i64)
  // CHECK: ^bb{{.*}}(%11: i64):
  // CHECK:    llvm.br ^bb{{.*}}
  // CHECK: ^bb{{.*}}:
  // CHECK:   llvm.call
  // CHECK:   llvm.cond_br {{.*}}, ^bb{{.*}}(%[[arg1]] : i64), ^bb{{.*}}(%[[arg2]] : i64)
  // CHECK: ^bb{{.*}}:
  // CHECK:   llvm.call
  // CHECK    llvm.br ^bb{{.*}}

  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = llvm.mlir.constant(1 : i64) : i64
  %2 = llvm.mlir.constant(2 : i64) : i64
  %3 = llvm.mlir.constant(3 : i64) : i64
  %4 = llvm.mlir.constant(4 : i64) : i64
  %10 = llvm.icmp "eq" %arg0, %0 : i64
  llvm.cond_br %10, ^bb1, ^bb14
^bb1:  // pred: ^bb0
  %11 = llvm.call @rand() : () -> i1
  llvm.cond_br %11, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  llvm.call @foo(%1) : (i64) -> ()
  llvm.br ^bb4
^bb3:  // pred: ^bb1
  llvm.call @foo(%2) : (i64) -> ()
  llvm.br ^bb4
^bb4:  // 2 preds: ^bb2, ^bb3
  %14 = llvm.call @rand() : () -> i1
  llvm.cond_br %14, ^bb5, ^bb6
^bb5:  // pred: ^bb4
  llvm.call @foo(%3) : (i64) -> ()
  llvm.br ^bb13
^bb6:  // pred: ^bb4
  llvm.call @foo(%4) : (i64) -> ()
  llvm.br ^bb13
^bb13:  // 2 preds: ^bb11, ^bb12
  llvm.br ^bb27
^bb14:  // pred: ^bb0
  %23 = llvm.call @rand() : () -> i1
  llvm.cond_br %23, ^bb15, ^bb16
^bb15:  // pred: ^bb14
  llvm.call @foo(%4) : (i64) -> ()
  llvm.br ^bb17
^bb16:  // pred: ^bb14
  llvm.call @foo(%3) : (i64) -> ()
  llvm.br ^bb17
^bb17:  // 2 preds: ^bb15, ^bb16
  %26 = llvm.call @rand() : () -> i1
  llvm.cond_br %26, ^bb18, ^bb19
^bb18:  // pred: ^bb17
  llvm.call @foo(%2) : (i64) -> ()
  llvm.br ^bb26
^bb19:  // pred: ^bb17
  llvm.call @foo(%1) : (i64) -> ()
  llvm.br ^bb26
^bb26:  // 2 preds: ^bb24, ^bb25
  llvm.br ^bb27
^bb27:  // 2 preds: ^bb13, ^bb26
  llvm.return
}
