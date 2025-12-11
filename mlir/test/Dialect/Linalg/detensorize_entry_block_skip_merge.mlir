// RUN: mlir-opt %s -allow-unregistered-dialect -pass-pipeline="builtin.module(func.func(linalg-detensorize))" | FileCheck %s

module {
  memref.global "private" constant @__constant_4x4xf32 : memref<4x4xf32> = dense<8.899000e+01> {alignment = 64 : i64}
  func.func private @parallel_compute_fn_with_aligned_loops(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: memref<4x4xf32>, %arg11: memref<4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
    %1 = arith.cmpi slt, %0, %c4 : index
    cf.cond_br %1, ^bb2(%0 : index), ^bb6
  ^bb2(%2: index):  // pred: ^bb1
    %3 = arith.addi %2, %c1 : index
    cf.br ^bb3(%c0 : index)
  ^bb3(%4: index):  // 2 preds: ^bb2, ^bb4
    %5 = arith.cmpi slt, %4, %c4 : index
    cf.cond_br %5, ^bb4(%4 : index), ^bb5
  ^bb4(%6: index):  // pred: ^bb3
    %7 = arith.addi %6, %c1 : index
    %8 = memref.load %arg10[%2, %6] : memref<4x4xf32>
    %9 = llvm.intr.tanh(%8) : (f32) -> f32
    memref.store %9, %arg11[%2, %6] : memref<4x4xf32>
    cf.br ^bb3(%7 : index)
  ^bb5:  // pred: ^bb3
    cf.br ^bb1(%3 : index)
  ^bb6:  // pred: ^bb1
    return
  }
}

// CHECK-LABEL: @parallel_compute_fn_with_aligned_loops
// CHECK-SAME: (%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index, %[[ARG4:.+]]: index, %[[ARG5:.+]]: index, %[[ARG6:.+]]: index, %[[ARG7:.+]]: index, %[[ARG8:.+]]: index, %[[ARG9:.+]]: index, %[[ARG10:.+]]: memref<4x4xf32>, %[[ARG11:.+]]: memref<4x4xf32>)
// CHECK: cf.br ^{{.*}}
// CHECK: ^{{.*}}:
// CHECK: arith.cmpi slt
// CHECK: cf.cond_br
// CHECK: arith.addi
// CHECK: memref.load
// CHECK: llvm.intr.tanh
// CHECK: memref.store
// CHECK: return
