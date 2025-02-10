// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Tests regression uncovered by "1009/1009_0029.f90" (from the Fujitsu test
// suite). This test replicates a simplified version of the block structure
// produced by the Fujitsu test.

llvm.func @test_block_structure() {
  %i1 = llvm.mlir.constant(1 : index) : i1
  %i64 = llvm.mlir.constant(1 : index) : i64
  llvm.br ^bb1(%i64, %i64 : i64, i64)

^bb1(%20: i64, %21: i64):  // 2 preds: ^bb0, ^bb5
  llvm.cond_br %i1, ^bb2, ^bb6

^bb2:  // pred: ^bb1
  llvm.br ^bb3(%i64, %i64 : i64, i64)

^bb3(%25: i64, %26: i64):  // 2 preds: ^bb2, ^bb4
  llvm.cond_br %i1, ^bb4, ^bb5

^bb4:  // pred: ^bb3
  omp.wsloop {
    omp.loop_nest (%arg0) : i64 = (%i64) to (%i64) inclusive step (%i64) {
      omp.yield
    }
  }
  llvm.br ^bb1(%i64, %i64 : i64, i64)

^bb5:  // pred: ^bb3
  llvm.br ^bb1(%i64, %i64 : i64, i64)

^bb6:  // pred: ^bb1
  llvm.return
}

// CHECK: define void @test_block_structure
// CHECK:   br label %[[AFTER_ALLOCA:.*]]

// CHECK: [[AFTER_ALLOCA:]]:
// CHECK:   br label %[[BB1:.*]]

// CHECK: [[BB1:]]:
// CHECK:   %{{.*}} = phi i64 
// CHECK:   br i1 true, label %[[BB2:.*]], label %{{.*}}

// CHECK: [[BB2]]:
// CHECK:   br label %[[BB3:.*]]

// CHECK: [[BB3]]:
// CHECK:   %{{.*}} = phi i64 
// CHECK:   br i1 true, label %[[BB4:.*]], label %{{.*}}

// CHECK: [[BB4]]:
// CHECK:   br label %omp_loop.preheader
