// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// tests firx for test-suite test: pr69183.f90. Makes sure we can handle inling
// private ops when the alloca block ends with a conditional branch.

omp.private {type = private} @_QFwsloop_privateEi_private_ref_i32 : i64

llvm.func @wsloop_private_(%arg0: !llvm.ptr {fir.bindc_name = "y"}) attributes {fir.internal_name = "_QPwsloop_private", frame_pointer = #llvm.framePointerKind<all>, target_cpu = "x86-64"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %3 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.mlir.constant(10 : i32) : i32
  %8 = llvm.mlir.constant(0 : i32) : i32
  %cond = llvm.mlir.constant(0 : i1) : i1
  llvm.cond_br %cond, ^bb1, ^bb2

^bb1:
  llvm.br ^bb3

^bb2:
  llvm.br ^bb3

^bb3:
    omp.wsloop private(@_QFwsloop_privateEi_private_ref_i32 %3 -> %arg2 : !llvm.ptr) {
      omp.loop_nest (%arg4) : i32 = (%8) to (%7) inclusive step (%6) {
        omp.yield
      }
    }
  llvm.return
}

// CHECK:   %[[INT:.*]] = alloca i32, i64 1, align 4
// CHECK:   br label %[[AFTER_ALLOC_BB:.*]]

// CHECK: [[AFTER_ALLOC_BB]]:
// CHECK:   br i1 false, label %[[BB1:.*]], label %[[BB2:.*]]

// CHECK: [[BB1]]:
// CHECK:   br label %[[BB3:.*]]

// CHECK: [[BB2]]:
// CHECK:   br label %[[BB3:.*]]

// CHECK: [[BB3]]:
// CHECK:   br label %[[OMP_PRIVATE_INIT:.*]]

// CHECK: [[OMP_PRIVATE_INIT]]:
// CHECK:   br label %omp_loop.preheader

