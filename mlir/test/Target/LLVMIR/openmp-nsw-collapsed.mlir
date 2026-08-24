// Test the behavior of nsw on loop IV increment for collapsed loops.

// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

//-------------------------------------------------------------------------//
// Case 1: Collapsed loop with trip count > INT_MAX (46341*46341)
//-------------------------------------------------------------------------//

// CHECK-LABEL: define void @collapsed_overflow_no_nsw
// CHECK: omp_collapsed.inc:
// CHECK: %omp_collapsed.next = add nuw i32 %omp_collapsed.iv, 1
// CHECK-NOT: nsw
module attributes {omp.integer_wrap_around = #omp.integer_wrap_around<integer_wrap_around = false>} {
  llvm.func @collapsed_overflow_no_nsw() {
    %lb = llvm.mlir.constant(0 : i32) : i32
    %ub1 = llvm.mlir.constant(46340 : i32) : i32
    %ub2 = llvm.mlir.constant(46340 : i32) : i32
    %step = llvm.mlir.constant(1 : i32) : i32
    omp.wsloop {
      omp.loop_nest (%iv1, %iv2) : i32 = (%lb, %lb) to (%ub1, %ub2) inclusive step (%step, %step) collapse(2) {
        omp.yield
      }
    }
    llvm.return
  }
}

// -----

//-------------------------------------------------------------------------//
// Case 2: Collapsed loop with trip count <= INT_MAX (99*99)
//-------------------------------------------------------------------------//

// CHECK-LABEL: define void @collapsed_small_with_nsw
// CHECK: omp_collapsed.inc:
// CHECK: %omp_collapsed.next = add nuw nsw i32 %omp_collapsed.iv, 1
module attributes {omp.integer_wrap_around = #omp.integer_wrap_around<integer_wrap_around = false>} {
  llvm.func @collapsed_small_with_nsw() {
    %lb = llvm.mlir.constant(0 : i32) : i32
    %ub = llvm.mlir.constant(99 : i32) : i32
    %step = llvm.mlir.constant(1 : i32) : i32
    omp.wsloop {
      omp.loop_nest (%iv1, %iv2) : i32 = (%lb, %lb) to (%ub, %ub) inclusive step (%step, %step) collapse(2) {
        omp.yield
      }
    }
    llvm.return
  }
}

// -----

//-------------------------------------------------------------------------//
// Case 3: Collapsed loop with dynamic trip count
//-------------------------------------------------------------------------//

// CHECK-LABEL: define void @collapsed_dynamic_no_nsw
// CHECK: omp_collapsed.inc:
// CHECK: %omp_collapsed.next = add nuw i32 %omp_collapsed.iv, 1
// CHECK-NOT: nsw
module attributes {omp.integer_wrap_around = #omp.integer_wrap_around<integer_wrap_around = false>} {
  llvm.func @collapsed_dynamic_no_nsw(%lb : i32, %ub1 : i32, %ub2 : i32, %step : i32) {
    omp.wsloop {
      omp.loop_nest (%iv1, %iv2) : i32 = (%lb, %lb) to (%ub1, %ub2) step (%step, %step) collapse(2) {
        omp.yield
      }
    }
    llvm.return
  }
}

// -----

//-------------------------------------------------------------------------//
// Case 4: Single loop with dynamic trip count
//-------------------------------------------------------------------------//

// CHECK-LABEL: define void @single_loop_with_nsw
// CHECK: omp_loop.inc:
// CHECK: %omp_loop.next = add nuw nsw i32 %omp_loop.iv, 1
module attributes {omp.integer_wrap_around = #omp.integer_wrap_around<integer_wrap_around = false>} {
  llvm.func @single_loop_with_nsw(%lb : i32, %ub : i32, %step : i32) {
    omp.wsloop {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
    llvm.return
  }
}
