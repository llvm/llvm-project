// Check no-wrap flags in both collapsed-loop mappings.

// RUN: mlir-translate -mlir-to-llvmir -openmp-disable-virtual-collapse -split-input-file %s | FileCheck %s --check-prefixes=CHECK,COMMON
// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s --check-prefixes=VIRTUAL,COMMON

//-------------------------------------------------------------------------//
// Case 1: Collapsed loop with trip count > INT_MAX (46341*46341)
//-------------------------------------------------------------------------//

// VIRTUAL-LABEL: define void @collapsed_overflow_no_nsw
// VIRTUAL: omp_collapsed.init:
// VIRTUAL: omp_collapsed.decode:
// VIRTUAL: %[[OUTER_IV:omp_loop.iv[0-9]*]] = phi i32
// VIRTUAL-SAME: {{.*}}, [ %[[OUTER_NEXT:[^ ]+]], %{{[^ ]+}} ]
// VIRTUAL: %[[INNER_IV:omp_loop.iv[0-9]*]] = phi i32
// VIRTUAL-SAME: {{.*}}, [ %[[INNER_NEXT:[^ ]+]], %{{[^ ]+}} ]
// VIRTUAL: %[[INNER_NEXT]] = add nuw nsw i32 %[[INNER_IV]], 1{{(,|$)}}
// VIRTUAL: %[[OUTER_NEXT]] = add nuw nsw i32 %[[OUTER_IV]], 1{{(,|$)}}
// VIRTUAL: = sub nuw i32 %omp_collapsed.row.remaining.0,
// VIRTUAL: ret void
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

// VIRTUAL-LABEL: define void @collapsed_small_with_nsw
// VIRTUAL: omp_collapsed.init:
// VIRTUAL: omp_collapsed.decode:
// VIRTUAL: %[[OUTER_IV:omp_loop.iv[0-9]*]] = phi i32
// VIRTUAL-SAME: {{.*}}, [ %[[OUTER_NEXT:[^ ]+]], %{{[^ ]+}} ]
// VIRTUAL: %[[INNER_IV:omp_loop.iv[0-9]*]] = phi i32
// VIRTUAL-SAME: {{.*}}, [ %[[INNER_NEXT:[^ ]+]], %{{[^ ]+}} ]
// VIRTUAL: %[[INNER_NEXT]] = add nuw nsw i32 %[[INNER_IV]], 1{{(,|$)}}
// VIRTUAL: %[[OUTER_NEXT]] = add nuw nsw i32 %[[OUTER_IV]], 1{{(,|$)}}
// VIRTUAL: = sub nuw i32 %omp_collapsed.row.remaining.0,
// VIRTUAL: ret void
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

// COMMON-LABEL: define void @collapsed_dynamic_no_nsw
// COMMON: omp_collapsed.inc:
// COMMON: %omp_collapsed.next = add nuw i32 %omp_collapsed.iv, 1
// COMMON-NOT: nsw
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

// COMMON-LABEL: define void @single_loop_with_nsw
// COMMON: omp_loop.inc:
// COMMON: %omp_loop.next = add nuw nsw i32 %omp_loop.iv, 1
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

// -----

// A large Virtual budget is unsigned even when source IVs cannot wrap.
// For 2*INT_MAX iterations on three static threads, the second range has
// Begin=Count=1431655765, so Count+Begin%INT_MAX exceeds signed i32.
// VIRTUAL-LABEL: define void @collapsed_large_remaining
// VIRTUAL: call void @__kmpc_for_static_init_4u
// VIRTUAL: omp_collapsed.init:
// VIRTUAL: omp_collapsed.decode:
// VIRTUAL: [[INNER_BEGIN:%[^ ]+]] = urem i32 %omp_collapsed.range.begin, 2147483647
// VIRTUAL: %omp_collapsed.row.remaining = add i32 %omp_collapsed.range.count, [[INNER_BEGIN]]{{(,|$)}}
// VIRTUAL: %[[OUTER_IV:omp_loop.iv[0-9]*]] = phi i32
// VIRTUAL-SAME: {{.*}}, [ %[[OUTER_NEXT:[^ ]+]], %{{[^ ]+}} ]
// VIRTUAL: %omp_collapsed.row.is_final = icmp ule i32 %omp_collapsed.row.remaining.0, 2147483647{{(,|$)}}
// VIRTUAL-NEXT: %omp_collapsed.row.stop = select i1 %omp_collapsed.row.is_final, i32 %omp_collapsed.row.remaining.0, i32 2147483647{{(,|$)}}
// VIRTUAL: %[[INNER_IV:omp_loop.iv[0-9]*]] = phi i32
// VIRTUAL-SAME: {{.*}}, [ %[[INNER_NEXT:[^ ]+]], %{{[^ ]+}} ]
// VIRTUAL: %[[INNER_NEXT]] = add nuw nsw i32 %[[INNER_IV]], 1{{(,|$)}}
// VIRTUAL: %[[OUTER_NEXT]] = add nuw nsw i32 %[[OUTER_IV]], 1{{(,|$)}}
// VIRTUAL: = sub nuw i32 %omp_collapsed.row.remaining.0, 2147483647{{(,|$)}}
// VIRTUAL: ret void
module attributes {omp.integer_wrap_around = #omp.integer_wrap_around<integer_wrap_around = false>} {
  llvm.func @collapsed_large_remaining() {
    %lb = llvm.mlir.constant(0 : i32) : i32
    %two = llvm.mlir.constant(2 : i32) : i32
    %max = llvm.mlir.constant(2147483647 : i32) : i32
    %step = llvm.mlir.constant(1 : i32) : i32
    omp.wsloop schedule(static) {
      omp.loop_nest (%i, %j) : i32 = (%lb, %lb) to (%two, %max) step (%step, %step) collapse(2) {
        omp.yield
      }
    }
    llvm.return
  }
}
