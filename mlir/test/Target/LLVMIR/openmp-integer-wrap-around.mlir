// Tests how the omp.integer_wrap_around module attribute controls nsw flags on IV arithmetic in the generated LLVM IR:
//   - <omp.integer_wrap_around = absent>  -> default behaviour, no nsw
//   - <omp.integer_wrap_around = true>    -> wrap around allowed (-fwrapv), no nsw
//   - <omp.integer_wrap_around = false>   -> wrap around disallowed, nsw emitted

// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

//-------------------------------------------------------------------------//
// Default behaviour: without the attribute, IV increment should NOT have nsw.
//-------------------------------------------------------------------------//

// CHECK-LABEL: define void @wsloop_default_no_nsw
// CHECK: omp_loop.header:
// CHECK: %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// CHECK: omp_loop.body:
// CHECK-NOT: add nsw
// CHECK: omp_loop.inc:
// CHECK: %omp_loop.next = add nuw i32 %omp_loop.iv, 1
llvm.func @wsloop_default_no_nsw(%lb : i32, %ub : i32, %step : i32) {
  omp.wsloop {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

//-----------------------------------------------------------------------------------//
// With omp.integer_wrap_around = true (-fwrapv), IV increment should NOT have nsw.
//----------------------------------------------------------------------------------//

// CHECK-LABEL: define void @wsloop_wrapv_no_nsw
// CHECK: omp_loop.header:
// CHECK: %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// CHECK: omp_loop.body:
// CHECK-NOT: add nsw
// CHECK: omp_loop.inc:
// CHECK: %omp_loop.next = add nuw i32 %omp_loop.iv, 1
module attributes {omp.integer_wrap_around = #omp.integer_wrap_around<integer_wrap_around = true>} {
  llvm.func @wsloop_wrapv_no_nsw(%lb : i32, %ub : i32, %step : i32) {
    omp.wsloop {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
    llvm.return
  }
}

// -----

//-----------------------------------------------------------------------//
// With omp.integer_wrap_around = false, IV increment should have nsw.
//-----------------------------------------------------------------------//
// CHECK-LABEL: define void @wsloop_nsw_iv
// CHECK: omp_loop.header:
// CHECK: %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]

// CHECK: omp_loop.body:
// CHECK: %[[IV_ADD:[0-9]+]] = add nsw i32 %omp_loop.iv, %{{[0-9]+}}
// CHECK: %[[MUL:[0-9]+]] = mul nsw i32 %[[IV_ADD]], %{{[0-9]+}}
// CHECK: %{{[0-9]+}} = add nsw i32 %[[MUL]], %{{[0-9]+}}

// CHECK: omp_loop.inc:
// CHECK: %omp_loop.next = add nuw nsw i32 %omp_loop.iv, 1
module attributes {omp.integer_wrap_around = #omp.integer_wrap_around<integer_wrap_around = false>} {
  llvm.func @wsloop_nsw_iv(%lb : i32, %ub : i32, %step : i32) {
    omp.wsloop {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
    llvm.return
  }
}
