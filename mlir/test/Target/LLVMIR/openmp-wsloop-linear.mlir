// Ensure that omp.wsloop with the linear clause is translated correctly.
// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// -----

// %.linear_result must appear only in the loop body and in the
// linear_lastiter_exit block, where it's used to update the original
// variable.
// CHECK-LABEL: void @wsloop_linear_post_use({{.*}})
// CHECK:           %.linear_result = alloca i32
// CHECK-NOT:       %.linear_result
// CHECK:         omp_loop.body:
// CHECK:           %.linear_result
// CHECK:         omp_loop.inc:
// CHECK-NOT:       %.linear_result
// CHECK:         omp_loop.linear_lastiter_exit:
// CHECK:           load i32, ptr %.linear_result
// CHECK-NOT:       %.linear_result

llvm.func @wsloop_linear_post_use(%lb : i32, %ub : i32, %step : i32,
                                  %x : !llvm.ptr, %out : !llvm.ptr) {
  omp.wsloop linear(%x : !llvm.ptr = %step : i32) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      %cur = llvm.load %x : !llvm.ptr -> i32
      llvm.store %cur, %out : i32, !llvm.ptr
      omp.yield
    }
  } {linear_var_types = [i32]}
  %after = llvm.load %x : !llvm.ptr -> i32
  llvm.store %after, %out : i32, !llvm.ptr
  llvm.return
}
