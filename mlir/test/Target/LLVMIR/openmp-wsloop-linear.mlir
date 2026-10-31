// Ensure that omp.wsloop with the linear clause is translated correctly.
// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// -----

// The private copy must appear only in the loop body and in the
// linear_lastiter_exit block, where it's used to update the original
// variable.
// CHECK-LABEL: void @wsloop_linear_post_use({{.*}})
// CHECK:           %[[PRIV:omp.private.alloc]] = alloca i32
// CHECK-NOT:       %[[PRIV]]
// CHECK:         omp_loop.body:
// CHECK:           store i32 %{{.*}}, ptr %[[PRIV]]
// CHECK:         omp.loop_nest.region:
// CHECK:           load i32, ptr %[[PRIV]]
// CHECK:         omp_loop.inc:
// CHECK-NOT:       %[[PRIV]]
// CHECK:         omp_loop.linear_lastiter_exit:
// CHECK:           load i32, ptr %[[PRIV]]
// CHECK-NOT:       %[[PRIV]]

omp.private {type = linear} @lin.i32 : i32

llvm.func @wsloop_linear_post_use(%lb : i32, %ub : i32, %step : i32,
                                  %x : !llvm.ptr, %out : !llvm.ptr) {
  omp.wsloop linear(@lin.i32 %x -> %lin0 : !llvm.ptr = %step : i32) linear_var_types([i32]) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      %cur = llvm.load %lin0 : !llvm.ptr -> i32
      llvm.store %cur, %out : i32, !llvm.ptr
      omp.yield
    }
  }
  %after = llvm.load %x : !llvm.ptr -> i32
  llvm.store %after, %out : i32, !llvm.ptr
  llvm.return
}
