// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Regression test: a dead saveIP/restoreIP pair in
// OpenMPIRBuilder::createTaskloop PostOutlineCB could introduce stale
// IRBuilder debug-location state and cause an intermittent crash during
// finalize(). Removing the dead pair avoids that restore path entirely.
// Verifies that an i32 loop is lowered correctly with a __kmpc_taskloop call.

omp.private {type = private} @_QPtest_taskloop_boundsEi_private_i32 : i32

llvm.func @_QPtest_taskloop_bounds() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %lb = llvm.mlir.constant(1 : i32) : i32
  %ub = llvm.mlir.constant(10 : i32) : i32
  %step = llvm.mlir.constant(1 : i32) : i32
  omp.taskloop private(@_QPtest_taskloop_boundsEi_private_i32 %1 -> %arg0 : !llvm.ptr) {
    omp.loop_nest (%arg1) : i32 = (%lb) to (%ub) inclusive step (%step) {
      llvm.store %arg1, %arg0 : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK-LABEL: define void @_QPtest_taskloop_bounds(
// CHECK:         call void @__kmpc_taskloop(
