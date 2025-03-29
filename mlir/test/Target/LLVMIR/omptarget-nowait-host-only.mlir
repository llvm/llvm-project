// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Tests `target ... nowait` when code gen targets the host rather than a
// device.

module attributes {omp.is_target_device = false} {
  llvm.func @omp_target_nowait_() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x f32 {bindc_name = "x"} : (i64) -> !llvm.ptr
    %3 = omp.map.info var_ptr(%1 : !llvm.ptr, f32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "x"}
    omp.target nowait map_entries(%3 -> %arg0 : !llvm.ptr) {
      %4 = llvm.mlir.constant(5.000000e+00 : f32) : f32
      llvm.store %4, %arg0 : f32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: define void @omp_target_nowait_()
// CHECK-NOT: define {{.*}} @
// CHECK-NOT: call ptr @__kmpc_omp_target_task_alloc({{.*}})
// CHECK:  call void @__kmpc_omp_task_begin_if0
// CHECK-NEXT: call void @.omp_target_task_proxy_func
// CHECK: call void @__kmpc_omp_task_complete_if0
// https://github.com/llvm/llvm-project/issues/126949 exposes two issues
// 1. Empty target task proxy functions
// 2. When 1 fixed, it leads to a second problem of calling the omp target kernel twice
//    Once via the target task proxy function and a second time after the target task is done.
// The following checks check problem #2.
// functions. The following checks tests the fix for this issue.
// CHECK-NEXT:  br label %[[BLOCK_AFTER_OUTLINED_TARGET_TASK_BODY:.*]]
// CHECK:[[BLOCK_AFTER_OUTLINED_TARGET_TASK_BODY]]:
// CHECK-NEXT:  ret void

// Verify that we directly emit a call to the "target" region's body from the
// parent function of the the `omp.target` op.
// CHECK: define internal void @omp_target_nowait_..omp_par
// CHECK: call void @__omp_offloading_[[DEV:.*]]_[[FIL:.*]]_omp_target_nowait__l[[LINE:.*]](ptr {{.*}})
// CHECK-NEXT: br label %[[BLOCK_AFTER_TARGET_TASK_BODY:.*]]
// CHECK: [[BLOCK_AFTER_TARGET_TASK_BODY]]:
// CHECK-NEXT: ret void

// CHECK: define internal void @__omp_offloading_[[DEV]]_[[FIL]]_omp_target_nowait__l[[LINE]](ptr %[[ADDR_X:.*]])
// CHECK: store float 5{{.*}}, ptr %[[ADDR_X]], align 4

// The following check test for the fix of problem #1 as described in https://github.com/llvm/llvm-project/issues/126949
// CHECK: define internal void @.omp_target_task_proxy_func
// CHECK: call void @omp_target_nowait_..omp_par
