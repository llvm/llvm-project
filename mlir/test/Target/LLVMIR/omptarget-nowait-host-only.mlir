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
// Verify that we directly emit a call to the "target" region's body from the
// parent function of the the `omp.target` op.
// CHECK: call void @__omp_offloading_[[DEV:.*]]_[[FIL:.*]]_omp_target_nowait__l[[LINE:.*]](ptr {{.*}})
// CHECK-NEXT: ret void

// CHECK: define internal void @__omp_offloading_[[DEV]]_[[FIL]]_omp_target_nowait__l[[LINE]](ptr %[[ADDR_X:.*]])
// CHECK: store float 5{{.*}}, ptr %[[ADDR_X]], align 4
