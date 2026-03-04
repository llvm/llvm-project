// Test that dist_schedule gets correctly translated with the correct schedule type and chunk size where appropriate

// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @distribute_dist_schedule_chunk_size(%lb : i32, %ub : i32, %step : i32, %x : i32) {
  // CHECK: call void @[[RUNTIME_FUNC:__kmpc_for_static_init_4u]](ptr @1, i32 %omp_global_thread_num, i32 91, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 1024)
  // We want to make sure that the next call is not another init builder.
  // CHECK-NOT: call void @[[RUNTIME_FUNC]]
  %1 = llvm.mlir.constant(1024: i32) : i32
  omp.distribute dist_schedule_static dist_schedule_chunk_size(%1 : i32) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// When a chunk size is present, we need to make sure the correct parallel accesses metadata is added
// CHECK: !2 = !{!"llvm.loop.parallel_accesses", !3}
// CHECK-NEXT: !3 = distinct !{}

// -----

llvm.func @distribute_dist_schedule(%lb : i32, %ub : i32, %step : i32, %x : i32) {
  // CHECK: call void @[[RUNTIME_FUNC:__kmpc_for_static_init_4u]](ptr @1, i32 %omp_global_thread_num, i32 92, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 0)
  // We want to make sure that the next call is not another init builder.
  // CHECK-NOT: call void @[[RUNTIME_FUNC]]
  omp.distribute dist_schedule_static {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}
