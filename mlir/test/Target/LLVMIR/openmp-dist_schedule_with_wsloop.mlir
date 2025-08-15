// Test that dist_schedule gets correctly translated with the correct schedule type and chunk size where appropriate while using workshare loops.

// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @distribute_wsloop_dist_scheule_chunked_schedule_chunked(%n: i32, %teams: i32, %threads: i32) {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %dcs = llvm.mlir.constant(1024 : i32) : i32
  %scs = llvm.mlir.constant(64 : i32) : i32

  omp.teams num_teams(to %teams : i32) thread_limit(%threads : i32) {
    omp.parallel {
      omp.distribute dist_schedule_static dist_schedule_chunk_size(%dcs : i32) {
        omp.wsloop schedule(static = %scs : i32) {
          omp.loop_nest (%i) : i32 = (%0) to (%n) step (%1) {
            omp.yield
          }
        } {omp.composite}
      } {omp.composite}
      omp.terminator
    } {omp.composite}
    omp.terminator
  }
  llvm.return
}
// CHECK: define internal void @distribute_wsloop_dist_scheule_chunked_schedule_chunked..omp_par(ptr %0) {
// CHECK: call void @__kmpc_for_static_init_4u(ptr @1, i32 %omp_global_thread_num9, i32 33, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 64)
// CHECK: call void @__kmpc_for_static_init_4u(ptr @1, i32 %omp_global_thread_num9, i32 91, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 1024)

llvm.func @distribute_wsloop_dist_scheule_chunked(%n: i32, %teams: i32, %threads: i32) {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %dcs = llvm.mlir.constant(1024 : i32) : i32

  omp.teams num_teams(to %teams : i32) thread_limit(%threads : i32) {
    omp.parallel {
      omp.distribute dist_schedule_static dist_schedule_chunk_size(%dcs : i32) {
        omp.wsloop schedule(static) {
          omp.loop_nest (%i) : i32 = (%0) to (%n) step (%1) {
            omp.yield
          }
        } {omp.composite}
      } {omp.composite}
      omp.terminator
    } {omp.composite}
    omp.terminator
  }
  llvm.return
}
// CHECK: define internal void @distribute_wsloop_dist_scheule_chunked..omp_par(ptr %0) {
// CHECK: call void @__kmpc_for_static_init_4u(ptr @1, i32 %omp_global_thread_num9, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 0)
// CHECK: call void @__kmpc_for_static_init_4u(ptr @1, i32 %omp_global_thread_num9, i32 91, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 1024)

llvm.func @distribute_wsloop_schedule_chunked(%n: i32, %teams: i32, %threads: i32) {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %scs = llvm.mlir.constant(64 : i32) : i32

  omp.teams num_teams(to %teams : i32) thread_limit(%threads : i32) {
    omp.parallel {
      omp.distribute dist_schedule_static {
        omp.wsloop schedule(static = %scs : i32) {
          omp.loop_nest (%i) : i32 = (%0) to (%n) step (%1) {
            omp.yield
          }
        } {omp.composite}
      } {omp.composite}
      omp.terminator
    } {omp.composite}
    omp.terminator
  }
  llvm.return
}
// CHECK: define internal void @distribute_wsloop_schedule_chunked..omp_par(ptr %0) {
// CHECK: call void @__kmpc_for_static_init_4u(ptr @1, i32 %omp_global_thread_num9, i32 33, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 64)
// CHECK: call void @__kmpc_for_static_init_4u(ptr @1, i32 %omp_global_thread_num9, i32 92, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 0)

llvm.func @distribute_wsloop_no_chunks(%n: i32, %teams: i32, %threads: i32) {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32

  omp.teams num_teams(to %teams : i32) thread_limit(%threads : i32) {
    omp.parallel {
      omp.distribute dist_schedule_static {
        omp.wsloop schedule(static) {
          omp.loop_nest (%i) : i32 = (%0) to (%n) step (%1) {
            omp.yield
          }
        } {omp.composite}
      } {omp.composite}
      omp.terminator
    } {omp.composite}
    omp.terminator
  }
  llvm.return
}
// CHECK: define internal void @distribute_wsloop_no_chunks..omp_par(ptr %0) {
// CHECK: call void @__kmpc_dist_for_static_init_4u(ptr @1, i32 %omp_global_thread_num9, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.distupperbound, ptr %p.stride, i32 1, i32 0)
// CHECK: call void @__kmpc_dist_for_static_init_4u(ptr @1, i32 %omp_global_thread_num9, i32 92, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.distupperbound10, ptr %p.stride, i32 1, i32 0)
