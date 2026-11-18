// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Check reduction lowering in a declare target function.
module attributes {omp.is_target_device = true, omp.is_gpu = true, llvm.target_triple = "nvptx64-unknown-unknown"} {
  omp.declare_reduction @add_f32 : f32 init {
  ^bb0(%arg0: f32):
    %0 = llvm.mlir.constant(0.0 : f32) : f32
    omp.yield(%0 : f32)
  } combiner {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = llvm.fadd %arg0, %arg1 : f32
    omp.yield(%0 : f32)
  }

  // A sections construct with a nowait clause must not emit a barrier before or after the reduction.
  // CHECK-LABEL: define hidden void @sections_nowait_reduction(
  // CHECK-NOT:   @__kmpc_barrier
  // CHECK:       call i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(
  // CHECK-NOT:   @__kmpc_barrier
  // CHECK:       {{^}}}
  llvm.func @sections_nowait_reduction(%sum: !llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} {
    omp.sections nowait reduction(@add_f32 %sum -> %private : !llvm.ptr) {
      omp.section {
      ^bb0(%value: !llvm.ptr):
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }

  // A sections construct without a nowait clause should emit a barrier after the reduction.
  // CHECK-LABEL: define hidden void @sections_reduction(
  // CHECK:       call i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(
  // CHECK:       @__kmpc_barrier
  // CHECK:       {{^}}}
  llvm.func @sections_reduction(%sum: !llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} {
    omp.sections reduction(@add_f32 %sum -> %private : !llvm.ptr) {
      omp.section {
      ^bb0(%value: !llvm.ptr):
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }

  // A wsloop with nowait must not emit a barrier before or after the reduction.
  // CHECK-LABEL: define hidden void @wsloop_reduction_nowait(
  // CHECK-NOT:   @__kmpc_barrier
  // CHECK:       call i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(
  // CHECK-NOT:   @__kmpc_barrier
  // CHECK:       {{^}}}
  llvm.func @wsloop_reduction_nowait(%sum: !llvm.ptr, %lb: i64, %ub: i64, %step: i64) attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} {
    omp.wsloop nowait reduction(@add_f32 %sum -> %private : !llvm.ptr) {
      omp.loop_nest (%iv) : i64 = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
    llvm.return
  }

  // A wsloop without a nowait clause should emit a barrier after the reduction.
  // CHECK-LABEL: define hidden void @wsloop_reduction(
  // CHECK:       call i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(
  // CHECK:       @__kmpc_barrier
  // CHECK:       {{^}}}
  llvm.func @wsloop_reduction(%sum: !llvm.ptr, %lb: i64, %ub: i64, %step: i64) attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} {
    omp.wsloop reduction(@add_f32 %sum -> %private : !llvm.ptr) {
      omp.loop_nest (%iv) : i64 = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
    llvm.return
  }

  // A scope op with nowait clause must not emit a barrier before or after the reduction.
  // CHECK-LABEL: define hidden void @scope_reduction_nowait(
  // CHECK-NOT:   @__kmpc_barrier
  // CHECK:       call i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(
  // CHECK-NOT:   @__kmpc_barrier
  // CHECK:       {{^}}}
  llvm.func @scope_reduction_nowait(%ptr: !llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} {
    omp.parallel {
      omp.scope nowait reduction(@add_f32 %ptr -> %arg0 : !llvm.ptr) {
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }

  // A scope op without a nowait clause must emit a barrier after the reduction.
  // CHECK-LABEL: define hidden void @scope_reduction(
  // CHECK:       call i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(
  // CHECK:       @__kmpc_barrier
  // CHECK:       {{^}}}
  llvm.func @scope_reduction(%ptr: !llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} {
    omp.parallel {
      omp.scope reduction(@add_f32 %ptr -> %arg0 : !llvm.ptr) {
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}
