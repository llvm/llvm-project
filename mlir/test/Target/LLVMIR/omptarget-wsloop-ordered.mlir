// The aim of the test is to check that a worksharing loop with an ordered
// clause is emitted as a dispatch loop on a target device. The
// __kmpc_*_static_loop_* entries cannot order iterations, so the loop has to
// go through __kmpc_dispatch_next / __kmpc_dispatch_fini instead.

// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true } {
  llvm.func @target_wsloop_ordered(%arg0: !llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
    %loop_ub = llvm.mlir.constant(9 : i32) : i32
    %loop_lb = llvm.mlir.constant(0 : i32) : i32
    %loop_step = llvm.mlir.constant(1 : i32) : i32
    omp.wsloop ordered(0) {
      omp.loop_nest (%loop_cnt) : i32 = (%loop_lb) to (%loop_ub) inclusive step (%loop_step) {
        llvm.store %loop_cnt, %arg0 : i32, !llvm.ptr
        omp.yield
      }
    }
    llvm.return
  }
}

// The three calls are emitted into different blocks, so their order in the
// output follows the block layout rather than the loop structure.

// CHECK-LABEL: define hidden void @target_wsloop_ordered
// CHECK-DAG:     call void @__kmpc_dispatch_init_4u
// CHECK-DAG:     call i32 @__kmpc_dispatch_next_4u
// CHECK-DAG:     call void @__kmpc_dispatch_fini_4u
// CHECK-NOT:     @__kmpc_for_static_loop

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true } {
  llvm.func @target_wsloop_no_ordered(%arg0: !llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
    %loop_ub = llvm.mlir.constant(9 : i32) : i32
    %loop_lb = llvm.mlir.constant(0 : i32) : i32
    %loop_step = llvm.mlir.constant(1 : i32) : i32
    omp.wsloop {
      omp.loop_nest (%loop_cnt) : i32 = (%loop_lb) to (%loop_ub) inclusive step (%loop_step) {
        llvm.store %loop_cnt, %arg0 : i32, !llvm.ptr
        omp.yield
      }
    }
    llvm.return
  }
}

// Without the clause the loop keeps taking the static-loop entry.

// CHECK-LABEL: define hidden void @target_wsloop_no_ordered
// CHECK:         call void @__kmpc_for_static_loop_4u
// CHECK-NOT:     @__kmpc_dispatch_init
