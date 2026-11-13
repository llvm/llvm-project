// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// On a GPU, whose threads of a warp may run in lockstep, the threads of each
// warp enter a critical region one at a time, so that threads sharing an
// instruction stream never spin for a lock that one of them holds.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true, omp.is_gpu = true} {
  // CHECK-LABEL: define {{.*}}void @omp_critical_amdgpu(
  // CHECK:       %omp.lanes.mask = call i64 @__kmpc_warp_active_thread_mask()
  // CHECK:       omp.lanes.header:
  // CHECK:         %omp.lanes.waiting = phi i64 [ %omp.lanes.mask, %{{.*}} ], [ %omp.lanes.rest, %omp.lanes.latch ]
  // CHECK:       omp.lanes.body:
  // CHECK:         call void @__kmpc_critical(
  // CHECK:         store i32 %{{.*}}, ptr %{{.*}}
  // CHECK:         call void @__kmpc_end_critical(
  // CHECK:       omp.lanes.latch:
  // CHECK-NEXT:    call void @__kmpc_syncwarp(i64 %omp.lanes.mask)
  // CHECK:       omp.lanes.exit:
  // CHECK-NEXT:    ret void
  llvm.func @omp_critical_amdgpu(%x : !llvm.ptr, %xval : i32) attributes {omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>} {
    omp.critical {
      llvm.store %xval, %x : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// -----

module attributes {llvm.target_triple = "nvptx64-nvidia-cuda", omp.is_target_device = true, omp.is_gpu = true} {
  // CHECK-LABEL: define {{.*}}void @omp_critical_nvptx(
  // CHECK:       %omp.lanes.mask = call i64 @__kmpc_warp_active_thread_mask()
  // CHECK:       omp.lanes.body:
  // CHECK:         call void @__kmpc_critical(
  // CHECK:         call void @__kmpc_end_critical(
  // CHECK:       omp.lanes.latch:
  // CHECK-NEXT:    call void @__kmpc_syncwarp(i64 %omp.lanes.mask)
  llvm.func @omp_critical_nvptx(%x : !llvm.ptr, %xval : i32) attributes {omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>} {
    omp.critical {
      llvm.store %xval, %x : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}
