// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// On a GPU the runtime lock is taken by one lane of a wavefront, so the region
// must additionally be given to one thread of the block at a time.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  llvm.func @critical_device(%x : !llvm.ptr, %xval : i32) attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>} {
    omp.critical {
      llvm.store %xval, %x : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define hidden void @critical_device(
// CHECK:         %[[MASK:.*]] = call i64 @__kmpc_warp_active_thread_mask()
// CHECK:         %[[TID:.*]] = call i32 @__kmpc_get_hardware_thread_id_in_block()
// CHECK:         %[[NTHREADS:.*]] = call i32 @__kmpc_get_hardware_num_threads_in_block()
// CHECK:         br label %omp.critical.serial.header

// CHECK:       omp.critical.serial.header:
// CHECK:         %[[TURN:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXT:.*]], %omp.critical.serial.sync ]
// CHECK:         %[[GO:.*]] = icmp slt i32 %[[TURN]], %[[NTHREADS]]
// CHECK:         br i1 %[[GO]], label %omp.critical.serial.turn, label %omp.critical.serial.exit

// CHECK:       omp.critical.serial.turn:
// CHECK:         %[[MINE:.*]] = icmp eq i32 %[[TID]], %[[TURN]]
// CHECK:         br i1 %[[MINE]], label %omp.critical.serial.region, label %omp.critical.serial.sync

// The lock must be acquired and released inside the region a single thread
// enters, not around the loop.
// CHECK:       omp.critical.serial.region:
// CHECK:         call void @__kmpc_critical(
// CHECK:       omp.critical.region:
// CHECK:         store i32 %{{.*}}, ptr %{{.*}}
// CHECK:       omp_region.finalize:
// CHECK:         call void @__kmpc_end_critical(
// CHECK:         br label %omp.critical.serial.sync

// CHECK:       omp.critical.serial.sync:
// CHECK:         call void @__kmpc_syncwarp(i64 %[[MASK]])
// CHECK:         %[[NEXT]] = add i32 %[[TURN]], 1
// CHECK:         br label %omp.critical.serial.header

// CHECK:       omp.critical.serial.exit:
