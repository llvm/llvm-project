// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Check that for a device generic-mode `omp.parallel` nested in a loop, its
// captured-arguments aggregate is allocated in shared memory at the call site
// (inside the loop body), balancing the free on the region-exit path, rather
// than being hoisted to the function entry block.

module attributes {omp.is_gpu = true, omp.is_target_device = true} {
  llvm.func @parallel_in_loop(%arg0: !llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>} {
    %0 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(from) capture(ByRef) name("d") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%0 -> %arg2 : !llvm.ptr) {
      %c0 = llvm.mlir.constant(0 : i32) : i32
      %c2 = llvm.mlir.constant(2 : i32) : i32
      %c1 = llvm.mlir.constant(1 : i32) : i32
      llvm.br ^bb1(%c0 : i32)
    ^bb1(%iv: i32):
      %cond = llvm.icmp "slt" %iv, %c2 : i32
      llvm.cond_br %cond, ^bb2, ^bb3
    ^bb2:
      omp.parallel {
        llvm.store %iv, %arg2 : i32, !llvm.ptr
        omp.terminator
      }
      %next = llvm.add %iv, %c1 : i32
      llvm.br ^bb1(%next : i32)
    ^bb3:
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}parallel_in_loop
// CHECK: user_code.entry:
// The aggregate holding the parallel's arguments must NOT be hoisted into the
// function entry block (outside the loop).
// CHECK-NOT: call align {{.*}} ptr @__kmpc_alloc_shared(i64 16)
// It is allocated at the call site, inside the loop body, right before the fork.
// CHECK: omp_parallel:
// CHECK: %[[STRUCTARG:.*]] = call align {{.*}} ptr @__kmpc_alloc_shared(i64 16)
// CHECK: call void @__kmpc_parallel_60(
// The matching free is on the region-exit path, also inside the loop body: the
// exit branches back to the loop header (which performs the loop test), so the
// alloc/free pair sits on the loop backedge and repeats every iteration.
// CHECK: omp.par.exit:
// CHECK: call void @__kmpc_free_shared(ptr %[[STRUCTARG]], i64 16)
// CHECK: br label %[[LOOP_HEADER:.*]]
// CHECK: [[LOOP_HEADER]]:
// CHECK: icmp slt i32 {{.*}}, 2
