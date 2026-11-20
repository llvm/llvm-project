// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Accesses through `noalias` pointer arguments (Fortran dummy arguments) must
// not be moved across points where OpenMP implies a flush. The `noalias`
// arguments are passed to llvm.omp.sync.marker on both sides of such points.

// CHECK-LABEL: define void @barrier(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]], ptr %{{[0-9]+}})
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK-NEXT: call i32 @__kmpc_global_thread_num
// CHECK-NEXT: call void @__kmpc_barrier
// CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
llvm.func @barrier(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr) {
  omp.barrier
  llvm.return
}

// The markers inside the region keep the store between the runtime calls.
// CHECK-LABEL: define void @critical(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]])
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: call void @__kmpc_critical
// CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: store i32 1, ptr %[[A]]
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: call void @__kmpc_end_critical
// CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
llvm.func @critical(%arg0: !llvm.ptr {llvm.noalias}) {
  omp.critical {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    llvm.store %c1, %arg0 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// The flush of single comes from its end barrier.
// CHECK-LABEL: define void @single(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]])
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: call i32 @__kmpc_single
// CHECK: call void @__kmpc_barrier
// CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
llvm.func @single(%arg0: !llvm.ptr {llvm.noalias}) {
  omp.single {
    omp.terminator
  }
  llvm.return
}

// With nowait there is no end barrier, so nothing is marked.
// CHECK-LABEL: define void @single_nowait(
// CHECK-NOT: @llvm.omp.sync.marker
// CHECK: ret void
llvm.func @single_nowait(%arg0: !llvm.ptr {llvm.noalias}) {
  omp.single nowait {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @wsloop(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]], i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: call void @__kmpc_for_static_init
// CHECK: call void @__kmpc_barrier
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
llvm.func @wsloop(%arg0: !llvm.ptr {llvm.noalias}, %lb: i32, %ub: i32, %step: i32) {
  omp.wsloop {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// CHECK-LABEL: define void @scope(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]])
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: call void @__kmpc_barrier
// CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
llvm.func @scope(%arg0: !llvm.ptr {llvm.noalias}) {
  omp.scope {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @scope_nowait(
// CHECK-NOT: @llvm.omp.sync.marker
// CHECK: ret void
llvm.func @scope_nowait(%arg0: !llvm.ptr {llvm.noalias}) {
  omp.scope nowait {
    omp.terminator
  }
  llvm.return
}

// Encountering a task is a task scheduling point. The task body is outlined
// and doesn't get the argument.
// CHECK-LABEL: define void @task(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]])
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: call i32 @__kmpc_omp_task(
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: define internal void @task..omp_par(i32 %{{.*}})
// CHECK-NOT: @llvm.omp.sync.marker
// CHECK: ret void
llvm.func @task(%arg0: !llvm.ptr {llvm.noalias}) {
  omp.task {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @taskwait(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]])
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: call i32 @__kmpc_omp_taskwait
// CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
llvm.func @taskwait(%arg0: !llvm.ptr {llvm.noalias}) {
  omp.taskwait
  llvm.return
}

// A flush without a list marks all `noalias` pointer arguments.
// CHECK-LABEL: define void @flush(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]], ptr noalias %[[B:[0-9]+]], ptr %{{[0-9]+}})
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr %[[B]])
// CHECK-NEXT: call void @__kmpc_flush
// CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr %[[B]])
llvm.func @flush(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}, %arg2: !llvm.ptr) {
  omp.flush
  llvm.return
}

// A flush with a list marks only the listed `noalias` arguments, once each.
// CHECK-LABEL: define void @flush_list(
// CHECK-SAME: ptr noalias %{{[0-9]+}}, ptr noalias %[[B:[0-9]+]], ptr %{{[0-9]+}})
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[B]])
// CHECK-NEXT: call void @__kmpc_flush
// CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr %[[B]])
// CHECK-NEXT: ret void
llvm.func @flush_list(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}, %arg2: !llvm.ptr) {
  omp.flush (%arg1, %arg2, %arg1 : !llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.return
}

// `noalias readonly` arguments (Fortran INTENT(IN) dummies) aren't marked, not
// even when they are listed in a flush: alias analysis assumes that their
// memory doesn't change during the call.
// CHECK-LABEL: define void @readonly_arg(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]], ptr noalias readonly %{{[0-9]+}})
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK-NEXT: call i32 @__kmpc_global_thread_num
// CHECK-NEXT: call void @__kmpc_barrier
// CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK-NEXT: call void @__kmpc_flush
// CHECK-NEXT: ret void
llvm.func @readonly_arg(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias, llvm.readonly}) {
  omp.barrier
  omp.flush (%arg1 : !llvm.ptr)
  llvm.return
}

// Atomic operations that are not relaxed imply a flush.
// CHECK-LABEL: define void @atomic_acquire(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]], ptr %{{[0-9]+}}, ptr %{{[0-9]+}})
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: load atomic i32, ptr %{{[0-9]+}} acquire
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK-NEXT: ret void
llvm.func @atomic_acquire(%arg0: !llvm.ptr {llvm.noalias}, %x: !llvm.ptr, %v: !llvm.ptr) {
  omp.atomic.read %v = %x memory_order(acquire) : !llvm.ptr, !llvm.ptr, i32
  llvm.return
}

// CHECK-LABEL: define void @atomic_release(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]], ptr %{{[0-9]+}}, i32 %{{[0-9]+}})
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: store atomic i32 %{{[0-9]+}}, ptr %{{[0-9]+}} release
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK-NEXT: ret void
llvm.func @atomic_release(%arg0: !llvm.ptr {llvm.noalias}, %x: !llvm.ptr, %e: i32) {
  omp.atomic.write %x = %e memory_order(release) : !llvm.ptr, i32
  llvm.return
}

// Relaxed atomic operations and masked don't imply a flush.
// CHECK-LABEL: define void @no_flush(
// CHECK-NOT: @llvm.omp.sync.marker
// CHECK: ret void
llvm.func @no_flush(%arg0: !llvm.ptr {llvm.noalias}, %x: !llvm.ptr, %v: !llvm.ptr) {
  omp.atomic.read %v = %x memory_order(relaxed) : !llvm.ptr, !llvm.ptr, i32
  omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, i32
  omp.masked {
    omp.terminator
  }
  llvm.return
}

// Generating a target task is a task scheduling point.
// CHECK-LABEL: define void @target(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]])
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: call void @__omp_offloading_{{.*}}_target_l{{[0-9]+}}(
// CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
llvm.func @target(%arg0: !llvm.ptr {llvm.noalias}) {
  omp.target kernel_type(generic) {
    omp.terminator
  }
  llvm.return
}

// The parallel construct is marked in the enclosing function. Constructs in
// the outlined region aren't, since its arguments aren't dummy arguments.
// CHECK-LABEL: define void @in_parallel(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]])
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr @{{[0-9]+}}, i32 0, ptr @in_parallel..omp_par)
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: define internal void @in_parallel..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr)
// CHECK-NOT: @llvm.omp.sync.marker
// CHECK: call void @__kmpc_barrier
// CHECK-NOT: @llvm.omp.sync.marker
// CHECK: call void @__kmpc_flush
// CHECK-NOT: @llvm.omp.sync.marker
// CHECK: ret void
llvm.func @in_parallel(%arg0: !llvm.ptr {llvm.noalias}) {
  omp.parallel {
    omp.barrier
    omp.flush
    omp.terminator
  }
  llvm.return
}

// -----

// The start and the end of a target data region are task scheduling points.
// CHECK-LABEL: define void @target_data(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]], ptr %{{[0-9]+}})
// CHECK: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
// CHECK: call void @__tgt_target_data_begin_mapper(
// CHECK: call void @__tgt_target_data_end_mapper(
// CHECK-NEXT: call void @llvm.omp.sync.marker.p0(ptr %[[A]])
module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @target_data(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr) {
    %0 = omp.map.info var_ptr(%arg1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    omp.target_data map_entries(%0 : !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}
