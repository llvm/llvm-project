// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

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
