// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// in_reduction on an explicit omp.task. Unlike taskgroup task_reduction, the
// task does not register a reduction; it participates in a reduction declared
// by an enclosing taskgroup. The lowering must, inside the outlined task body:
//   1. Obtain the executing thread's gtid via __kmpc_global_thread_num;
//   2. Look up the per-task private storage via
//      __kmpc_task_reduction_get_th_data(gtid, null, orig) -- the NULL
//      descriptor makes the runtime walk up enclosing taskgroups to find the
//      matching task_reduction registration for `orig`;
//   3. Use the returned private pointer for all updates in the task body, never
//      the original shared variable.

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @task_in_reduction_single(%x : !llvm.ptr) {
  omp.task in_reduction(@add_i32 %x -> %prv : !llvm.ptr) {
    %v = llvm.load %prv : !llvm.ptr -> i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %s = llvm.add %v, %c1 : i32
    llvm.store %s, %prv : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// The encountering function must NOT register a reduction: no taskgroup, no
// descriptor array, and no __kmpc_taskred_init for in_reduction.
// CHECK-LABEL: define void @task_in_reduction_single(
// CHECK-NOT:     @__kmpc_taskred_init
// CHECK-NOT:     @__kmpc_taskgroup

// Outlined task body looks up per-task storage via the runtime with a NULL
// descriptor, and updates that private storage (not the original pointer).
// CHECK-LABEL: define internal void @task_in_reduction_single..omp_par(
// CHECK:         %[[BODY_GEP:.+]] = getelementptr {{.+}}, i32 0, i32 0
// CHECK:         %[[BODY_ORIG:.+]] = load ptr, ptr %[[BODY_GEP]]
// CHECK:         %[[BODY_GTID:.+]] = call i32 @__kmpc_global_thread_num(
// CHECK:         %[[PRIV:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[BODY_GTID]], ptr null, ptr %[[BODY_ORIG]])
// CHECK:         %[[LD:.+]] = load i32, ptr %[[PRIV]]
// CHECK:         %[[ADD:.+]] = add i32 %[[LD]], 1
// CHECK:         store i32 %[[ADD]], ptr %[[PRIV]]

// -----

// Multiple in_reduction items: the body issues one
// __kmpc_task_reduction_get_th_data per item, each with a NULL descriptor.

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @task_in_reduction_multi(%x : !llvm.ptr, %y : !llvm.ptr) {
  omp.task in_reduction(@add_i32 %x -> %px, @add_i32 %y -> %py : !llvm.ptr, !llvm.ptr) {
    %vx = llvm.load %px : !llvm.ptr -> i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %sx = llvm.add %vx, %c1 : i32
    llvm.store %sx, %px : i32, !llvm.ptr
    %vy = llvm.load %py : !llvm.ptr -> i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %sy = llvm.add %vy, %c2 : i32
    llvm.store %sy, %py : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// Each item is threaded through independently: the two original pointers come
// from distinct slots of the task shareds aggregate, each is passed to its own
// __kmpc_task_reduction_get_th_data lookup (NULL descriptor), and each item's
// body load/store targets only the matching private pointer -- never the
// original shared pointer.
// CHECK-LABEL: define internal void @task_in_reduction_multi..omp_par(
// CHECK:         %[[GEP0:.+]] = getelementptr {{.+}}, i32 0, i32 0
// CHECK:         %[[ORIG0:.+]] = load ptr, ptr %[[GEP0]]
// CHECK:         %[[GEP1:.+]] = getelementptr {{.+}}, i32 0, i32 1
// CHECK:         %[[ORIG1:.+]] = load ptr, ptr %[[GEP1]]
// CHECK:         %[[PRIV0:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %{{.+}}, ptr null, ptr %[[ORIG0]])
// CHECK:         %[[PRIV1:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %{{.+}}, ptr null, ptr %[[ORIG1]])
// CHECK:         %[[LDX:.+]] = load i32, ptr %[[PRIV0]]
// CHECK:         %[[ADDX:.+]] = add i32 %[[LDX]], 1
// CHECK:         store i32 %[[ADDX]], ptr %[[PRIV0]]
// CHECK:         %[[LDY:.+]] = load i32, ptr %[[PRIV1]]
// CHECK:         %[[ADDY:.+]] = add i32 %[[LDY]], 2
// CHECK:         store i32 %[[ADDY]], ptr %[[PRIV1]]
// CHECK-NOT:     store i32 %{{.+}}, ptr %[[ORIG0]]
// CHECK-NOT:     store i32 %{{.+}}, ptr %[[ORIG1]]

// -----

// Regression: a plain omp.task with no in_reduction must not emit any
// __kmpc_task_reduction_get_th_data call.

llvm.func @task_plain(%x : !llvm.ptr) {
  omp.task {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    llvm.store %c1, %x : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @task_plain(
// CHECK-NOT:     @__kmpc_task_reduction_get_th_data

// -----

// Nested case: an explicit omp.task carrying in_reduction inside an
// omp.taskgroup that declares the matching task_reduction. Registration happens
// once, on the enclosing taskgroup (__kmpc_taskred_init over a
// kmp_taskred_input_t descriptor); the explicit task does not register its own
// reduction. Inside the outlined task body the item is resolved with
// __kmpc_task_reduction_get_th_data and a NULL descriptor, which makes the
// runtime walk the enclosing taskgroup chain to find the registration.

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @task_in_reduction_nested(%x : !llvm.ptr) {
  omp.taskgroup task_reduction(@add_i32 %x -> %tgprv : !llvm.ptr) {
    omp.task in_reduction(@add_i32 %x -> %prv : !llvm.ptr) {
      %v = llvm.load %prv : !llvm.ptr -> i32
      %c1 = llvm.mlir.constant(1 : i32) : i32
      %s = llvm.add %v, %c1 : i32
      llvm.store %s, %prv : i32, !llvm.ptr
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// The enclosing taskgroup registers the reduction and then spawns the explicit
// task. The end_taskgroup block is emitted textually before the task-spawn
// block, so it is matched first here.
// CHECK-LABEL: define void @task_in_reduction_nested(
// CHECK:         %[[ARR:.+]] = alloca [1 x %kmp_taskred_input_t]
// CHECK:         call void @__kmpc_taskgroup(
// CHECK:         call ptr @__kmpc_taskred_init(i32 %{{.+}}, i32 1, ptr %[[ARR]])
// CHECK:         call void @__kmpc_end_taskgroup(
// CHECK:         call ptr @__kmpc_omp_task_alloc({{.+}}@task_in_reduction_nested..omp_par)
// CHECK:         call i32 @__kmpc_omp_task(

// The outlined task body resolves the in_reduction item with a NULL descriptor
// and updates only the private storage. It never registers its own reduction
// and never writes back to the original shared pointer.
// CHECK-LABEL: define internal void @task_in_reduction_nested..omp_par(
// CHECK:         %[[BODY_GEP:.+]] = getelementptr {{.+}}, i32 0, i32 0
// CHECK:         %[[BODY_ORIG:.+]] = load ptr, ptr %[[BODY_GEP]]
// CHECK:         %[[BODY_GTID:.+]] = call i32 @__kmpc_global_thread_num(
// CHECK:         %[[PRIV:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[BODY_GTID]], ptr null, ptr %[[BODY_ORIG]])
// CHECK:         %[[LD:.+]] = load i32, ptr %[[PRIV]]
// CHECK:         %[[ADD:.+]] = add i32 %[[LD]], 1
// CHECK:         store i32 %[[ADD]], ptr %[[PRIV]]
// CHECK-NOT:     store i32 %{{.+}}, ptr %[[BODY_ORIG]]
// CHECK-NOT:     call ptr @__kmpc_taskred_init

// -----

// Non-default address space: the in_reduction storage pointer lives in
// addrspace(1). __kmpc_task_reduction_get_th_data takes and returns a generic,
// default-addrspace ptr, so the original is addrspacecast to the generic space
// before the lookup, and the returned private pointer is cast back to
// addrspace(1) before the body uses it.

omp.declare_reduction @add_i32_as1 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @task_in_reduction_as1(%x : !llvm.ptr<1>) {
  omp.task in_reduction(@add_i32_as1 %x -> %prv : !llvm.ptr<1>) {
    %v = llvm.load %prv : !llvm.ptr<1> -> i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %s = llvm.add %v, %c1 : i32
    llvm.store %s, %prv : i32, !llvm.ptr<1>
    omp.terminator
  }
  llvm.return
}

// The encountering function still registers nothing for in_reduction.
// CHECK-LABEL: define void @task_in_reduction_as1(
// CHECK-NOT:     @__kmpc_taskred_init
// CHECK-NOT:     @__kmpc_taskgroup

// The outlined body normalizes the addrspace(1) original to a generic ptr for
// the runtime lookup, casts the returned private back to addrspace(1), and uses
// that private pointer for the body load/store.
// CHECK-LABEL: define internal void @task_in_reduction_as1..omp_par(
// CHECK:         %[[GEP:.+]] = getelementptr {{.+}}, i32 0, i32 0
// CHECK:         %[[ORIG:.+]] = load ptr addrspace(1), ptr %[[GEP]]
// CHECK:         %[[GTID:.+]] = call i32 @__kmpc_global_thread_num(
// CHECK:         %[[ORIG_CAST:.+]] = addrspacecast ptr addrspace(1) %[[ORIG]] to ptr
// CHECK:         %[[PRIV:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[GTID]], ptr null, ptr %[[ORIG_CAST]])
// CHECK:         %[[PRIV_CAST:.+]] = addrspacecast ptr %[[PRIV]] to ptr addrspace(1)
// CHECK:         %[[LD:.+]] = load i32, ptr addrspace(1) %[[PRIV_CAST]]
// CHECK:         %[[ADD:.+]] = add i32 %[[LD]], 1
// CHECK:         store i32 %[[ADD]], ptr addrspace(1) %[[PRIV_CAST]]
