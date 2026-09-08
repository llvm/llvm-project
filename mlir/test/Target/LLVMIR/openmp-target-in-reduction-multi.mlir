// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Multiple in_reduction items on omp.target. Each item is captured into the
// target region through its own map_entries entry and accessed inside the
// body via the corresponding map_entries block argument. On the host, every
// item performs an independent
// __kmpc_task_reduction_get_th_data lookup using its own captured original
// pointer, and the returned per-task private pointer is bound to that item's
// map block argument. This test pins down the pairing so it cannot pass if the
// two items were swapped or collapsed onto a single pointer.

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

llvm.func @target_inreduction_multi(%x : !llvm.ptr, %y : !llvm.ptr) {
  %mx = omp.map.info var_ptr(%x : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
  %my = omp.map.info var_ptr(%y : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
  omp.target kernel_type(generic) in_reduction(@add_i32 %x, @add_i32 %y : !llvm.ptr, !llvm.ptr)
      map_entries(%mx -> %mxarg, %my -> %myarg : !llvm.ptr, !llvm.ptr) {
    // First item (x): load, += 1, store back.
    %vx = llvm.load %mxarg : !llvm.ptr -> i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %sx = llvm.add %vx, %c1 : i32
    llvm.store %sx, %mxarg : i32, !llvm.ptr
    // Second item (y): load, += 2, store back.
    %vy = llvm.load %myarg : !llvm.ptr -> i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %sy = llvm.add %vy, %c2 : i32
    llvm.store %sy, %myarg : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// The host stub forwards both captured pointers into the outlined target
// kernel (the trailing argument is the unused descriptor slot).
// CHECK-LABEL: define void @target_inreduction_multi(
// CHECK:         call void @__omp_offloading_{{.*}}_target_inreduction_multi_{{.*}}(ptr %{{.+}}, ptr %{{.+}}, ptr null)

// The two captured original pointers arrive as distinct kernel arguments.
// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_target_inreduction_multi_
// CHECK-SAME:    (ptr %[[CAPTX:.+]], ptr %[[CAPTY:.+]], ptr %{{.+}})

// A single gtid is computed once for the whole target body and shared by both
// lookups; each item then performs its own __kmpc_task_reduction_get_th_data
// call against its own captured pointer, and the returned per-task private
// pointer is bound to that item's map block argument.
// CHECK:         %[[GTID:.+]] = call i32 @__kmpc_global_thread_num(
// CHECK:         %[[PRIVX:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[GTID]], ptr null, ptr %[[CAPTX]])
// CHECK:         %[[PRIVY:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[GTID]], ptr null, ptr %[[CAPTY]])

// The first item's private storage is the base of the +1 load/store; the
// CHECK-NOT below ensures the second item's pointer is not touched until the
// first item's accumulation has completed (i.e. the items are not swapped or
// merged onto a single private pointer).
// CHECK:         %[[LX:.+]] = load i32, ptr %[[PRIVX]]
// CHECK:         %[[SX:.+]] = add i32 %[[LX]], 1
// CHECK-NOT:     %[[PRIVY]]
// CHECK:         store i32 %[[SX]], ptr %[[PRIVX]]

// The second item's private storage is the base of the +2 load/store.
// CHECK:         %[[LY:.+]] = load i32, ptr %[[PRIVY]]
// CHECK:         %[[SY:.+]] = add i32 %[[LY]], 2
// CHECK:         store i32 %[[SY]], ptr %[[PRIVY]]

// Exactly two reduction lookups are emitted; no third call sneaks in. The
// `call` form is used so this does not match the runtime declaration.
// CHECK-NOT:     call ptr @__kmpc_task_reduction_get_th_data
