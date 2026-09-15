// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// in_reduction on omp.target: the in_reduction variable is also captured
// into the target region as a map entry (a frontend is expected to emit this
// implicit map). The in_reduction clause does not define an entry block
// argument; inside the target body the variable is accessed through its
// map_entries block argument. The captured pointer is passed to
// __kmpc_task_reduction_get_th_data with a NULL descriptor; the runtime
// walks enclosing taskgroups to locate the matching task_reduction
// registration. The returned per-task private pointer is bound to the
// map_entries block argument so subsequent loads/stores inside the region
// use the private copy.

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

llvm.func @target_inreduction(%x : !llvm.ptr) {
  %m = omp.map.info var_ptr(%x : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
  omp.target kernel_type(generic) in_reduction(@add_i32 %x : !llvm.ptr) map_entries(%m -> %marg : !llvm.ptr) {
    %v = llvm.load %marg : !llvm.ptr -> i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %s = llvm.add %v, %c1 : i32
    llvm.store %s, %marg : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// The host stub forwards the captured pointer into the outlined target
// kernel.
// CHECK-LABEL: define void @target_inreduction(
// CHECK:         call void @__omp_offloading_{{.*}}_target_inreduction_{{.*}}(ptr %{{.+}}, ptr null)

// In the outlined target body the in_reduction private pointer is
// obtained from the runtime using the captured original pointer; that
// pointer is then the base of the load and store inside the region.
// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_target_inreduction_
// CHECK-SAME:    (ptr %[[CAPT:.+]], ptr %{{.+}})
// CHECK:         %[[GTID:.+]] = call i32 @__kmpc_global_thread_num(
// CHECK:         %[[PRIV:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[GTID]], ptr null, ptr %[[CAPT]])
// CHECK:         %[[LOADED:.+]] = load i32, ptr %[[PRIV]]
// CHECK:         %[[SUM:.+]] = add i32 %[[LOADED]], 1
// CHECK:         store i32 %[[SUM]], ptr %[[PRIV]]

// -----

// Same as the first case but the in_reduction variable lives in a non-default
// address space (addrspace(1)). __kmpc_task_reduction_get_th_data is declared
// to take and return a generic (default-AS) `ptr`, so the host
// lowering must (1) addrspacecast the captured addrspace(1) original pointer to
// generic `ptr` before the runtime lookup, and (2) addrspacecast the returned
// generic private pointer back to addrspace(1) so the body's load/store use the
// private copy in the right address space. This pins down the address-space
// handling so a regression that passed the addrspace(1) pointer straight into
// the runtime call (a bad-signature crash) cannot pass.

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

llvm.func @target_inreduction_as1(%x : !llvm.ptr<1>) {
  %m = omp.map.info var_ptr(%x : !llvm.ptr<1>, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr<1>
  omp.target kernel_type(generic) in_reduction(@add_i32_as1 %x : !llvm.ptr<1>) map_entries(%m -> %marg : !llvm.ptr<1>) {
    %v = llvm.load %marg : !llvm.ptr<1> -> i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %s = llvm.add %v, %c1 : i32
    llvm.store %s, %marg : i32, !llvm.ptr<1>
    omp.terminator
  }
  llvm.return
}

// In the outlined target body the addrspace(1) original pointer is normalized
// to generic `ptr` before the NULL-descriptor in_reduction lookup, and the
// returned generic private pointer is cast back to addrspace(1) for the body's
// load and store. The original captured pointer arrives as the addrspace(1)
// kernel argument and is only used to derive the generic lookup pointer.
// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_target_inreduction_as1_
// CHECK-SAME:    (ptr addrspace(1) %[[CAPT_AS1:.+]], ptr %{{.+}})
// CHECK:         %[[GTID_AS1:.+]] = call i32 @__kmpc_global_thread_num(
// CHECK:         %[[ORIG_GEN:.+]] = addrspacecast ptr addrspace(1) %[[CAPT_AS1]] to ptr
// CHECK:         %[[PRIV_AS1_GEN:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[GTID_AS1]], ptr null, ptr %[[ORIG_GEN]])
// CHECK:         %[[PRIV_AS1:.+]] = addrspacecast ptr %[[PRIV_AS1_GEN]] to ptr addrspace(1)
// CHECK:         %[[LOADED_AS1:.+]] = load i32, ptr addrspace(1) %[[PRIV_AS1]]
// CHECK:         %[[SUM_AS1:.+]] = add i32 %[[LOADED_AS1]], 1
// CHECK:         store i32 %[[SUM_AS1]], ptr addrspace(1) %[[PRIV_AS1]]

// The body must not load or store through the original captured addrspace(1)
// pointer; all accesses go through the runtime-returned private copy.
// CHECK-NOT:     store i32 %{{.+}}, ptr addrspace(1) %[[CAPT_AS1]]
