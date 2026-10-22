// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// in_reduction on omp.target compiled for the target *device*. From the
// device's perspective an in_reduction list item behaves as a regular mapped
// variable: the host-only redirect to the per-task reduction-private storage
// (the __kmpc_global_thread_num + __kmpc_task_reduction_get_th_data lookup) is
// guarded by !isTargetDevice and is therefore not emitted. Verify the device
// body accesses the mapped variable directly and that none of the host
// runtime lookup is present.

module attributes {dlti.dl_spec = #dlti.dl_spec<"dlti.alloca_memory_space" = 5 : ui64, "dlti.global_memory_space" = 1 : ui64>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
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

  llvm.func @target_inreduction_device(%x : !llvm.ptr) {
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
}

// The device kernel receives the mapped pointer as a captured argument and
// accesses the in_reduction variable directly through it.
// CHECK-LABEL: define {{.*}}amdgpu_kernel void @__omp_offloading_{{.*}}_target_inreduction_device_
// CHECK-SAME:    (ptr %[[ARG:.+]], ptr %{{.+}})
// CHECK:         store ptr %[[ARG]], ptr %[[ALLOCA:.+]]
// CHECK:         %[[MAPPED:.+]] = load ptr, ptr %[[ALLOCA]]
// The host-only in_reduction redirect must not run on the device: there is no
// gtid lookup, no __kmpc_task_reduction_get_th_data call and no private copy.
// CHECK-NOT:     @__kmpc_global_thread_num
// CHECK-NOT:     @__kmpc_task_reduction_get_th_data
// CHECK-NOT:     omp.inred.priv
// The body reads and writes the mapped variable in place.
// CHECK:         %[[LOADED:.+]] = load i32, ptr %[[MAPPED]]
// CHECK:         %[[SUM:.+]] = add i32 %[[LOADED]], 1
// CHECK:         store i32 %[[SUM]], ptr %[[MAPPED]]
