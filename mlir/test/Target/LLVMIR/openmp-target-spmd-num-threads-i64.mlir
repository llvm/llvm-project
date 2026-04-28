// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true, omp.is_gpu = true} {
  llvm.func @main(%x : i32) {
    omp.target host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
      %numThreads = llvm.mlir.constant(137 : i64) : i64
      omp.teams {
        omp.parallel num_threads(%numThreads : i64) {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: define internal void @[[TARGET_OUTLINE:.*]]({{.*}})
// CHECK: call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 137, i32 {{.*}}, ptr @[[PARALLEL_OUTLINE:.*]], ptr {{.*}}, ptr {{.*}}, i64 {{.*}}, i32 {{.*}})
