// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true, omp.is_gpu = true} {
  llvm.func @main(%nt : !llvm.ptr) {
    %map_nt = omp.map.info var_ptr(%nt : !llvm.ptr, i64) map_clauses(to) capture(ByCopy) -> !llvm.ptr {name = "nthreads"}
    omp.target map_entries(%map_nt -> %arg_nt : !llvm.ptr) {
      %num_threads = llvm.load %arg_nt : !llvm.ptr -> i64
      omp.parallel num_threads(%num_threads : i64) {
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_main_l{{[0-9]+}}
// CHECK: [[NT32:%.*]] = trunc i64 {{.*}} to i32
// CHECK: call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 [[NT32]], i32 {{.*}}, ptr @{{.*}}, ptr {{.*}}, ptr {{.*}}, i64 {{.*}}, i32 {{.*}})
