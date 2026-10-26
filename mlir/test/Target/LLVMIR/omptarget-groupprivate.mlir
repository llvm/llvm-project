// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = true, llvm.target_triple = "amdgcn-amd-amdhsa",
                    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>} {
  llvm.func @_QQmain() attributes {fir.bindc_name = "main"} {

    %ga = llvm.mlir.addressof @global_a : !llvm.ptr
    %map_a = omp.map.info var_ptr(%ga : !llvm.ptr, i32) map_clauses(tofrom) capture(ByCopy) name("i") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%map_a -> %arg1 : !llvm.ptr) {
      %loaded = llvm.load %arg1 : !llvm.ptr -> i32

      %any_gp = omp.groupprivate @global_any device_type(any) : !llvm.ptr
      llvm.store %loaded, %any_gp : i32, !llvm.ptr

      %host_gp = omp.groupprivate @global_host device_type(host) : !llvm.ptr
      llvm.store %loaded, %host_gp : i32, !llvm.ptr

      %nohost_gp = omp.groupprivate @global_nohost device_type(nohost) : !llvm.ptr
      llvm.store %loaded, %nohost_gp : i32, !llvm.ptr

      omp.terminator
    }
    llvm.return
  }

  // A groupprivate directive is equivalent whether it appears directly inside a
  // 'target' region (as in @_QQmain above) or nested in a 'teams' region: both
  // allocate a per-contention-group copy in the shared address space.
  llvm.func @teams_equiv() {
    %gt = llvm.mlir.addressof @global_teams : !llvm.ptr
    %map_t = omp.map.info var_ptr(%gt : !llvm.ptr, i32) map_clauses(tofrom) capture(ByCopy) name("t") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%map_t -> %arg1 : !llvm.ptr) {
      %loaded = llvm.load %arg1 : !llvm.ptr -> i32
      omp.teams {
        %teams_gp = omp.groupprivate @global_teams device_type(any) : !llvm.ptr
        llvm.store %loaded, %teams_gp : i32, !llvm.ptr
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }

  llvm.mlir.global internal @global_a() : i32
  llvm.mlir.global internal @global_any() : i32
  llvm.mlir.global internal @global_host() : i32
  llvm.mlir.global internal @global_nohost() : i32
  llvm.mlir.global internal @global_teams() : i32
}

// CHECK-DAG: @global_a = internal global i32 undef
// CHECK-DAG: @global_any = internal global i32 undef
// CHECK-DAG: @global_host = internal global i32 undef
// CHECK-DAG: @global_nohost = internal global i32 undef
// CHECK-DAG: @[[SHARED_ANY:global_any.*]] = internal addrspace(3) global i32 poison
// CHECK-DAG: @[[SHARED_NOHOST:global_nohost.*]] = internal addrspace(3) global i32 poison
// CHECK-DAG: @[[SHARED_TEAMS:global_teams.*]] = internal addrspace(3) global i32 poison
// CHECK: define {{.*}} amdgpu_kernel void @__omp_offloading_{{.*}}_{{.*}}__QQmain_{{.*}}(ptr %{{.*}}, ptr %{{.*}}) #{{[0-9]+}} {
// CHECK:        %[[LOAD:.*]] = load i32, ptr %{{.*}}, align 4
// CHECK-NEXT :  store i32 %[[LOAD]], ptr addrspace(3) @[[SHARED_ANY]], align 4
// CHECK-NEXT :  store i32 %[[LOAD]], ptr @global_host, align 4
// CHECK-NEXT :  store i32 %[[LOAD]], ptr addrspace(3) @[[SHARED_NOHOST]], align 4

// CHECK: store i32 %{{.*}}, ptr addrspace(3) @[[SHARED_TEAMS]], align 4
