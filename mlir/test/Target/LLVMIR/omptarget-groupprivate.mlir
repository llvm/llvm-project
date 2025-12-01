// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = true, llvm.target_triple = "amdgcn-amd-amdhsa",
                    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>} {
  llvm.func @_QQmain() attributes {fir.bindc_name = "main"} {

    %ga = llvm.mlir.addressof @global_a : !llvm.ptr
    %map_a = omp.map.info var_ptr(%ga : !llvm.ptr, i32) map_clauses(tofrom) capture(ByCopy) -> !llvm.ptr {name = "i"}
    omp.target map_entries(%map_a -> %arg1 : !llvm.ptr) {
      %loaded = llvm.load %arg1 : !llvm.ptr -> i32

      %any_addr = llvm.mlir.addressof @global_any : !llvm.ptr
      %any_gp = omp.groupprivate %any_addr : !llvm.ptr, device_type(any) -> !llvm.ptr
      llvm.store %loaded, %any_gp : i32, !llvm.ptr

      %host_addr = llvm.mlir.addressof @global_host : !llvm.ptr
      %host_gp = omp.groupprivate %host_addr : !llvm.ptr, device_type(host) -> !llvm.ptr
      llvm.store %loaded, %host_gp : i32, !llvm.ptr

      %nohost_addr = llvm.mlir.addressof @global_nohost : !llvm.ptr
      %nohost_gp = omp.groupprivate %nohost_addr : !llvm.ptr, device_type(nohost) -> !llvm.ptr
      llvm.store %loaded, %nohost_gp : i32, !llvm.ptr

      omp.terminator
    }
    llvm.return
  }
  llvm.mlir.global internal @global_a() : i32
  llvm.mlir.global internal @global_any() : i32
  llvm.mlir.global internal @global_host() : i32
  llvm.mlir.global internal @global_nohost() : i32
}

// CHECK-DAG: @global_a = internal global i32 undef
// CHECK-DAG: @global_any = internal global i32 undef
// CHECK-DAG: @global_host = internal global i32 undef
// CHECK-DAG: @global_nohost = internal global i32 undef
// CHECK-DAG: {{.*}} = internal addrspace(3) global i32 poison
// CHECK-DAG: {{.*}} = internal addrspace(3) global i32 poison
// CHECK: define {{.*}} amdgpu_kernel void @__omp_offloading_{{.*}}_{{.*}}__QQmain_{{.*}}(ptr %{{.*}}, ptr %{{.*}}) #{{[0-9]+}} {
// CHECK-LABEL:  omp.target:
// CHECK-NEXT :    %[[LOAD:.*]] = load i32, ptr %{{.*}}, align 4
// CHECK-NEXT :    store i32 %[[LOAD]], ptr addrspace(3) {{.*}}, align 4
// CHECK-NEXT :    store i32 %[[LOAD]], ptr @global_host, align 4
// CHECK-NEXT :    store i32 %[[LOAD]], ptr addrspace(3) {{.*}}, align 4
