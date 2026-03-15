// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  // CHECK-LABEL: define void @device_shared_mem(
  // CHECK-SAME:  i32 %[[N0:.*]], i64 %[[N1:.*]])
  llvm.func @device_shared_mem(%n0: i32, %n1: i64) attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>} {
    // CHECK:      %[[CAST_N0:.*]] = zext i32 %[[N0]] to i64
    // CHECK-NEXT: %[[ALLOC0_SZ:.*]] = mul i64 8, %[[CAST_N0]]
    // CHECK-NEXT: %[[ALLOC0:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 %[[ALLOC0_SZ]])
    %0 = omp.alloc_shared_mem %n0 x i64 : (i32) -> !llvm.ptr

    // CHECK:      %[[ALLOC1_SZ:.*]] = mul i64 8, %[[N1]]
    // CHECK-NEXT: %[[ALLOC1:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 %[[ALLOC1_SZ]])
    %1 = omp.alloc_shared_mem %n1 x i64 : (i64) -> !llvm.ptr

    // CHECK:      %[[ALLOC2_SZ:.*]] = mul i64 64, %[[N1]]
    // CHECK-NEXT: %[[ALLOC2:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 %[[ALLOC2_SZ]])
    %2 = omp.alloc_shared_mem %n1 x vector<16xf32> : (i64) -> !llvm.ptr

    // CHECK:      %[[ALLOC3_SZ:.*]] = mul i64 128, %[[N1]]
    // CHECK-NEXT: %[[ALLOC3:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 %[[ALLOC3_SZ]])
    %3 = omp.alloc_shared_mem %n1 x vector<16xf32> {alignment = 128} : (i64) -> !llvm.ptr

    // CHECK:      %[[CAST_N0_1:.*]] = zext i32 %[[N0]] to i64
    // CHECK-NEXT: %[[FREE0_SZ:.*]] = mul i64 8, %[[CAST_N0_1]]
    // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[ALLOC0]], i64 %[[FREE0_SZ]])
    omp.free_shared_mem %0 : !llvm.ptr

    // CHECK:      %[[FREE1_SZ:.*]] = mul i64 8, %[[N1]]
    // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[ALLOC1]], i64 %[[FREE1_SZ]])
    omp.free_shared_mem %1 : !llvm.ptr

    // CHECK:      %[[FREE2_SZ:.*]] = mul i64 64, %[[N1]]
    // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[ALLOC2]], i64 %[[FREE2_SZ]])
    omp.free_shared_mem %2 : !llvm.ptr

    // CHECK:      %[[FREE3_SZ:.*]] = mul i64 128, %[[N1]]
    // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[ALLOC3]], i64 %[[FREE3_SZ]])
    omp.free_shared_mem %3 : !llvm.ptr
    llvm.return
  }
}
