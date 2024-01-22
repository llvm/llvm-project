; RUN: opt -S -passes=openmp-opt < %s
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%"struct.ompx::state::TeamStateTy" = type { %"struct.ompx::state::ICVStateTy", i32, i32, ptr }
%"struct.ompx::state::ICVStateTy" = type { i32, i32, i32, i32, i32, i32, i32 }

@_ZN4ompx5state9TeamStateE = internal addrspace(3) global %"struct.ompx::state::TeamStateTy" undef

define amdgpu_kernel void @__omp_offloading_32_70c2e76c_main_l24(ptr %dyn) {
  %1 = tail call i32 @__kmpc_target_init(ptr null, ptr %dyn)
  call void @__kmpc_parallel_51(ptr null, i32 0, i32 0, i32 0, i32 0, ptr @__omp_offloading_32_70c2e76c_main_l24_omp_outlined, ptr null, ptr null, i64 0)
  ret void
}

define void @__omp_offloading_32_70c2e76c_main_l24_omp_outlined(ptr %0) {
  call void @__kmpc_for_static_init_4()
  br label %2

2:                                                ; preds = %2, %1
  %3 = load ptr, ptr addrspace(1) null, align 4294967296
  %4 = call i32 %3(i32 0)
  store i32 %4, ptr %0, align 4
  br label %2
}

define internal i32 @__kmpc_target_init(ptr %0, ptr) {
  store i32 0, ptr addrspace(3) @_ZN4ompx5state9TeamStateE, align 16
  ret i32 0
}

declare void @__kmpc_parallel_51(ptr, i32, i32, i32, i32, ptr, ptr, ptr, i64)

define void @__kmpc_for_static_init_4() {
  %1 = load i32, ptr addrspace(3) @_ZN4ompx5state9TeamStateE, align 8
  ret void
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 7, !"openmp", i32 51}
!1 = !{i32 7, !"openmp-device", i32 51}
