; RUN: opt -S -passes=openmp-opt < %s
target triple = "amdgpu7.00-amd-amdhsa"

%"struct.ompx::state::TeamStateTy" = type { %"struct.ompx::state::ICVStateTy", i32, i32, ptr }
%"struct.ompx::state::ICVStateTy" = type { i32, i32, i32, i32, i32, i32, i32 }

@_ZN4ompx5state9TeamStateE = internal addrspace(3) global %"struct.ompx::state::TeamStateTy" undef

define amdgpu_kernel void @__omp_offloading_32_70c2e76c_main_l24(ptr %dyn) {
  %1 = tail call i32 @__kmpc_target_init(ptr null, ptr %dyn)
  call void @__kmpc_parallel_60(ptr null, i32 0, i32 0, i32 0, i32 0, ptr @__omp_offloading_32_70c2e76c_main_l24_omp_outlined, ptr null, ptr null, i64 0, i32 0)
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

declare void @__kmpc_parallel_60(ptr, i32, i32, i32, i32, ptr, ptr, ptr, i64, i32)

define void @__kmpc_for_static_init_4() {
  %1 = load i32, ptr addrspace(3) @_ZN4ompx5state9TeamStateE, align 8
  ret void
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 7, !"openmp", i32 51}
!1 = !{i32 7, !"openmp-device", i32 51}
