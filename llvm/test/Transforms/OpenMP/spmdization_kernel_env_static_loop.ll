; RUN: opt -S -passes=openmp-opt < %s | FileCheck %s

; The __kmpc_*_static_loop_* entries take the loop body as a callback. Treating
; it as opaque made every kernel whose parallel region contains a device
; workshare loop look like it nested parallelism. Resolve the callback: here it
; is a plain loop body, so MayUseNestedParallelism must be refined to 0.
; Field order is UseGenericStateMachine, MayUseNestedParallelism, ExecMode.

%struct.KernelEnvironmentTy = type { %struct.ConfigurationEnvironmentTy, ptr, ptr }
%struct.ConfigurationEnvironmentTy = type { i8, i8, i8, i32, i32, i32, i32 }

@kernel_environment = local_unnamed_addr constant %struct.KernelEnvironmentTy {
  %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 2, i32 1, i32 256, i32 1, i32 1 },
  ptr null, ptr null }

; CHECK: @kernel_environment = local_unnamed_addr constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 0, i8 0, i8 2,

define weak amdgpu_kernel void @kernel(ptr %args) #0 {
entry:
  %0 = call i32 @__kmpc_target_init(ptr @kernel_environment, ptr null)
  call void @__kmpc_parallel_60(ptr null, i32 0, i32 1, i32 -1, i32 -1,
                                ptr @omp_par, ptr null, ptr %args, i64 1, i32 0)
  call void @__kmpc_target_deinit()
  ret void
}

; The parallel region: its whole body is a device workshare loop.
define internal void @omp_par(ptr %tid, ptr %zero, ptr %args) {
entry:
  call void @__kmpc_distribute_for_static_loop_4u(ptr null, ptr @loop_body,
                                                  ptr %args, i32 64, i32 0,
                                                  i32 0, i32 0, i8 0)
  ret void
}

; The callback: plain work, no parallel region of its own.
define internal void @loop_body(i32 %iv, ptr %args) {
entry:
  %p = load ptr, ptr %args, align 8
  %idx = zext i32 %iv to i64
  %gep = getelementptr inbounds i32, ptr %p, i64 %idx
  store i32 %iv, ptr %gep, align 4
  ret void
}

; Negative case: the callback does contain a parallel region, so the refinement
; must not fire and MayUseNestedParallelism stays 1.

@kernel_environment_nested = local_unnamed_addr constant %struct.KernelEnvironmentTy {
  %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 2, i32 1, i32 256, i32 1, i32 1 },
  ptr null, ptr null }

; CHECK: @kernel_environment_nested = local_unnamed_addr constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 2,

define weak amdgpu_kernel void @kernel_nested(ptr %args) #0 {
entry:
  %0 = call i32 @__kmpc_target_init(ptr @kernel_environment_nested, ptr null)
  call void @__kmpc_parallel_60(ptr null, i32 0, i32 1, i32 -1, i32 -1,
                                ptr @omp_par_nested, ptr null, ptr %args,
                                i64 1, i32 0)
  call void @__kmpc_target_deinit()
  ret void
}

define internal void @omp_par_nested(ptr %tid, ptr %zero, ptr %args) {
entry:
  call void @__kmpc_distribute_for_static_loop_4u(ptr null, ptr @loop_body_nested,
                                                  ptr %args, i32 64, i32 0,
                                                  i32 0, i32 0, i8 0)
  ret void
}

define internal void @loop_body_nested(i32 %iv, ptr %args) {
entry:
  call void @__kmpc_parallel_60(ptr null, i32 0, i32 1, i32 -1, i32 -1,
                                ptr @inner_par, ptr null, ptr %args, i64 1, i32 0)
  ret void
}

define internal void @inner_par(ptr %tid, ptr %zero, ptr %args) {
entry:
  ret void
}

declare i32 @__kmpc_target_init(ptr, ptr)
declare void @__kmpc_target_deinit()
declare void @__kmpc_parallel_60(ptr, i32, i32, i32, i32, ptr, ptr, ptr, i64, i32)
declare void @__kmpc_distribute_for_static_loop_4u(ptr, ptr, ptr, i32, i32, i32, i32, i8)

attributes #0 = { "kernel" "omp_target_thread_limit"="256" }

!llvm.module.flags = !{!0, !1}
!0 = !{i32 7, !"openmp", i32 51}
!1 = !{i32 7, !"openmp-device", i32 51}
