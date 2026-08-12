; RUN: opt --mtriple=amdgcn-amd-amdhsa -S -passes=openmp-opt < %s | FileCheck %s

; When OpenMPOpt transforms a generic-mode kernel to SPMD mode, its generic-mode
; worker state machine is gone, so the parallel data-sharing wrapper (argument
; index 6 of __kmpc_parallel_60) is dead. OpenMPOpt nulls that argument so the
; otherwise-dead wrapper -- and any LDS it references -- can be eliminated instead
; of lingering as a non-kernel function that uses LDS (which the AMDGPU backend
; warns about). A kernel that stays in generic mode still needs its wrapper for
; the worker state machine, so it must be left untouched.

%struct.ident_t = type { i32, i32, i32, i32, ptr }
%struct.KernelEnvironmentTy = type { %struct.ConfigurationEnvironmentTy, ptr, ptr }
%struct.ConfigurationEnvironmentTy = type { i8, i8, i8, i32, i32, i32, i32, i32, i32 }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr @0 }, align 8

; The SPMD-amenable kernel is promoted: ExecMode 1 (GENERIC) -> 3 (GENERIC_SPMD).
; CHECK: @spmd_kernel_environment = {{.*}}ConfigurationEnvironmentTy { i8 0, i8 0, i8 3
@spmd_kernel_environment = local_unnamed_addr constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 0, i8 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 }, ptr @1, ptr null }
; The kernel with an unguardable side effect stays generic: ExecMode remains 1.
; CHECK: @generic_kernel_environment = {{.*}}ConfigurationEnvironmentTy { i8 1, i8 0, i8 1
@generic_kernel_environment = local_unnamed_addr constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 0, i8 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 }, ptr @1, ptr null }

define weak amdgpu_kernel void @spmd_kernel() "kernel" {
  %init = call i32 @__kmpc_target_init(ptr @spmd_kernel_environment, ptr null)
  call void @spmd_outlined(ptr null, ptr null)
  call void @__kmpc_target_deinit()
  ret void
}

define weak amdgpu_kernel void @generic_kernel() "kernel" {
  %init = call i32 @__kmpc_target_init(ptr @generic_kernel_environment, ptr null)
  call void @generic_outlined(ptr null, ptr null)
  call void @__kmpc_target_deinit()
  ret void
}

; SPMD kernel: the wrapper (arg 6) is nulled.
; CHECK-LABEL: define internal void @spmd_outlined(
; CHECK: call void @__kmpc_parallel_60(ptr @1, i32 0, i32 1, i32 -1, i32 -1, ptr @spmd_body, ptr null, ptr null, i64 0, i32 0)
define internal void @spmd_outlined(ptr %.global_tid., ptr %.bound_tid.) {
  call void @__kmpc_parallel_60(ptr @1, i32 0, i32 1, i32 -1, i32 -1, ptr @spmd_body, ptr @spmd_body_wrapper, ptr null, i64 0, i32 0)
  ret void
}

; Generic kernel: the wrapper (arg 6) is preserved.
; CHECK-LABEL: define internal void @generic_outlined(
; CHECK: call void @__kmpc_parallel_60(ptr @1, i32 0, i32 1, i32 -1, i32 -1, ptr @generic_body, ptr @generic_body_wrapper, ptr null, i64 0, i32 0)
define internal void @generic_outlined(ptr %.global_tid., ptr %.bound_tid.) {
  call void @unknown()
  call void @__kmpc_parallel_60(ptr @1, i32 0, i32 1, i32 -1, i32 -1, ptr @generic_body, ptr @generic_body_wrapper, ptr null, i64 0, i32 0)
  ret void
}

define internal void @spmd_body(ptr %.global_tid., ptr %.bound_tid.) {
  ret void
}
define internal void @spmd_body_wrapper(i16 zeroext %0, i32 %1) {
  call void @spmd_body(ptr null, ptr null)
  ret void
}
define internal void @generic_body(ptr %.global_tid., ptr %.bound_tid.) {
  ret void
}
define internal void @generic_body_wrapper(i16 zeroext %0, i32 %1) {
  call void @generic_body(ptr null, ptr null)
  ret void
}

declare void @unknown()
declare i32 @__kmpc_target_init(ptr, ptr)
declare void @__kmpc_target_deinit()
declare void @__kmpc_parallel_60(ptr, i32, i32, i32, i32, ptr, ptr, ptr, i64, i32)
declare fastcc i32 @__kmpc_get_hardware_thread_id_in_block()
declare void @__kmpc_barrier_simple_spmd(ptr, i32)

!llvm.module.flags = !{!0, !1}
!0 = !{i32 7, !"openmp", i32 51}
!1 = !{i32 7, !"openmp-device", i32 51}
