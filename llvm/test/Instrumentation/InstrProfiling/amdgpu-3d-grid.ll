;; Test that AMDGPU PGO instrumentation generates contiguous counter arrays
;; and profile section symbols with CUID-based naming. The __llvm_profile_sampling_gpu
;; library function handles 3D block linearization internally.

; RUN: opt %s -mtriple=amdgpu-amd-amdhsa -passes=instrprof -S | FileCheck %s

@__hip_cuid_abcdef789 = addrspace(1) global i8 0
@__profn_kernel_3d = private constant [9 x i8] c"kernel_3d"

define amdgpu_kernel void @kernel_3d() {
  call void @llvm.instrprof.increment(ptr @__profn_kernel_3d, i64 12345, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

;; Per-function comdat counters (3D grid linearization is handled in the runtime library)
; CHECK: @__profc_kernel_3d = linkonce_odr protected addrspace(1) global [1 x i64]
; CHECK: @__llvm_prf_unifcnt_kernel_3d = linkonce_odr protected addrspace(1) global [1 x i64]

;; Check sampling guard calls library function
; CHECK: call i32 @__llvm_profile_sampling_gpu(i32 3)
; CHECK: call void @__llvm_profile_instrument_gpu(ptr addrspacecast (ptr addrspace(1) @__profc_kernel_3d to ptr), ptr addrspacecast (ptr addrspace(1) @__llvm_prf_unifcnt_kernel_3d to ptr), i64 1)
