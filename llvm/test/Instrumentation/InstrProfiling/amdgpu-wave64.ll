;; Test that AMDGPU PGO instrumentation generates library calls for Wave64.
;; Wave64 targets (e.g., gfx908) should embed wave size 64 in profile data.

; RUN: opt %s -mtriple=amdgpu-amd-amdhsa -passes=instrprof -S | FileCheck %s

@__hip_cuid_abcdef123 = addrspace(1) global i8 0
@__profn_kernel_w64 = private constant [10 x i8] c"kernel_w64"

define amdgpu_kernel void @kernel_w64() #0 {
  call void @llvm.instrprof.increment(ptr @__profn_kernel_w64, i64 12345, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

attributes #0 = { "target-cpu"="gfx908" }

;; Per-function comdat counters
; CHECK: @__profc_kernel_w64 = linkonce_odr protected addrspace(1) global [1 x i64]
; CHECK: @__llvm_prf_unifcnt_kernel_w64 = linkonce_odr protected addrspace(1) global [1 x i64]
; CHECK: @__profd_kernel_w64 = linkonce_odr protected addrspace(1) global { {{.*}} i16 0, i32 0 }

;; Check wave size stored via intrinsic
; CHECK: %wavesize.i16 = trunc i32 %{{.*}} to i16
; CHECK: store i16 %wavesize.i16, ptr addrspace(1) getelementptr inbounds {{.*}} @__profd_kernel_w64

;; Check sampling guard
; CHECK: %pgo.sampled = call i32 @__llvm_profile_sampling_gpu(i32 3)
; CHECK: %pgo.matched = icmp ne i32 %pgo.sampled, 0
; CHECK: br i1 %pgo.matched, label %po_then, label %po_cont

;; Check library call
; CHECK: po_then:
; CHECK: call void @__llvm_profile_instrument_gpu(ptr addrspacecast (ptr addrspace(1) @__profc_kernel_w64 to ptr), ptr addrspacecast (ptr addrspace(1) @__llvm_prf_unifcnt_kernel_w64 to ptr), i64 1)
