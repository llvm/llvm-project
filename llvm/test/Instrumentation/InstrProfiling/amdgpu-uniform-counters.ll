;; Test that AMDGPU targets generate uniform counter arrays alongside regular
;; counters. The uniform counter is passed to __llvm_profile_instrument_gpu which
;; updates it when all lanes in the wave are active.

; RUN: opt %s -mtriple=amdgcn-amd-amdhsa -passes=instrprof -S | FileCheck %s

@__hip_cuid_test123 = addrspace(1) global i8 0
@__profn_test_kernel = private constant [11 x i8] c"test_kernel"

define amdgpu_kernel void @test_kernel() {
  call void @llvm.instrprof.increment(ptr @__profn_test_kernel, i64 12345, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

;; Per-function counter + uniform-counter globals (comdat)
; CHECK: @__profc_test_kernel = linkonce_odr protected addrspace(1) global [1 x i64]
; CHECK: @__llvm_prf_unifcnt_test_kernel = linkonce_odr protected addrspace(1) global [1 x i64]

;; __llvm_profile_instrument_gpu receives counter and uniform-counter bases
; CHECK: call void @__llvm_profile_instrument_gpu(ptr addrspacecast (ptr addrspace(1) @__profc_test_kernel to ptr), ptr addrspacecast (ptr addrspace(1) @__llvm_prf_unifcnt_test_kernel to ptr), i64 1)
