;; Per-kernel __profc_* arrays land in section __llvm_prf_cnts with one slot
;; per counter, and a parallel __llvm_prf_unifcnt_* array lands in
;; __llvm_prf_ucnts. Counter increments lower to __llvm_profile_instrument_gpu
;; calls whose first pointer argument is a GEP into the per-kernel counter
;; array and whose second is a GEP into the matching uniform-counter array.
;; Sampling is disabled so the increment lowers to a direct call.

; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -offload-pgo-sampling=0 -passes=instrprof < %s | FileCheck %s

@__profn_kernel1 = private constant [7 x i8] c"kernel1"
@__profn_kernel2 = private constant [7 x i8] c"kernel2"

; CHECK: @__profc_kernel1 = linkonce_odr protected addrspace(1) global [2 x i64] zeroinitializer, section "__llvm_prf_cnts"
; CHECK: @__llvm_prf_unifcnt_kernel1 = linkonce_odr protected addrspace(1) global [2 x i64] zeroinitializer, section "__llvm_prf_ucnts"
; CHECK: @__profc_kernel2 = linkonce_odr protected addrspace(1) global [1 x i64] zeroinitializer, section "__llvm_prf_cnts"

define amdgpu_kernel void @kernel1() {
  call void @llvm.instrprof.increment(ptr @__profn_kernel1, i64 12345, i32 2, i32 0)
  call void @llvm.instrprof.increment(ptr @__profn_kernel1, i64 12345, i32 2, i32 1)
  ret void
}

define amdgpu_kernel void @kernel2() {
  call void @llvm.instrprof.increment(ptr @__profn_kernel2, i64 67890, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

;; Second counter slot uses a GEP into the per-kernel counter array and the
;; matching uniform-counter array.
; CHECK: call void @__llvm_profile_instrument_gpu(ptr addrspacecast (ptr addrspace(1) getelementptr inbounds ([2 x i64], ptr addrspace(1) @__profc_kernel1, i32 0, i32 1) to ptr), ptr addrspacecast (ptr addrspace(1) getelementptr inbounds ([2 x i64], ptr addrspace(1) @__llvm_prf_unifcnt_kernel1, i32 0, i32 1) to ptr), i64 1)
