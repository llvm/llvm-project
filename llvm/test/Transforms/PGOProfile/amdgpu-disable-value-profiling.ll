;; Test that value profiling (indirect call profiling) is disabled for GPU targets.
;; The device-side profiling runtime does not implement
;; __llvm_profile_instrument_target, so indirect call profiling must not be emitted.

; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@fptr = addrspace(1) global ptr null, align 8

;; Verify that regular block instrumentation IS emitted
; CHECK: call void @llvm.instrprof.increment

;; Verify that value profiling for indirect calls is NOT emitted
; CHECK-NOT: call void @llvm.instrprof.value.profile

define amdgpu_kernel void @test_indirect_call() {
entry:
  %fp = load ptr, ptr addrspace(1) @fptr, align 8
  call void %fp()
  ret void
}
