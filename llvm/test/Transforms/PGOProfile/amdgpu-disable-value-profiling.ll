;; Test that value profiling (indirect call profiling) is disabled for GPU targets.
;; The device-side profiling runtime does not implement
;; __llvm_profile_instrument_target, so indirect call profiling must not be emitted.

; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s

target triple = "amdgpu7.00-amd-amdhsa"

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
