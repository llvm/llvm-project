;; Test AMDGPU PGO instrumentation lowering with multiple basic blocks.
;; Verifies:
;; 1. Sampling decision is computed once in the entry block.
;; 2. Each instrumentation point calls __llvm_profile_instrument_gpu behind the
;;    sampling guard branch.
;; 3. No-sampling mode (sampling=0) calls __llvm_profile_instrument_gpu unconditionally.

; RUN: opt %s -mtriple=amdgcn-amd-amdhsa -passes=instrprof \
; RUN:   -offload-pgo-sampling=3 -S \
; RUN:   | FileCheck %s --check-prefix=SAMPLED
; RUN: opt %s -mtriple=amdgcn-amd-amdhsa -passes=instrprof \
; RUN:   -offload-pgo-sampling=0 -S \
; RUN:   | FileCheck %s --check-prefix=NOSAMPLE

@__hip_cuid_test01 = addrspace(1) global i8 0
@__profn_test_kernel = private constant [11 x i8] c"test_kernel"

define amdgpu_kernel void @test_kernel(ptr addrspace(1) %out, i32 %n) {
entry:
  call void @llvm.instrprof.increment(ptr @__profn_test_kernel, i64 111, i32 4, i32 0)
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @llvm.instrprof.increment(ptr @__profn_test_kernel, i64 111, i32 4, i32 1)
  store i32 1, ptr addrspace(1) %out
  br label %if.end

if.end:
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

;; ---- Sampled mode (sampling=3) ----

;; Entry block: sampling decision computed once
; SAMPLED-LABEL: define {{.*}} @test_kernel
; SAMPLED: entry:
; SAMPLED: %pgo.sampled = call i32 @__llvm_profile_sampling_gpu(i32 3)
; SAMPLED: %pgo.matched = icmp ne i32 %pgo.sampled, 0
; SAMPLED: br i1 %pgo.matched, label %po_then, label %po_cont

;; Second instrumentation point reuses same sampling decision
; SAMPLED: br i1 %pgo.matched, label %po_then{{[0-9]+}}, label %po_cont{{[0-9]+}}

;; Both instrumentation points call the library function
; SAMPLED: call void @__llvm_profile_instrument_gpu(
; SAMPLED: call void @__llvm_profile_instrument_gpu(

;; ---- No-sampling mode (sampling=0) ----

;; No sampling guard — direct call
; NOSAMPLE-LABEL: define {{.*}} @test_kernel
; NOSAMPLE: entry:
; NOSAMPLE-NOT: @__llvm_profile_sampling_gpu
; NOSAMPLE: call void @__llvm_profile_instrument_gpu(
