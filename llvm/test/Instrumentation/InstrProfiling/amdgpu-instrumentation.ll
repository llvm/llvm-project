;; Test basic AMDGPU PGO instrumentation lowering.
;; Verifies that each instrumentation point lowers directly to a call to
;; __llvm_profile_instrument_gpu with a null uniform-counter argument.

; RUN: opt %s -mtriple=amdgcn-amd-amdhsa -passes=instrprof -S | FileCheck %s

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

; CHECK-LABEL: define {{.*}} @test_kernel
; CHECK-NOT: @__llvm_profile_sampling_gpu
; CHECK: call void @__llvm_profile_instrument_gpu(
; CHECK-SAME: ptr addrspacecast (ptr addrspace(1) @__profc_test_kernel to ptr), ptr null, i64 1)
; CHECK: call void @__llvm_profile_instrument_gpu(
; CHECK-SAME: ptr addrspacecast (ptr addrspace(1) getelementptr inbounds ([4 x i64], ptr addrspace(1) @__profc_test_kernel, i32 0, i32 1) to ptr), ptr null, i64 1)
