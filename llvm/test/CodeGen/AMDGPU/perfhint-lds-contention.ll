; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-perf-hint < %s | FileCheck %s

; Test that AMDGPUPerfHintsAnalysis sets the amdgpu-lds-contention attribute
; based on the ratio of LDS loads to matrix operations in innermost loops.
; Contention is flagged when: LDSLoadCount * 2 >= MatrixInstCount

; High LDS ratio: 2 LDS loads, 1 matrix op -> 2*2=4 >= 1, contention=true
; CHECK-LABEL: define void @high_lds_ratio(
; CHECK-SAME: #[[HIGH_ATTR:[0-9]+]]
define void @high_lds_ratio(ptr addrspace(3) %lds, float %a, float %b, <32 x float> %c, ptr addrspace(1) %out) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.next, %loop]
  %acc = phi <32 x float> [%c, %entry], [%result, %loop]
  %ptr1 = getelementptr float, ptr addrspace(3) %lds, i32 %i
  %lds.val1 = load float, ptr addrspace(3) %ptr1
  %i2 = add i32 %i, 1
  %ptr2 = getelementptr float, ptr addrspace(3) %lds, i32 %i2
  %lds.val2 = load float, ptr addrspace(3) %ptr2
  %sum = fadd float %lds.val1, %lds.val2
  %result = call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float %sum, float %b, <32 x float> %acc, i32 0, i32 0, i32 0)
  %i.next = add i32 %i, 2
  %cond = icmp ult i32 %i.next, 100
  br i1 %cond, label %loop, label %exit

exit:
  store <32 x float> %result, ptr addrspace(1) %out
  ret void
}

; Low LDS ratio: 1 LDS load, 4 matrix ops -> 1*2=2 < 4, contention=false
; CHECK-LABEL: define void @low_lds_ratio(
; CHECK-SAME: #[[LOW_ATTR:[0-9]+]]
define void @low_lds_ratio(ptr addrspace(3) %lds, float %a, float %b, <32 x float> %c, ptr addrspace(1) %out) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.next, %loop]
  %acc = phi <32 x float> [%c, %entry], [%result4, %loop]
  %ptr1 = getelementptr float, ptr addrspace(3) %lds, i32 %i
  %lds.val = load float, ptr addrspace(3) %ptr1
  %result1 = call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float %lds.val, float %b, <32 x float> %acc, i32 0, i32 0, i32 0)
  %result2 = call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float %lds.val, float %b, <32 x float> %result1, i32 0, i32 0, i32 0)
  %result3 = call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float %lds.val, float %b, <32 x float> %result2, i32 0, i32 0, i32 0)
  %result4 = call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float %lds.val, float %b, <32 x float> %result3, i32 0, i32 0, i32 0)
  %i.next = add i32 %i, 1
  %cond = icmp ult i32 %i.next, 100
  br i1 %cond, label %loop, label %exit

exit:
  store <32 x float> %result4, ptr addrspace(1) %out
  ret void
}

; Boundary case: 1 LDS load, 2 matrix ops -> 1*2=2 >= 2, contention=true
; CHECK-LABEL: define void @boundary_lds_ratio(
; CHECK-SAME: #[[BOUNDARY_ATTR:[0-9]+]]
define void @boundary_lds_ratio(ptr addrspace(3) %lds, float %a, float %b, <32 x float> %c, ptr addrspace(1) %out) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.next, %loop]
  %acc = phi <32 x float> [%c, %entry], [%result2, %loop]
  %ptr1 = getelementptr float, ptr addrspace(3) %lds, i32 %i
  %lds.val = load float, ptr addrspace(3) %ptr1
  %result1 = call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float %lds.val, float %b, <32 x float> %acc, i32 0, i32 0, i32 0)
  %result2 = call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float %lds.val, float %b, <32 x float> %result1, i32 0, i32 0, i32 0)
  %i.next = add i32 %i, 1
  %cond = icmp ult i32 %i.next, 100
  br i1 %cond, label %loop, label %exit

exit:
  store <32 x float> %result2, ptr addrspace(1) %out
  ret void
}

; User-specified amdgpu-lds-contention="false" should not be overridden by heuristic
; CHECK-LABEL: define void @user_override_false(
; CHECK-SAME: #[[USER_FALSE_ATTR:[0-9]+]]
define void @user_override_false(ptr addrspace(3) %lds, float %a, float %b, <32 x float> %c, ptr addrspace(1) %out) #0 {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.next, %loop]
  %acc = phi <32 x float> [%c, %entry], [%result, %loop]
  %ptr1 = getelementptr float, ptr addrspace(3) %lds, i32 %i
  %lds.val1 = load float, ptr addrspace(3) %ptr1
  %i2 = add i32 %i, 1
  %ptr2 = getelementptr float, ptr addrspace(3) %lds, i32 %i2
  %lds.val2 = load float, ptr addrspace(3) %ptr2
  %sum = fadd float %lds.val1, %lds.val2
  %result = call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float %sum, float %b, <32 x float> %acc, i32 0, i32 0, i32 0)
  %i.next = add i32 %i, 2
  %cond = icmp ult i32 %i.next, 100
  br i1 %cond, label %loop, label %exit

exit:
  store <32 x float> %result, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-lds-contention"="false" }

; High ratio and boundary case should have lds-contention=true (same attribute group)
; CHECK-DAG: attributes #[[HIGH_ATTR]]{{.*}}"amdgpu-lds-contention"="true"
; Low ratio should have memory-bound but NOT lds-contention
; CHECK-DAG: attributes #[[LOW_ATTR]] = { "amdgpu-memory-bound"="true" }
; User-specified false should be preserved (not overridden to true)
; CHECK-DAG: attributes #[[USER_FALSE_ATTR]] = { "amdgpu-lds-contention"="false"

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32 immarg, i32 immarg, i32 immarg)
