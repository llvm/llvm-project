; REQUIRES: asserts
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-perf-hint --debug-only=amdgpu-perf-hint < %s 2>&1 | FileCheck %s

; Test that all matrix intrinsic families (wmma, mfma, swmmac, smfmac) are
; recognized by the perf hint analysis for LDS contention detection.

; CHECK: [AMDGPUPerfHint] process test_wmma_contention
; CHECK: Found cycle with LDS contention: 2 LDS loads, 1 matrix ops
define void @test_wmma_contention(ptr addrspace(3) %lds, <16 x half> %a, <16 x half> %b, <16 x half> %c, ptr addrspace(1) %out) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.next, %loop]
  %acc = phi <16 x half> [%c, %entry], [%result, %loop]
  %ptr1 = getelementptr <16 x half>, ptr addrspace(3) %lds, i32 %i
  %lds.val1 = load <16 x half>, ptr addrspace(3) %ptr1
  %i2 = add i32 %i, 1
  %ptr2 = getelementptr <16 x half>, ptr addrspace(3) %lds, i32 %i2
  %lds.val2 = load <16 x half>, ptr addrspace(3) %ptr2
  %sum = fadd <16 x half> %lds.val1, %lds.val2
  %result = call <16 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16(<16 x half> %sum, <16 x half> %b, <16 x half> %acc, i1 false)
  %i.next = add i32 %i, 2
  %cond = icmp ult i32 %i.next, 100
  br i1 %cond, label %loop, label %exit

exit:
  store <16 x half> %result, ptr addrspace(1) %out
  ret void
}

; CHECK: [AMDGPUPerfHint] process test_mfma_contention
; CHECK: Found cycle with LDS contention: 2 LDS loads, 1 matrix ops
define void @test_mfma_contention(ptr addrspace(3) %lds, float %a, float %b, <32 x float> %c, ptr addrspace(1) %out) {
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

; CHECK: [AMDGPUPerfHint] process test_swmmac_contention
; CHECK: Found cycle with LDS contention: 2 LDS loads, 1 matrix ops
define void @test_swmmac_contention(ptr addrspace(3) %lds, <8 x half> %a, <16 x half> %b, <8 x float> %c, i16 %idx, ptr addrspace(1) %out) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.next, %loop]
  %acc = phi <8 x float> [%c, %entry], [%result, %loop]
  %ptr1 = getelementptr <8 x half>, ptr addrspace(3) %lds, i32 %i
  %lds.val1 = load <8 x half>, ptr addrspace(3) %ptr1
  %i2 = add i32 %i, 1
  %ptr2 = getelementptr <8 x half>, ptr addrspace(3) %lds, i32 %i2
  %lds.val2 = load <8 x half>, ptr addrspace(3) %ptr2
  %sum = fadd <8 x half> %lds.val1, %lds.val2
  %result = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.f16(<8 x half> %sum, <16 x half> %b, <8 x float> %acc, i16 %idx)
  %i.next = add i32 %i, 2
  %cond = icmp ult i32 %i.next, 100
  br i1 %cond, label %loop, label %exit

exit:
  store <8 x float> %result, ptr addrspace(1) %out
  ret void
}

; CHECK: [AMDGPUPerfHint] process test_smfmac_contention
; CHECK: Found cycle with LDS contention: 2 LDS loads, 1 matrix ops
define void @test_smfmac_contention(ptr addrspace(3) %lds, <4 x half> %a, <8 x half> %b, <4 x float> %c, ptr addrspace(1) %out) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.next, %loop]
  %acc = phi <4 x float> [%c, %entry], [%result, %loop]
  %ptr1 = getelementptr <4 x half>, ptr addrspace(3) %lds, i32 %i
  %lds.val1 = load <4 x half>, ptr addrspace(3) %ptr1
  %i2 = add i32 %i, 1
  %ptr2 = getelementptr <4 x half>, ptr addrspace(3) %lds, i32 %i2
  %lds.val2 = load <4 x half>, ptr addrspace(3) %ptr2
  %sum = fadd <4 x half> %lds.val1, %lds.val2
  %result = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x32.f16(<4 x half> %sum, <8 x half> %b, <4 x float> %acc, i32 0, i32 0, i32 0)
  %i.next = add i32 %i, 2
  %cond = icmp ult i32 %i.next, 100
  br i1 %cond, label %loop, label %exit

exit:
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <16 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16(<16 x half>, <16 x half>, <16 x half>, i1 immarg)
declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32 immarg, i32 immarg, i32 immarg)
declare <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.f16(<8 x half>, <16 x half>, <8 x float>, i16)
declare <4 x float> @llvm.amdgcn.smfmac.f32.16x16x32.f16(<4 x half>, <8 x half>, <4 x float>, i32, i32, i32)
