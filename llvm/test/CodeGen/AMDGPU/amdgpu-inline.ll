; RUN: opt -mtriple=amdgcn--amdhsa -O3 -S -amdgpu-function-calls -inline-threshold=1 < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-INL1 %s
; RUN: opt -mtriple=amdgcn--amdhsa -O3 -S -amdgpu-function-calls < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-INLDEF %s

define coldcc float @foo(float %x, float %y) {
entry:
  %cmp = fcmp ogt float %x, 0.000000e+00
  %div = fdiv float %y, %x
  %mul = fmul float %x, %y
  %cond = select i1 %cmp, float %div, float %mul
  ret float %cond
}

define coldcc void @foo_private_ptr(float* nocapture %p) {
entry:
  %tmp1 = load float, float* %p, align 4
  %cmp = fcmp ogt float %tmp1, 1.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %div = fdiv float 1.000000e+00, %tmp1
  store float %div, float* %p, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define coldcc void @foo_private_ptr2(float* nocapture %p1, float* nocapture %p2) {
entry:
  %tmp1 = load float, float* %p1, align 4
  %cmp = fcmp ogt float %tmp1, 1.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %div = fdiv float 2.000000e+00, %tmp1
  store float %div, float* %p2, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define coldcc float @sin_wrapper(float %x) {
bb:
  %call = tail call float @_Z3sinf(float %x)
  ret float %call
}

define void @foo_noinline(float* nocapture %p) #0 {
entry:
  %tmp1 = load float, float* %p, align 4
  %mul = fmul float %tmp1, 2.000000e+00
  store float %mul, float* %p, align 4
  ret void
}

; GCN: define amdgpu_kernel void @test_inliner(
; GCN-INL1:   %c1 = tail call coldcc float @foo(
; GCN-INLDEF: %cmp.i = fcmp ogt float %tmp2, 0.000000e+00
; GCN:        %div.i{{[0-9]*}} = fdiv float 1.000000e+00, %c
; GCN:        %div.i{{[0-9]*}} = fdiv float 2.000000e+00, %tmp1.i
; GCN:        call void @foo_noinline(
; GCN:        tail call float @_Z3sinf(
define amdgpu_kernel void @test_inliner(float addrspace(1)* nocapture %a, i32 %n) {
entry:
  %pvt_arr = alloca [64 x float], align 4
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i32 %tid
  %tmp2 = load float, float addrspace(1)* %arrayidx, align 4
  %add = add i32 %tid, 1
  %arrayidx2 = getelementptr inbounds float, float addrspace(1)* %a, i32 %add
  %tmp5 = load float, float addrspace(1)* %arrayidx2, align 4
  %c1 = tail call coldcc float @foo(float %tmp2, float %tmp5)
  %or = or i32 %tid, %n
  %arrayidx5 = getelementptr inbounds [64 x float], [64 x float]* %pvt_arr, i32 0, i32 %or
  store float %c1, float* %arrayidx5, align 4
  %arrayidx7 = getelementptr inbounds [64 x float], [64 x float]* %pvt_arr, i32 0, i32 %or
  call coldcc void @foo_private_ptr(float* %arrayidx7)
  %arrayidx8 = getelementptr inbounds [64 x float], [64 x float]* %pvt_arr, i32 0, i32 1
  %arrayidx9 = getelementptr inbounds [64 x float], [64 x float]* %pvt_arr, i32 0, i32 2
  call coldcc void @foo_private_ptr2(float* %arrayidx8, float* %arrayidx9)
  call void @foo_noinline(float* %arrayidx7)
  %and = and i32 %tid, %n
  %arrayidx11 = getelementptr inbounds [64 x float], [64 x float]* %pvt_arr, i32 0, i32 %and
  %tmp12 = load float, float* %arrayidx11, align 4
  %c2 = call coldcc float @sin_wrapper(float %tmp12)
  store float %c2, float* %arrayidx7, align 4
  %xor = xor i32 %tid, %n
  %arrayidx16 = getelementptr inbounds [64 x float], [64 x float]* %pvt_arr, i32 0, i32 %xor
  %tmp16 = load float, float* %arrayidx16, align 4
  store float %tmp16, float addrspace(1)* %arrayidx, align 4
  ret void
}

; GCN: define amdgpu_kernel void @test_inliner_multi_pvt_ptr(
; GCN: %div.i{{[0-9]*}} = fdiv float 2.000000e+00, %tmp1.i
define amdgpu_kernel void @test_inliner_multi_pvt_ptr(float addrspace(1)* nocapture %a, i32 %n, float %v) {
entry:
  %pvt_arr1 = alloca [32 x float], align 4
  %pvt_arr2 = alloca [32 x float], align 4
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i32 %tid
  %or = or i32 %tid, %n
  %arrayidx4 = getelementptr inbounds [32 x float], [32 x float]* %pvt_arr1, i32 0, i32 %or
  %arrayidx5 = getelementptr inbounds [32 x float], [32 x float]* %pvt_arr2, i32 0, i32 %or
  store float %v, float* %arrayidx4, align 4
  store float %v, float* %arrayidx5, align 4
  %arrayidx8 = getelementptr inbounds [32 x float], [32 x float]* %pvt_arr1, i32 0, i32 1
  %arrayidx9 = getelementptr inbounds [32 x float], [32 x float]* %pvt_arr2, i32 0, i32 2
  call coldcc void @foo_private_ptr2(float* %arrayidx8, float* %arrayidx9)
  %xor = xor i32 %tid, %n
  %arrayidx15 = getelementptr inbounds [32 x float], [32 x float]* %pvt_arr1, i32 0, i32 %xor
  %arrayidx16 = getelementptr inbounds [32 x float], [32 x float]* %pvt_arr2, i32 0, i32 %xor
  %tmp15 = load float, float* %arrayidx15, align 4
  %tmp16 = load float, float* %arrayidx16, align 4
  %tmp17 = fadd float %tmp15, %tmp16
  store float %tmp17, float addrspace(1)* %arrayidx, align 4
  ret void
}

; GCN: define amdgpu_kernel void @test_inliner_multi_pvt_ptr_cutoff(
; GCN-INL1:   call coldcc void @foo_private_ptr2
; GCN-INLDEF: %div.i{{[0-9]*}} = fdiv float 2.000000e+00, %tmp1.i
define amdgpu_kernel void @test_inliner_multi_pvt_ptr_cutoff(float addrspace(1)* nocapture %a, i32 %n, float %v) {
entry:
  %pvt_arr1 = alloca [32 x float], align 4
  %pvt_arr2 = alloca [33 x float], align 4
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i32 %tid
  %or = or i32 %tid, %n
  %arrayidx4 = getelementptr inbounds [32 x float], [32 x float]* %pvt_arr1, i32 0, i32 %or
  %arrayidx5 = getelementptr inbounds [33 x float], [33 x float]* %pvt_arr2, i32 0, i32 %or
  store float %v, float* %arrayidx4, align 4
  store float %v, float* %arrayidx5, align 4
  %arrayidx8 = getelementptr inbounds [32 x float], [32 x float]* %pvt_arr1, i32 0, i32 1
  %arrayidx9 = getelementptr inbounds [33 x float], [33 x float]* %pvt_arr2, i32 0, i32 2
  call coldcc void @foo_private_ptr2(float* %arrayidx8, float* %arrayidx9)
  %xor = xor i32 %tid, %n
  %arrayidx15 = getelementptr inbounds [32 x float], [32 x float]* %pvt_arr1, i32 0, i32 %xor
  %arrayidx16 = getelementptr inbounds [33 x float], [33 x float]* %pvt_arr2, i32 0, i32 %xor
  %tmp15 = load float, float* %arrayidx15, align 4
  %tmp16 = load float, float* %arrayidx16, align 4
  %tmp17 = fadd float %tmp15, %tmp16
  store float %tmp17, float addrspace(1)* %arrayidx, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare float @_Z3sinf(float) #1

attributes #0 = { noinline }
attributes #1 = { nounwind readnone }
