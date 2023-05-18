; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=hawaii -passes=loop-unroll -unroll-threshold=100 -unroll-peel-count=0 -unroll-allow-partial=false -unroll-max-iteration-count-to-analyze=16 < %s | FileCheck %s

; CHECK-LABEL: @test_intrinsic_call_cost(
; CHECK-NOT: br i1
define amdgpu_kernel void @test_intrinsic_call_cost(ptr addrspace(1) noalias nocapture %out, ptr addrspace(1) noalias nocapture %in) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi float [ %fmul, %for.body ], [ 0.0, %entry ]
  %arrayidx.in = getelementptr inbounds float, ptr addrspace(1) %in, i32 %indvars.iv
  %arrayidx.out = getelementptr inbounds float, ptr addrspace(1) %out, i32 %indvars.iv
  %load = load float, ptr addrspace(1) %arrayidx.in
  %call = call float @llvm.minnum.f32(float %load, float 1.0);
  %fmul = fmul float %call, %sum.02
  store float %fmul, ptr addrspace(1) %arrayidx.out
  %indvars.iv.next = add i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 16
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK-LABEL: @test_func_call_cost(
; CHECK: br i1 %exitcond
define amdgpu_kernel void @test_func_call_cost(ptr addrspace(1) noalias nocapture %out, ptr addrspace(1) noalias nocapture %in) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi float [ %fmul, %for.body ], [ 0.0, %entry ]
  %arrayidx.in = getelementptr inbounds float, ptr addrspace(1) %in, i32 %indvars.iv
  %arrayidx.out = getelementptr inbounds float, ptr addrspace(1) %out, i32 %indvars.iv
  %load = load float, ptr addrspace(1) %arrayidx.in
  %fptr = load ptr, ptr addrspace(4) null
  %call = tail call float %fptr(float %load, float 1.0)
  %fmul = fmul float %call, %sum.02
  store float %fmul, ptr addrspace(1) %arrayidx.out
  %indvars.iv.next = add i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 16
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK-LABEL: @test_indirect_call_cost(
; CHECK: br i1 %exitcond
define amdgpu_kernel void @test_indirect_call_cost(ptr addrspace(1) noalias nocapture %out, ptr addrspace(1) noalias nocapture %in) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi float [ %fmul, %for.body ], [ 0.0, %entry ]
  %arrayidx.in = getelementptr inbounds float, ptr addrspace(1) %in, i32 %indvars.iv
  %arrayidx.out = getelementptr inbounds float, ptr addrspace(1) %out, i32 %indvars.iv
  %load = load float, ptr addrspace(1) %arrayidx.in
  %min = call float @func(float %load, float 1.0);
  %fmul = fmul float %min, %sum.02
  store float %fmul, ptr addrspace(1) %arrayidx.out
  %indvars.iv.next = add i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 16
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @llvm.minnum.f32(float, float) #1
declare float @func(float, float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
