; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=hawaii -loop-unroll -unroll-threshold=49 -unroll-peel-count=0 -unroll-allow-partial=false -unroll-max-iteration-count-to-analyze=16 < %s | FileCheck %s

; CHECK-LABEL: @test_func_addrspacecast_cost_noop(
; CHECK-NOT: br i1
define amdgpu_kernel void @test_func_addrspacecast_cost_noop(float addrspace(1)* noalias nocapture %out, float addrspace(1)* noalias nocapture %in) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi float [ %fmul, %for.body ], [ 0.0, %entry ]
  %arrayidx.in = getelementptr inbounds float, float addrspace(1)* %in, i32 %indvars.iv
  %arrayidx.out = getelementptr inbounds float, float addrspace(1)* %out, i32 %indvars.iv
  %cast.in = addrspacecast float addrspace(1)* %arrayidx.in to float*
  %cast.out = addrspacecast float addrspace(1)* %arrayidx.out to float*
  %load = load float, float* %cast.in
  %fmul = fmul float %load, %sum.02
  store float %fmul, float* %cast.out
  %indvars.iv.next = add i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 16
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; Free, but not a no-op
; CHECK-LABEL: @test_func_addrspacecast_cost_free(
; CHECK-NOT: br i1
define amdgpu_kernel void @test_func_addrspacecast_cost_free(float* noalias nocapture %out, float* noalias nocapture %in) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi float [ %fmul, %for.body ], [ 0.0, %entry ]
  %arrayidx.in = getelementptr inbounds float, float* %in, i32 %indvars.iv
  %arrayidx.out = getelementptr inbounds float, float* %out, i32 %indvars.iv
  %cast.in = addrspacecast float* %arrayidx.in to float addrspace(3)*
  %cast.out = addrspacecast float* %arrayidx.out to float addrspace(3)*
  %load = load float, float addrspace(3)* %cast.in
  %fmul = fmul float %load, %sum.02
  store float %fmul, float addrspace(3)* %cast.out
  %indvars.iv.next = add i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 16
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK-LABEL: @test_func_addrspacecast_cost_nonfree(
; CHECK: br i1 %exitcond
define amdgpu_kernel void @test_func_addrspacecast_cost_nonfree(float addrspace(3)* noalias nocapture %out, float addrspace(3)* noalias nocapture %in) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi float [ %fmul, %for.body ], [ 0.0, %entry ]
  %arrayidx.in = getelementptr inbounds float, float addrspace(3)* %in, i32 %indvars.iv
  %arrayidx.out = getelementptr inbounds float, float addrspace(3)* %out, i32 %indvars.iv
  %cast.in = addrspacecast float addrspace(3)* %arrayidx.in to float*
  %cast.out = addrspacecast float addrspace(3)* %arrayidx.out to float*
  %load = load float, float* %cast.in
  %fmul = fmul float %load, %sum.02
  store float %fmul, float* %cast.out
  %indvars.iv.next = add i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 16
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
