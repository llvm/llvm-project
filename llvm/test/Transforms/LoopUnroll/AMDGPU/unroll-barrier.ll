; RUN: opt -mtriple=amdgcn-unknown-amdhsa -mcpu=hawaii -passes=loop-unroll -S < %s | FileCheck %s

; CHECK-LABEL: @test_unroll_convergent_barrier(
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK-NOT: br
define amdgpu_kernel void @test_unroll_convergent_barrier(ptr addrspace(1) noalias nocapture %out, ptr addrspace(1) noalias nocapture %in) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx.in = getelementptr inbounds i32, ptr addrspace(1) %in, i32 %indvars.iv
  %arrayidx.out = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %indvars.iv
  %load = load i32, ptr addrspace(1) %arrayidx.in
  call void @llvm.amdgcn.s.barrier() #1
  %add = add i32 %load, %sum.02
  store i32 %add, ptr addrspace(1) %arrayidx.out
  %indvars.iv.next = add i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 4
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare void @llvm.amdgcn.s.barrier() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind convergent }
