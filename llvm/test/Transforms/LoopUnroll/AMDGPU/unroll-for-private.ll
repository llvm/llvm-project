; RUN: opt -data-layout=A5 -mtriple=amdgcn-unknown-amdhsa -loop-unroll -S -amdgpu-unroll-threshold-private=20000 %s | FileCheck %s

; Check that we full unroll loop to be able to eliminate alloca
; CHECK-LABEL: @non_invariant_ind
; CHECK:       for.body:
; CHECK-NOT:   br
; CHECK:       store i32 %tmp15, i32 addrspace(1)* %arrayidx7, align 4
; CHECK:       ret void

define amdgpu_kernel void @non_invariant_ind(i32 addrspace(1)* nocapture %a, i32 %x) {
entry:
  %arr = alloca [64 x i32], align 4, addrspace(5)
  %tmp1 = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %arrayidx5 = getelementptr inbounds [64 x i32], [64 x i32] addrspace(5)* %arr, i32 0, i32 %x
  %tmp15 = load i32, i32 addrspace(5)* %arrayidx5, align 4
  %arrayidx7 = getelementptr inbounds i32, i32 addrspace(1)* %a, i32 %tmp1
  store i32 %tmp15, i32 addrspace(1)* %arrayidx7, align 4
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.015 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %idxprom = sext i32 %i.015 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %idxprom
  %tmp16 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %add = add nsw i32 %i.015, %tmp1
  %rem = srem i32 %add, 64
  %arrayidx3 = getelementptr inbounds [64 x i32], [64 x i32] addrspace(5)* %arr, i32 0, i32 %rem
  store i32 %tmp16, i32 addrspace(5)* %arrayidx3, align 4
  %inc = add nuw nsw i32 %i.015, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Check that we unroll inner loop but not outer
; CHECK-LABEL: @invariant_ind
; CHECK:       %[[exitcond:[^ ]+]] = icmp eq i32 %{{.*}}, 32
; CHECK:       br i1 %[[exitcond]]
; CHECK-NOT:   icmp eq i32 %{{.*}}, 100

define amdgpu_kernel void @invariant_ind(i32 addrspace(1)* nocapture %a, i32 %x) {
entry:
  %arr = alloca [64 x i32], align 4, addrspace(5)
  %tmp1 = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %for.cond.cleanup5, %entry
  %i.026 = phi i32 [ 0, %entry ], [ %inc10, %for.cond.cleanup5 ]
  %idxprom = sext i32 %i.026 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %idxprom
  %tmp15 = load i32, i32 addrspace(1)* %arrayidx, align 4
  br label %for.body6

for.cond.cleanup:                                 ; preds = %for.cond.cleanup5
  %arrayidx13 = getelementptr inbounds [64 x i32], [64 x i32] addrspace(5)* %arr, i32 0, i32 %x
  %tmp16 = load i32, i32 addrspace(5)* %arrayidx13, align 4
  %arrayidx15 = getelementptr inbounds i32, i32 addrspace(1)* %a, i32 %tmp1
  store i32 %tmp16, i32 addrspace(1)* %arrayidx15, align 4
  ret void

for.cond.cleanup5:                                ; preds = %for.body6
  %inc10 = add nuw nsw i32 %i.026, 1
  %exitcond27 = icmp eq i32 %inc10, 32
  br i1 %exitcond27, label %for.cond.cleanup, label %for.cond2.preheader

for.body6:                                        ; preds = %for.body6, %for.cond2.preheader
  %j.025 = phi i32 [ 0, %for.cond2.preheader ], [ %inc, %for.body6 ]
  %add = add nsw i32 %j.025, %tmp1
  %rem = srem i32 %add, 64
  %arrayidx8 = getelementptr inbounds [64 x i32], [64 x i32] addrspace(5)* %arr, i32 0, i32 %rem
  store i32 %tmp15, i32 addrspace(5)* %arrayidx8, align 4
  %inc = add nuw nsw i32 %j.025, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.cond.cleanup5, label %for.body6
}

; Check we do not enforce unroll if alloca is too big
; CHECK-LABEL: @too_big
; CHECK:       for.body:
; CHECK:       icmp eq i32 %{{.*}}, 100
; CHECK:       br

define amdgpu_kernel void @too_big(i32 addrspace(1)* nocapture %a, i32 %x) {
entry:
  %arr = alloca [256 x i32], align 4, addrspace(5)
  %tmp1 = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %arrayidx5 = getelementptr inbounds [256 x i32], [256 x i32] addrspace(5)* %arr, i32 0, i32 %x
  %tmp15 = load i32, i32 addrspace(5)* %arrayidx5, align 4
  %arrayidx7 = getelementptr inbounds i32, i32 addrspace(1)* %a, i32 %tmp1
  store i32 %tmp15, i32 addrspace(1)* %arrayidx7, align 4
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.015 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %idxprom = sext i32 %i.015 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %idxprom
  %tmp16 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %add = add nsw i32 %i.015, %tmp1
  %rem = srem i32 %add, 64
  %arrayidx3 = getelementptr inbounds [256 x i32], [256 x i32] addrspace(5)* %arr, i32 0, i32 %rem
  store i32 %tmp16, i32 addrspace(5)* %arrayidx3, align 4
  %inc = add nuw nsw i32 %i.015, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Check we do not enforce unroll if alloca is dynamic
; CHECK-LABEL: @dynamic_size_alloca(
; CHECK: alloca i32, i32 %n
; CHECK:       for.body:
; CHECK:       icmp eq i32 %{{.*}}, 100
; CHECK:       br

define amdgpu_kernel void @dynamic_size_alloca(i32 addrspace(1)* nocapture %a, i32 %n, i32 %x) {
entry:
  %arr = alloca i32, i32 %n, align 4, addrspace(5)
  %tmp1 = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %arrayidx5 = getelementptr inbounds i32, i32 addrspace(5)* %arr, i32 %x
  %tmp15 = load i32, i32 addrspace(5)* %arrayidx5, align 4
  %arrayidx7 = getelementptr inbounds i32, i32 addrspace(1)* %a, i32 %tmp1
  store i32 %tmp15, i32 addrspace(1)* %arrayidx7, align 4
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.015 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %idxprom = sext i32 %i.015 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %idxprom
  %tmp16 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %add = add nsw i32 %i.015, %tmp1
  %rem = srem i32 %add, 64
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(5)* %arr, i32 %rem
  store i32 %tmp16, i32 addrspace(5)* %arrayidx3, align 4
  %inc = add nuw nsw i32 %i.015, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

declare i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #1

declare i32 @llvm.amdgcn.workitem.id.x() #1

declare i32 @llvm.amdgcn.workgroup.id.x() #1

declare i8 addrspace(2)* @llvm.amdgcn.implicitarg.ptr() #1

attributes #1 = { nounwind readnone }
