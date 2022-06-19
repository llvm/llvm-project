; RUN: opt -mtriple amdgcn-unknown-amdhsa -passes='print<divergence>' -disable-output %s 2>&1 | FileCheck %s

declare i32 @gf2(i32)
declare i32 @gf1(i32)

define  void @tw1(i32 addrspace(4)* noalias nocapture readonly %A, i32 addrspace(4)* noalias nocapture %B) local_unnamed_addr #2 {
; CHECK: Divergence Analysis' for function 'tw1':
; CHECK: DIVERGENT: i32 addrspace(4)* %A
; CHECK: DIVERGENT: i32 addrspace(4)* %B
entry:
; CHECK: DIVERGENT:       %call = tail call i32 @gf2(i32 0) #0
; CHECK: DIVERGENT:       %cmp = icmp ult i32 %call, 16
; CHECK: DIVERGENT:       br i1 %cmp, label %if.then, label %new_exit
  %call = tail call  i32 @gf2(i32 0) #3
  %cmp = icmp ult i32 %call, 16
  br i1 %cmp, label %if.then, label %new_exit

if.then:
; CHECK: DIVERGENT:       %call1 = tail call i32 @gf1(i32 0) #0
; CHECK: DIVERGENT:       %arrayidx = getelementptr inbounds i32, i32 addrspace(4)* %A, i32 %call1
; CHECK: DIVERGENT:       %0 = load i32, i32 addrspace(4)* %arrayidx, align 4
; CHECK: DIVERGENT:       %cmp225 = icmp sgt i32 %0, 0
; CHECK: DIVERGENT:       %arrayidx10 = getelementptr inbounds i32, i32 addrspace(4)* %B, i32 %call1
; CHECK: DIVERGENT:       br i1 %cmp225, label %while.body.preheader, label %if.then.while.end_crit_edge
  %call1 = tail call  i32 @gf1(i32 0) #4
  %arrayidx = getelementptr inbounds i32, i32 addrspace(4)* %A, i32 %call1
  %0 = load i32, i32 addrspace(4)* %arrayidx, align 4
  %cmp225 = icmp sgt i32 %0, 0
  %arrayidx10 = getelementptr inbounds i32, i32 addrspace(4)* %B, i32 %call1
  br i1 %cmp225, label %while.body.preheader, label %if.then.while.end_crit_edge

while.body.preheader:
  br label %while.body

if.then.while.end_crit_edge:
; CHECK: DIVERGENT:       %.pre = load i32, i32 addrspace(4)* %arrayidx10, align 4
  %.pre = load i32, i32 addrspace(4)* %arrayidx10, align 4
  br label %while.end

while.body:
; CHECK-NOT: DIVERGENT:                  %i.026 = phi i32 [ %inc, %if.end.while.body_crit_edge ], [ 0, %while.body.preheader ]
; CHECK: DIVERGENT:       %call3 = tail call i32 @gf1(i32 0) #0
; CHECK: DIVERGENT:       %cmp4 = icmp ult i32 %call3, 10
; CHECK: DIVERGENT:       %arrayidx6 = getelementptr inbounds i32, i32 addrspace(4)* %A, i32 %i.026
; CHECK: DIVERGENT:       %1 = load i32, i32 addrspace(4)* %arrayidx6, align 4
; CHECK: DIVERGENT:       br i1 %cmp4, label %if.then5, label %if.else
  %i.026 = phi i32 [ %inc, %if.end.while.body_crit_edge ], [ 0, %while.body.preheader ]
  %call3 = tail call  i32 @gf1(i32 0) #4
  %cmp4 = icmp ult i32 %call3, 10
  %arrayidx6 = getelementptr inbounds i32, i32 addrspace(4)* %A, i32 %i.026
  %1 = load i32, i32 addrspace(4)* %arrayidx6, align 4
  br i1 %cmp4, label %if.then5, label %if.else

if.then5:
; CHECK: DIVERGENT:       %mul = shl i32 %1, 1
; CHECK: DIVERGENT:       %2 = load i32, i32 addrspace(4)* %arrayidx10, align 4
; CHECK: DIVERGENT:       %add = add nsw i32 %2, %mul
  %mul = shl i32 %1, 1
  %2 = load i32, i32 addrspace(4)* %arrayidx10, align 4
  %add = add nsw i32 %2, %mul
  br label %if.end

if.else:
; CHECK: DIVERGENT:       %mul9 = shl i32 %1, 2
; CHECK: DIVERGENT:       %3 = load i32, i32 addrspace(4)* %arrayidx10, align 4
; CHECK: DIVERGENT:       %add11 = add nsw i32 %3, %mul9
  %mul9 = shl i32 %1, 2
  %3 = load i32, i32 addrspace(4)* %arrayidx10, align 4
  %add11 = add nsw i32 %3, %mul9
  br label %if.end

if.end:
; CHECK: DIVERGENT:       %storemerge = phi i32 [ %add11, %if.else ], [ %add, %if.then5 ]
; CHECK: DIVERGENT:       store i32 %storemerge, i32 addrspace(4)* %arrayidx10, align 4
; CHECK-NOT: DIVERGENT:                  %inc = add nuw nsw i32 %i.026, 1
; CHECK: DIVERGENT:       %exitcond = icmp ne i32 %inc, %0
; CHECK: DIVERGENT:       br i1 %exitcond, label %if.end.while.body_crit_edge, label %while.end.loopexit
  %storemerge = phi i32 [ %add11, %if.else ], [ %add, %if.then5 ]
  store i32 %storemerge, i32 addrspace(4)* %arrayidx10, align 4
  %inc = add nuw nsw i32 %i.026, 1
  %exitcond = icmp ne i32 %inc, %0
  br i1 %exitcond, label %if.end.while.body_crit_edge, label %while.end.loopexit

if.end.while.body_crit_edge:
  br label %while.body

while.end.loopexit:
; CHECK: DIVERGENT:       %storemerge.lcssa = phi i32 [ %storemerge, %if.end ]
  %storemerge.lcssa = phi i32 [ %storemerge, %if.end ]
  br label %while.end

while.end:
; CHECK: DIVERGENT:       %4 = phi i32 [ %.pre, %if.then.while.end_crit_edge ], [ %storemerge.lcssa, %while.end.loopexit ]
; CHECK: DIVERGENT:       %i.0.lcssa = phi i32 [ 0, %if.then.while.end_crit_edge ], [ %0, %while.end.loopexit ]
; CHECK: DIVERGENT:       %sub = sub nsw i32 %4, %i.0.lcssa
; CHECK: DIVERGENT:       store i32 %sub, i32 addrspace(4)* %arrayidx10, align 4
  %4 = phi i32 [ %.pre, %if.then.while.end_crit_edge ], [ %storemerge.lcssa, %while.end.loopexit ]
  %i.0.lcssa = phi i32 [ 0, %if.then.while.end_crit_edge ], [ %0, %while.end.loopexit ]
  %sub = sub nsw i32 %4, %i.0.lcssa
  store i32 %sub, i32 addrspace(4)* %arrayidx10, align 4
  br label %new_exit

new_exit:
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind readnone }
