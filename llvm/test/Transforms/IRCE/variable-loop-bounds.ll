; RUN: opt -irce -S -verify-loop-info -irce-print-changed-loops -irce-skip-profitability-checks < %s 2>&1 | FileCheck %s

; CHECK: irce: in function test_inc_eq: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%if.then,%for.inc<latch><exiting>
; CHECK: irce: in function test_inc_ne: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%if.then,%for.inc<latch><exiting>
; CHECK: irce: in function test_inc_slt: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%if.then,%for.inc<latch><exiting>
; CHECK: irce: in function test_inc_ult: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%if.then,%for.inc<latch><exiting>

; CHECK-LABEL: test_inc_eq(
; CHECK: main.exit.selector:
; CHECK: [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %inc, %for.inc ]
; CHECK: [[COND:%[^ ]+]] = icmp ult i32 [[PSEUDO_PHI]], %N
; CHECK: br i1 [[COND]], label %main.pseudo.exit, label %for.cond.cleanup.loopexit
define void @test_inc_eq(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %cmp16 = icmp sgt i32 %N, 0
  br i1 %cmp16, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.017 = phi i32 [ %inc, %for.inc ], [ 0, %entry ]
  %cmp1 = icmp ult i32 %i.017, 512
  %arrayidx = getelementptr inbounds i32, i32* %b, i32 %i.017
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %c, i32 %i.017
  %1 = load i32, i32* %arrayidx2, align 4
  br i1 %cmp1, label %if.then, label %if.else

if.then:
  %sub = sub i32 %0, %1
  %arrayidx3 = getelementptr inbounds i32, i32* %a, i32 %i.017
  %2 = load i32, i32* %arrayidx3, align 4
  %add = add nsw i32 %sub, %2
  store i32 %add, i32* %arrayidx3, align 4
  br label %for.inc

if.else:
  %add6 = add nsw i32 %1, %0
  %arrayidx7 = getelementptr inbounds i32, i32* %a, i32 %i.017
  store i32 %add6, i32* %arrayidx7, align 4
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %i.017, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: test_inc_ne
; CHECK: main.exit.selector:
; CHECK: [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %inc, %for.inc ]
; CHECK: [[COND:%[^ ]+]] = icmp slt i32 [[PSEUDO_PHI]], %N
; CHECK: br i1 [[COND]], label %main.pseudo.exit, label %for.cond.cleanup.loopexit
define void @test_inc_ne(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %cmp16 = icmp sgt i32 %N, 0
  br i1 %cmp16, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.017 = phi i32 [ %inc, %for.inc ], [ 0, %entry ]
  %cmp1 = icmp ult i32 %i.017, 512
  %arrayidx = getelementptr inbounds i32, i32* %b, i32 %i.017
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %c, i32 %i.017
  %1 = load i32, i32* %arrayidx2, align 4
  br i1 %cmp1, label %if.then, label %if.else

if.then:
  %sub = sub i32 %0, %1
  %arrayidx3 = getelementptr inbounds i32, i32* %a, i32 %i.017
  %2 = load i32, i32* %arrayidx3, align 4
  %add = add nsw i32 %sub, %2
  store i32 %add, i32* %arrayidx3, align 4
  br label %for.inc

if.else:
  %add6 = add nsw i32 %1, %0
  %arrayidx7 = getelementptr inbounds i32, i32* %a, i32 %i.017
  store i32 %add6, i32* %arrayidx7, align 4
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %i.017, 1
  %exitcond = icmp ne i32 %inc, %N
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

; CHECK-LABEL: test_inc_slt(
; CHECK: main.exit.selector:
; CHECK: [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %inc, %for.inc ]
; CHECK: [[COND:%[^ ]+]] = icmp slt i32 [[PSEUDO_PHI]], %N
; CHECK: br i1 [[COND]], label %main.pseudo.exit, label %for.cond.cleanup.loopexit
define void @test_inc_slt(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %cmp16 = icmp sgt i32 %N, 0
  br i1 %cmp16, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.017 = phi i32 [ %inc, %for.inc ], [ 0, %entry ]
  %cmp1 = icmp ult i32 %i.017, 512
  %arrayidx = getelementptr inbounds i32, i32* %b, i32 %i.017
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %c, i32 %i.017
  %1 = load i32, i32* %arrayidx2, align 4
  br i1 %cmp1, label %if.then, label %if.else

if.then:
  %sub = sub i32 %0, %1
  %arrayidx3 = getelementptr inbounds i32, i32* %a, i32 %i.017
  %2 = load i32, i32* %arrayidx3, align 4
  %add = add nsw i32 %sub, %2
  store i32 %add, i32* %arrayidx3, align 4
  br label %for.inc

if.else:
  %add6 = add nsw i32 %1, %0
  %arrayidx7 = getelementptr inbounds i32, i32* %a, i32 %i.017
  store i32 %add6, i32* %arrayidx7, align 4
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %i.017, 1
  %exitcond = icmp slt i32 %inc, %N
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

; CHECK-LABEL: test_inc_ult
; CHECK: main.exit.selector:
; CHECK: [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %inc, %for.inc ]
; CHECK: [[COND:%[^ ]+]] = icmp ult i32 [[PSEUDO_PHI]], %N
; CHECK: br i1 [[COND]], label %main.pseudo.exit, label %for.cond.cleanup.loopexit
define void @test_inc_ult(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %cmp16 = icmp ugt i32 %N, 0
  br i1 %cmp16, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.017 = phi i32 [ %inc, %for.inc ], [ 0, %entry ]
  %cmp1 = icmp ult i32 %i.017, 512
  %arrayidx = getelementptr inbounds i32, i32* %b, i32 %i.017
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %c, i32 %i.017
  %1 = load i32, i32* %arrayidx2, align 4
  br i1 %cmp1, label %if.then, label %if.else

if.then:
  %sub = sub i32 %0, %1
  %arrayidx3 = getelementptr inbounds i32, i32* %a, i32 %i.017
  %2 = load i32, i32* %arrayidx3, align 4
  %add = add nsw i32 %sub, %2
  store i32 %add, i32* %arrayidx3, align 4
  br label %for.inc

if.else:
  %add6 = add nsw i32 %1, %0
  %arrayidx7 = getelementptr inbounds i32, i32* %a, i32 %i.017
  store i32 %add6, i32* %arrayidx7, align 4
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %i.017, 1
  %exitcond = icmp ult i32 %inc, %N
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}
