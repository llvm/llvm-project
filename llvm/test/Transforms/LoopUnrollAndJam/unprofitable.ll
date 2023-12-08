; RUN: opt -passes=loop-unroll-and-jam -allow-unroll-and-jam -pass-remarks=loop-unroll < %s -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8m.main-arm-none-eabi"

;; Common check for all tests. None should be unroll and jammed due to profitability
; CHECK-NOT: remark: {{.*}} unroll and jammed


; CHECK-LABEL: unprof1
; Multiple inner loop blocks
define void @unprof1(i32 %I, i32 %J, ptr noalias nocapture %A, ptr noalias nocapture readonly %B) #0 {
; CHECK: %i = phi i32 [ %addinc, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner2 ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %addinc, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner2 ]
  %sum1 = phi i32 [ 0, %for.outer ], [ %add, %for.inner2 ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx, align 4
  %add = add i32 %0, %sum1
br label %for.inner2

for.inner2:
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner2 ]
  %arrayidx6 = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 %add.lcssa, ptr %arrayidx6, align 4
  %addinc = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %addinc, %I
  br i1 %exitcond25, label %for.loopexit, label %for.outer

for.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: unprof2
; Constant inner loop count
define void @unprof2(i32 %I, i32 %J, ptr noalias nocapture %A, ptr noalias nocapture readonly %B) #0 {
; CHECK: %i = phi i32 [ %addinc, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %addinc, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx, align 4
  %add = add i32 %0, %sum1
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 %add.lcssa, ptr %arrayidx6, align 4
  %addinc = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %addinc, %I
  br i1 %exitcond25, label %for.loopexit, label %for.outer

for.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: unprof3
; Complex inner loop
define void @unprof3(i32 %I, i32 %J, ptr noalias nocapture %A, ptr noalias nocapture readonly %B) #0 {
; CHECK: %i = phi i32 [ %addinc, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %addinc, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx, align 4
  %add = add i32 %0, %sum1
  %add0 = add i32 %0, %sum1
  %add1 = add i32 %0, %sum1
  %add2 = add i32 %0, %sum1
  %add3 = add i32 %0, %sum1
  %add4 = add i32 %0, %sum1
  %add5 = add i32 %0, %sum1
  %add6 = add i32 %0, %sum1
  %add7 = add i32 %0, %sum1
  %add8 = add i32 %0, %sum1
  %add9 = add i32 %0, %sum1
  %add10 = add i32 %0, %sum1
  %add11 = add i32 %0, %sum1
  %add12 = add i32 %0, %sum1
  %add13 = add i32 %0, %sum1
  %add14 = add i32 %0, %sum1
  %add15 = add i32 %0, %sum1
  %add16 = add i32 %0, %sum1
  %add17 = add i32 %0, %sum1
  %add18 = add i32 %0, %sum1
  %add19 = add i32 %0, %sum1
  %add20 = add i32 %0, %sum1
  %add21 = add i32 %0, %sum1
  %add22 = add i32 %0, %sum1
  %add23 = add i32 %0, %sum1
  %add24 = add i32 %0, %sum1
  %add25 = add i32 %0, %sum1
  %add26 = add i32 %0, %sum1
  %add27 = add i32 %0, %sum1
  %add28 = add i32 %0, %sum1
  %add29 = add i32 %0, %sum1
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 %add.lcssa, ptr %arrayidx6, align 4
  %addinc = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %addinc, %I
  br i1 %exitcond25, label %for.loopexit, label %for.outer

for.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: unprof4
; No loop invariant loads
define void @unprof4(i32 %I, i32 %J, ptr noalias nocapture %A, ptr noalias nocapture readonly %B) #0 {
; CHECK: %i = phi i32 [ %addinc, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %addinc, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %j2 = add i32 %j, %i
  %arrayidx = getelementptr inbounds i32, ptr %B, i32 %j2
  %0 = load i32, ptr %arrayidx, align 4
  %add = add i32 %0, %sum1
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 %add.lcssa, ptr %arrayidx6, align 4
  %addinc = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %addinc, %I
  br i1 %exitcond25, label %for.loopexit, label %for.outer

for.loopexit:
  br label %for.end

for.end:
  ret void
}
