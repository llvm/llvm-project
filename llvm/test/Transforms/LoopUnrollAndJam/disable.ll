; RUN: opt -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-and-jam-count=4 -pass-remarks=loop-unroll-and-jam < %s -S 2>&1 | FileCheck %s
; RUN: opt -passes='loop-unroll-and-jam' -allow-unroll-and-jam -unroll-and-jam-count=4 -pass-remarks=loop-unroll-and-jam < %s -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

;; Common check for all tests. None should be unroll and jammed
; CHECK-NOT: remark: {{.*}} unroll and jammed


; CHECK-LABEL: disabled1
; Tests for(i) { sum = A[i]; for(j) sum += B[j]; A[i+1] = sum; }
; A[i] to A[i+1] dependency should block unrollandjam
define void @disabled1(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i.029 = phi i32 [ %add10, %for.latch ], [ 0, %for.preheader ]
; CHECK: %j.026 = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp127 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp127, %cmp
  br i1 %or.cond, label %for.preheader, label %return

for.preheader:
  br label %for.outer

for.outer:
  %i.029 = phi i32 [ %add10, %for.latch ], [ 0, %for.preheader ]
  %b.028 = phi i32 [ %inc8, %for.latch ], [ 1, %for.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.029
  %0 = load i32, i32* %arrayidx, align 4
  br label %for.inner

for.inner:
  %j.026 = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1.025 = phi i32 [ %0, %for.outer ], [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, i32* %B, i32 %j.026
  %1 = load i32, i32* %arrayidx6, align 4
  %add = add i32 %1, %sum1.025
  %inc = add nuw i32 %j.026, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %arrayidx7 = getelementptr inbounds i32, i32* %A, i32 %b.028
  store i32 %add, i32* %arrayidx7, align 4
  %inc8 = add nuw nsw i32 %b.028, 1
  %add10 = add nuw nsw i32 %i.029, 1
  %exitcond30 = icmp eq i32 %add10, %I
  br i1 %exitcond30, label %return, label %for.outer

return:
  ret void
}


; CHECK-LABEL: disabled2
; Tests an incompatible block layout (for.outer jumps past for.inner)
; FIXME: Make this work
define void @disabled2(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i.032 = phi i32 [ %add13, %for.latch ], [ 0, %for.preheader ]
; CHECK: %j.030 = phi i32 [ %inc, %for.inner ], [ 0, %for.inner.preheader ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp131 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp131, %cmp
  br i1 %or.cond, label %for.preheader, label %for.end14

for.preheader:
  br label %for.outer

for.outer:
  %i.032 = phi i32 [ %add13, %for.latch ], [ 0, %for.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.032
  %0 = load i32, i32* %arrayidx, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %for.latch, label %for.inner

for.inner:
  %j.030 = phi i32 [ %inc, %for.inner ], [ 0, %for.outer ]
  %sum1.029 = phi i32 [ %sum1.1, %for.inner ], [ 0, %for.outer ]
  %arrayidx6 = getelementptr inbounds i32, i32* %B, i32 %j.030
  %1 = load i32, i32* %arrayidx6, align 4
  %tobool7 = icmp eq i32 %1, 0
  %sub = add i32 %sum1.029, 10
  %add = sub i32 %sub, %1
  %sum1.1 = select i1 %tobool7, i32 %sum1.029, i32 %add
  %inc = add nuw i32 %j.030, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %sum1.1.lcssa = phi i32 [ 0, %for.outer ], [ %sum1.1, %for.inner ]
  %arrayidx11 = getelementptr inbounds i32, i32* %A, i32 %i.032
  store i32 %sum1.1.lcssa, i32* %arrayidx11, align 4
  %add13 = add nuw i32 %i.032, 1
  %exitcond33 = icmp eq i32 %add13, %I
  br i1 %exitcond33, label %for.end14, label %for.outer

for.end14:
  ret void
}


; CHECK-LABEL: disabled3
; Tests loop carry dependencies in an array S
define void @disabled3(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i.029 = phi i32 [ 0, %for.preheader ], [ %add12, %for.latch ]
; CHECK: %j.027 = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
entry:
  %S = alloca [4 x i32], align 4
  %cmp = icmp eq i32 %J, 0
  br i1 %cmp, label %return, label %if.end

if.end:
  %0 = bitcast [4 x i32]* %S to i8*
  %cmp128 = icmp eq i32 %I, 0
  br i1 %cmp128, label %for.cond.cleanup, label %for.preheader

for.preheader:
  %arrayidx9 = getelementptr inbounds [4 x i32], [4 x i32]* %S, i32 0, i32 0
  br label %for.outer

for.cond.cleanup:
  br label %return

for.outer:
  %i.029 = phi i32 [ 0, %for.preheader ], [ %add12, %for.latch ]
  br label %for.inner

for.inner:
  %j.027 = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j.027
  %l2 = load i32, i32* %arrayidx, align 4
  %add = add i32 %j.027, %i.029
  %rem = urem i32 %add, %J
  %arrayidx6 = getelementptr inbounds i32, i32* %B, i32 %rem
  %l3 = load i32, i32* %arrayidx6, align 4
  %mul = mul i32 %l3, %l2
  %rem7 = urem i32 %j.027, 3
  %arrayidx8 = getelementptr inbounds [4 x i32], [4 x i32]* %S, i32 0, i32 %rem7
  store i32 %mul, i32* %arrayidx8, align 4
  %inc = add nuw i32 %j.027, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %l1 = load i32, i32* %arrayidx9, align 4
  %arrayidx10 = getelementptr inbounds i32, i32* %A, i32 %i.029
  store i32 %l1, i32* %arrayidx10, align 4
  %add12 = add nuw i32 %i.029, 1
  %exitcond31 = icmp eq i32 %add12, %I
  br i1 %exitcond31, label %for.cond.cleanup, label %for.outer

return:
  ret void
}


; CHECK-LABEL: disabled4
; Inner looop induction variable is not consistent
; ie for(i = 0..n) for (j = 0..i) sum+=B[j]
define void @disabled4(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %indvars.iv = phi i32 [ %indvars.iv.next, %for.latch ], [ 1, %for.preheader ]
; CHECK: %j.021 = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ugt i32 %I, 1
  %or.cond = and i1 %cmp122, %cmp
  br i1 %or.cond, label %for.preheader, label %for.end9

for.preheader:
  br label %for.outer

for.outer:
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.latch ], [ 1, %for.preheader ]
  br label %for.inner

for.inner:
  %j.021 = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1.020 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j.021
  %0 = load i32, i32* %arrayidx, align 4
  %add = add i32 %0, %sum1.020
  %inc = add nuw i32 %j.021, 1
  %exitcond = icmp eq i32 %inc, %indvars.iv
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %indvars.iv
  store i32 %add, i32* %arrayidx6, align 4
  %indvars.iv.next = add nuw i32 %indvars.iv, 1
  %exitcond24 = icmp eq i32 %indvars.iv.next, %I
  br i1 %exitcond24, label %for.end9, label %for.outer

for.end9:
  ret void
}


; CHECK-LABEL: disabled5
; Test odd uses of phi nodes where the outer IV cannot be moved into Fore as it hits a PHI
@f = hidden global i32 0, align 4
define i32 @disabled5() #0 {
; CHECK: %0 = phi i32 [ %f.promoted10, %entry ], [ 2, %for.latch ]
; CHECK: %1 = phi i32 [ %0, %for.outer ], [ 2, %for.inner ]
entry:
  %f.promoted10 = load i32, i32* @f, align 4
  br label %for.outer

for.outer:
  %0 = phi i32 [ %f.promoted10, %entry ], [ 2, %for.latch ]
  %d.018 = phi i16 [ 0, %entry ], [ %odd.lcssa, %for.latch ]
  %inc5.sink9 = phi i32 [ 2, %entry ], [ %inc5, %for.latch ]
  br label %for.inner

for.inner:
  %1 = phi i32 [ %0, %for.outer ], [ 2, %for.inner ]
  %inc.sink8 = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %inc = add nuw nsw i32 %inc.sink8, 1
  %exitcond = icmp ne i32 %inc, 7
  br i1 %exitcond, label %for.inner, label %for.latch

for.latch:
  %.lcssa = phi i32 [ %1, %for.inner ]
  %odd.lcssa = phi i16 [ 1, %for.inner ]
  %inc5 = add nuw nsw i32 %inc5.sink9, 1
  %exitcond11 = icmp ne i32 %inc5, 7
  br i1 %exitcond11, label %for.outer, label %for.end

for.end:
  %.lcssa.lcssa = phi i32 [ %.lcssa, %for.latch ]
  %inc.lcssa.lcssa = phi i32 [ 7, %for.latch ]
  ret i32 0
}


; CHECK-LABEL: disabled6
; There is a dependency in here, between @d and %0 (=@f)
@d6 = hidden global i16 5, align 2
@f6 = hidden global i16* @d6, align 4
define i32 @disabled6() #0 {
; CHECK: %inc8.sink14.i = phi i16 [ 1, %entry ], [ %inc8.i, %for.cond.cleanup.i ]
; CHECK: %c.013.i = phi i32 [ 0, %for.body.i ], [ %inc.i, %for.body6.i ]
entry:
  store i16 1, i16* @d6, align 2
  %0 = load i16*, i16** @f6, align 4
  br label %for.body.i

for.body.i:
  %inc8.sink14.i = phi i16 [ 1, %entry ], [ %inc8.i, %for.cond.cleanup.i ]
  %1 = load i16, i16* %0, align 2
  br label %for.body6.i

for.cond.cleanup.i:
  %inc8.i = add nuw nsw i16 %inc8.sink14.i, 1
  store i16 %inc8.i, i16* @d6, align 2
  %cmp.i = icmp ult i16 %inc8.i, 6
  br i1 %cmp.i, label %for.body.i, label %test.exit

for.body6.i:
  %c.013.i = phi i32 [ 0, %for.body.i ], [ %inc.i, %for.body6.i ]
  %inc.i = add nuw nsw i32 %c.013.i, 1
  %exitcond.i = icmp eq i32 %inc.i, 7
  br i1 %exitcond.i, label %for.cond.cleanup.i, label %for.body6.i

test.exit:
  %conv2.i = sext i16 %1 to i32
  ret i32 0
}


; CHECK-LABEL: disabled7
; Has negative output dependency
define void @disabled7(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i.028 = phi i32 [ %add11, %for.cond3.for.cond.cleanup5_crit_edge ], [ 0, %for.body.preheader ]
; CHECK: %j.026 = phi i32 [ 0, %for.body ], [ %add9, %for.body6 ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp127 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp127, %cmp
  br i1 %or.cond, label %for.body.preheader, label %for.end12

for.body.preheader:
  br label %for.body

for.body:
  %i.028 = phi i32 [ %add11, %for.cond3.for.cond.cleanup5_crit_edge ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.028
  store i32 0, i32* %arrayidx, align 4
  %sub = add i32 %i.028, -1
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %sub
  store i32 2, i32* %arrayidx2, align 4
  br label %for.body6

for.cond3.for.cond.cleanup5_crit_edge:
  store i32 %add, i32* %arrayidx, align 4
  %add11 = add nuw i32 %i.028, 1
  %exitcond29 = icmp eq i32 %add11, %I
  br i1 %exitcond29, label %for.end12, label %for.body

for.body6:
  %0 = phi i32 [ 0, %for.body ], [ %add, %for.body6 ]
  %j.026 = phi i32 [ 0, %for.body ], [ %add9, %for.body6 ]
  %arrayidx7 = getelementptr inbounds i32, i32* %B, i32 %j.026
  %1 = load i32, i32* %arrayidx7, align 4
  %add = add i32 %1, %0
  %add9 = add nuw i32 %j.026, 1
  %exitcond = icmp eq i32 %add9, %J
  br i1 %exitcond, label %for.cond3.for.cond.cleanup5_crit_edge, label %for.body6

for.end12:
  ret void
}


; CHECK-LABEL: disabled8
; Same as above with an extra outer loop nest
define void @disabled8(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i.036 = phi i32 [ %add15, %for.latch ], [ 0, %for.body ]
; CHECK: %j.034 = phi i32 [ 0, %for.outer ], [ %add13, %for.inner ]
entry:
  %cmp = icmp eq i32 %J, 0
  %cmp335 = icmp eq i32 %I, 0
  %or.cond = or i1 %cmp, %cmp335
  br i1 %or.cond, label %for.end18, label %for.body.preheader

for.body.preheader:
  br label %for.body

for.body:
  %x.037 = phi i32 [ %inc, %for.cond.cleanup4 ], [ 0, %for.body.preheader ]
  br label %for.outer

for.cond.cleanup4:
  %inc = add nuw nsw i32 %x.037, 1
  %exitcond40 = icmp eq i32 %inc, 5
  br i1 %exitcond40, label %for.end18, label %for.body

for.outer:
  %i.036 = phi i32 [ %add15, %for.latch ], [ 0, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.036
  store i32 0, i32* %arrayidx, align 4
  %sub = add i32 %i.036, -1
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %sub
  store i32 2, i32* %arrayidx6, align 4
  br label %for.inner

for.latch:
  store i32 %add, i32* %arrayidx, align 4
  %add15 = add nuw i32 %i.036, 1
  %exitcond38 = icmp eq i32 %add15, %I
  br i1 %exitcond38, label %for.cond.cleanup4, label %for.outer

for.inner:
  %0 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %j.034 = phi i32 [ 0, %for.outer ], [ %add13, %for.inner ]
  %arrayidx11 = getelementptr inbounds i32, i32* %B, i32 %j.034
  %1 = load i32, i32* %arrayidx11, align 4
  %add = add i32 %1, %0
  %add13 = add nuw i32 %j.034, 1
  %exitcond = icmp eq i32 %add13, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.end18:
  ret void
}


; CHECK-LABEL: disabled9
; Can't prove alias between A and B
define void @disabled9(i32 %I, i32 %J, i32* nocapture %A, i32* nocapture readonly %B) #0 {
; CHECK: %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx, align 4
  %add = add i32 %0, %sum1
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 %add.lcssa, i32* %arrayidx6, align 4
  %add8 = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %add8, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: disable10
; Simple call
declare void @f10(i32, i32) #0
define void @disable10(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx, align 4
  %add = add i32 %0, %sum1
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %J
  tail call void @f10(i32 %i, i32 %j) nounwind
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 %add.lcssa, i32* %arrayidx6, align 4
  %add8 = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %add8, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: disable11
; volatile
define void @disable11(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load volatile i32, i32* %arrayidx, align 4
  %add = add i32 %0, %sum1
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 %add.lcssa, i32* %arrayidx6, align 4
  %add8 = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %add8, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: disable12
; Multiple aft blocks
define void @disable12(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i = phi i32 [ %add8, %for.latch3 ], [ 0, %for.outer.preheader ]
; CHECK: %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %add8, %for.latch3 ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx, align 4
  %add = add i32 %0, %sum1
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 %add.lcssa, i32* %arrayidx6, align 4
  %cmpl = icmp eq i32 %add.lcssa, 10
  br i1 %cmpl, label %for.latch2, label %for.latch3

for.latch2:
  br label %for.latch3

for.latch3:
  %add8 = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %add8, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: disable13
; Two subloops
define void @disable13(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
; CHECK: %j2 = phi i32 [ %inc2, %for.inner2 ], [ 0, %for.inner2.preheader ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx, align 4
  %add = add i32 %0, %sum1
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.inner2, label %for.inner

for.inner2:
  %j2 = phi i32 [ 0, %for.inner ], [ %inc2, %for.inner2 ]
  %sum12 = phi i32 [ 0, %for.inner ], [ %add2, %for.inner2 ]
  %arrayidx2 = getelementptr inbounds i32, i32* %B, i32 %j2
  %l0 = load i32, i32* %arrayidx2, align 4
  %add2 = add i32 %l0, %sum12
  %inc2 = add nuw i32 %j2, 1
  %exitcond2 = icmp eq i32 %inc2, %J
  br i1 %exitcond2, label %for.latch, label %for.inner2

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner2 ]
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 %add.lcssa, i32* %arrayidx6, align 4
  %add8 = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %add8, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: disable14
; Multiple exits blocks
define void @disable14(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j = phi i32 [ %inc, %for.inner ], [ 0, %for.inner.preheader ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
  %add8 = add nuw i32 %i, 1
  %exitcond23 = icmp eq i32 %add8, %I
  br i1 %exitcond23, label %for.end.loopexit, label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx, align 4
  %add = add i32 %0, %sum1
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 %add.lcssa, i32* %arrayidx6, align 4
  %exitcond25 = icmp eq i32 %add8, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: disable15
; Latch != exit
define void @disable15(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j = phi i32 [ %inc, %for.inner ], [ 0, %for.inner.preheader ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
  %add8 = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %add8, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx, align 4
  %add = add i32 %0, %sum1
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 %add.lcssa, i32* %arrayidx6, align 4
  br label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: disable16
; Cannot move other before inner loop
define void @disable16(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
  %otherphi = phi i32 [ %other, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx, align 4
  %add = add i32 %0, %sum1
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 %add.lcssa, i32* %arrayidx6, align 4
  %add8 = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %add8, %I
  %loadarr = getelementptr inbounds i32, i32* %A, i32 %i
  %load = load i32, i32* %arrayidx6, align 4
  %other = add i32 %otherphi, %load
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}
