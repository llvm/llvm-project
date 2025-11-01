; RUN: opt -S -loop-reduce -mcpu=corei7-avx -mtriple=x86_64-apple-macosx < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @indvar_expansion(ptr nocapture readonly %rowsptr, i1 %arg) {
entry:
  br label %for.cond

; SCEVExpander used to create induction variables in the loop %for.cond while
; expanding the recurrence start value of loop strength reduced values from
; %vector.body.

; CHECK-LABEL: indvar_expansion
; CHECK: for.cond:
; CHECK-NOT: phi i3
; CHECK: br i1 {{.+}}, label %for.cond

for.cond:
  %indvars.iv44 = phi i64 [ %indvars.iv.next45, %for.cond ], [ 0, %entry ]
  %cmp = icmp eq i8 undef, 0
  %indvars.iv.next45 = add nuw nsw i64 %indvars.iv44, 1
  br i1 %cmp, label %for.cond, label %for.cond2

for.cond2:
  br i1 %arg, label %for.cond2, label %for.body14.lr.ph

for.body14.lr.ph:
  %sext = shl i64 %indvars.iv44, 32
  %0 = ashr exact i64 %sext, 32
  %1 = sub i64 undef, %indvars.iv44
  %2 = and i64 %1, 4294967295
  %3 = add i64 %2, 1
  %fold = add i64 %1, 1
  %n.mod.vf = and i64 %fold, 7
  %n.vec = sub i64 %3, %n.mod.vf
  %end.idx.rnd.down = add i64 %n.vec, %0
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ %0, %for.body14.lr.ph ]
  %4 = getelementptr inbounds i8, ptr %rowsptr, i64 %index
  %wide.load = load <4 x i8>, ptr %4, align 1
  %index.next = add i64 %index, 8
  %5 = icmp eq i64 %index.next, %end.idx.rnd.down
  br i1 %5, label %for.end24, label %vector.body

for.end24:
  ret void
}
