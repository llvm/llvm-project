; RUN: opt %loadNPMPolly -passes=polly-codegen < %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: %polly.access.sext.A = sext i32 %n to i64
; CHECK: %polly.access.mul.A = mul i64 %polly.access.sext.A, %0
; CHECK: %polly.access.add.A = add i64 %polly.access.mul.A, 1
; CHECK: %polly.access.A = getelementptr double, ptr %A, i64 %polly.access.add.A
; CHECK: icmp ule ptr %polly.access.A, %y


define void @init_array(i32 %n, ptr %A, ptr %y) {
entry:
  %add3 = add nsw i32 %n, 1
  %tmp = zext i32 %add3 to i64
  br label %for.body

for.body:
  %i.04 = phi i32 [ %inc39, %for.cond.loopexit ], [ 0, %entry ]
  store double 1.0, ptr %y
  %cmp251 = icmp slt i32 %n, 0
  %inc39 = add nsw i32 %i.04, 1
  br i1 %cmp251, label %for.cond.loopexit, label %for.body27

for.body27:
  %idxprom35 = sext i32 %i.04 to i64
  %tmp1 = mul nsw i64 %idxprom35, %tmp
  %arrayidx36.sum = add i64 0, %tmp1
  %arrayidx37 = getelementptr inbounds double, ptr %A, i64 %arrayidx36.sum
  store double 1.0, ptr %arrayidx37
  br label %for.cond.loopexit

for.cond.loopexit:
  %cmp = icmp slt i32 %i.04, %n
  br i1 %cmp, label %for.body, label %for.end40


for.end40:
  ret void
}
