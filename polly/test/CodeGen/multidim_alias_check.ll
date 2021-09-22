; RUN: opt %loadPolly -polly-codegen < %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: %polly.access.sext.A = sext i32 %n to i64
; CHECK: %polly.access.mul.A = mul i64 %polly.access.sext.A, %0
; CHECK: %polly.access.add.A = add i64 %polly.access.mul.A, 1
; CHECK: %polly.access.A = getelementptr double, double* %A, i64 %polly.access.add.A
; CHECK: %polly.access.y = getelementptr double, double* %y, i64 0
; CHECK: icmp ule double* %polly.access.A, %polly.access.y


define void @init_array(i32 %n, double* %A, double* %y) {
entry:
  %add3 = add nsw i32 %n, 1
  %tmp = zext i32 %add3 to i64
  br label %for.body

for.body:
  %i.04 = phi i32 [ %inc39, %for.cond.loopexit ], [ 0, %entry ]
  %arrayidx16 = getelementptr inbounds double, double* %y, i64 0
  store double 1.0, double* %arrayidx16
  %cmp251 = icmp slt i32 %n, 0
  %inc39 = add nsw i32 %i.04, 1
  br i1 %cmp251, label %for.cond.loopexit, label %for.body27

for.body27:
  %idxprom35 = sext i32 %i.04 to i64
  %tmp1 = mul nsw i64 %idxprom35, %tmp
  %arrayidx36.sum = add i64 0, %tmp1
  %arrayidx37 = getelementptr inbounds double, double* %A, i64 %arrayidx36.sum
  store double 1.0, double* %arrayidx37
  br label %for.cond.loopexit

for.cond.loopexit:
  %cmp = icmp slt i32 %i.04, %n
  br i1 %cmp, label %for.body, label %for.end40


for.end40:
  ret void
}
