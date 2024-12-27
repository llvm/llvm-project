; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; int f() {
;   int i, sum = 0, sth = 0;
;   for (i = 0; i < 1024; i++) {
;     sum += 5;
;     sth = sth + sth * sth + sth;
;     sum *= 5;
;   }
;   return sum + sth;
; }
;
; CHECK-NOT: Reduction Type: +
; CHECK-NOT: Reduction Type: *
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define i32 @f() {
entry:
  %sum.0 = alloca i32
  %sth.0 = alloca i32
  br label %entry.split

entry.split:                                      ; preds = %entry
  store i32 0, ptr %sum.0
  store i32 0, ptr %sth.0
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry.split
  %i.0 = phi i32 [ 0, %entry.split ], [ %inc, %for.cond ]
  %sth.0.reload = load i32, ptr %sth.0
  %sum.0.reload = load i32, ptr %sum.0
  %exitcond = icmp ne i32 %i.0, 1024
  %mul = mul nsw i32 %sth.0.reload, %sth.0.reload
  %add1 = add nsw i32 %sth.0.reload, %mul
  %tmp = mul i32 %sum.0.reload, 5
  store i32 %tmp, ptr %sum.0
  %sum.1.reload = load i32, ptr %sum.0
  %mul3 = add i32 %sum.1.reload, 25
  %add2 = add nsw i32 %add1, %sth.0.reload
  %inc = add nsw i32 %i.0, 1
  store i32 %mul3, ptr %sum.0
  store i32 %add2, ptr %sth.0
  br i1 %exitcond, label %for.cond, label %for.end

for.end:                                          ; preds = %for.cond
  %sum.0.reload.2 = load i32, ptr %sum.0
  %sth.0.reload.2 = load i32, ptr %sth.0
  %add4 = add nsw i32 %sum.0.reload.2, %sth.0.reload.2
  ret i32 %add4
}
