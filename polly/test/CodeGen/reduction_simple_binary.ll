; RUN: opt %loadNPMPolly '-passes=print<polly-ast>' -polly-ast-detect-parallel -disable-output < %s | FileCheck %s
;
; CHECK: pragma simd reduction
;
; int prod;
; void f() {
;   for (int i = 0; i < 100; i++)
;     prod *= i;
; }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

@prod = common global i32 0, align 4

define void @f() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i1.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i1.0, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add2 = add nsw i32 %i1.0, 3
  %tmp1 = load i32, ptr @prod, align 4
  %mul3 = mul nsw i32 %tmp1, %add2
  store i32 %mul3, ptr @prod, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i1.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
