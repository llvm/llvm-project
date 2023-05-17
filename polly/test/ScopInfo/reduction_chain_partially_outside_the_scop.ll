; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK: Reduction Type: NONE
;
;    int c, d;
;    void f(int *sum) {
;      for (int i = 0; i < 1024; i++)
;        *sum = c + d;
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

@c = common global i32 0, align 4
@d = common global i32 0, align 4

define void @loads_outside_scop(ptr %sum) {
entry:
  %tmp = load i32, ptr @c, align 4
  %tmp1 = load i32, ptr @d, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add = add nsw i32 %tmp, %tmp1
  store i32 %add, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}


define void @binop_outside_scop(ptr %sum) {
entry:
  %tmp = load i32, ptr @c, align 4
  %tmp1 = load i32, ptr @d, align 4
  %add = add nsw i32 %tmp, %tmp1
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 %add, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
