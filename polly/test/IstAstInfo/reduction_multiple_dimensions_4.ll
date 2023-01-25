; RUN: opt %loadPolly -polly-print-ast -polly-ast-detect-parallel -disable-output < %s | FileCheck %s
;
; CHECK-NOT:#pragma known-parallel reduction
; CHECK:    #pragma known-parallel
; CHECK:    for (int c0 = 0; c0 <= 2047; c0 += 1)
; CHECK:      for (int c1 = 0; c1 <= 1023; c1 += 1)
; CHECK:        #pragma simd reduction
; CHECK:        for (int c2 = 0; c2 <= 511; c2 += 1)
; CHECK:          Stmt_for_body6(c0, c1, c2);
;
;    void rmd4(int *A) {
;      for (long i = 0; i < 2048; i++)
;        for (long j = 0; j < 1024; j++)
;          for (long k = 0; k < 512; k++)
;            A[i] += i;
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @rmd4(ptr %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc10, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc11, %for.inc10 ]
  %exitcond2 = icmp ne i32 %i.0, 2048
  br i1 %exitcond2, label %for.body, label %for.end12

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc7, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc8, %for.inc7 ]
  %exitcond1 = icmp ne i32 %j.0, 1024
  br i1 %exitcond1, label %for.body3, label %for.end9

for.body3:                                        ; preds = %for.cond1
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %k.0 = phi i32 [ 0, %for.body3 ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %k.0, 512
  br i1 %exitcond, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i.0
  %tmp = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %tmp, %i.0
  store i32 %add, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %inc = add nsw i32 %k.0, 1
  br label %for.cond4

for.end:                                          ; preds = %for.cond4
  br label %for.inc7

for.inc7:                                         ; preds = %for.end
  %inc8 = add nsw i32 %j.0, 1
  br label %for.cond1

for.end9:                                         ; preds = %for.cond1
  br label %for.inc10

for.inc10:                                        ; preds = %for.end9
  %inc11 = add nsw i32 %i.0, 1
  br label %for.cond

for.end12:                                        ; preds = %for.cond
  ret void
}
