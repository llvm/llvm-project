; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;    void foo(long n, float A[100]) {
;      for (long j = 0; j < n; j++) {
;        for (long i = j; i < n; i++) {
;          if (i < 0)
;            goto end;
;
;          if (i >= 100)
;            goto end;
;
;          A[i] += i;
;        }
;      }
;    end:
;      return;
;    }
;
; CHECK: Domain :=
; CHECK:  [n] -> { Stmt_if_end_7[i0, i1] : i0 >= 0 and 0 <= i1 <= 99 - i0 and i1 < n - i0 };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i64 %n, float* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc.8, %entry
  %j.0 = phi i64 [ 0, %entry ], [ %inc9, %for.inc.8 ]
  %cmp = icmp slt i64 %j.0, %n
  br i1 %cmp, label %for.body, label %for.end.10

for.body:                                         ; preds = %for.cond
  br label %for.cond.1

for.cond.1:                                       ; preds = %for.inc, %for.body
  %i.0 = phi i64 [ %j.0, %for.body ], [ %inc, %for.inc ]
  %cmp2 = icmp slt i64 %i.0, %n
  br i1 %cmp2, label %for.body.3, label %for.end

for.body.3:                                       ; preds = %for.cond.1
  br i1 false, label %if.then, label %if.end

if.then:                                          ; preds = %for.body.3
  br label %end

if.end:                                           ; preds = %for.body.3
  %cmp5 = icmp sgt i64 %i.0, 99
  br i1 %cmp5, label %if.then.6, label %if.end.7

if.then.6:                                        ; preds = %if.end
  br label %end

if.end.7:                                         ; preds = %if.end
  %conv = sitofp i64 %i.0 to float
  %arrayidx = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp = load float, float* %arrayidx, align 4
  %add = fadd float %tmp, %conv
  store float %add, float* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end.7
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond.1

for.end:                                          ; preds = %for.cond.1
  br label %for.inc.8

for.inc.8:                                        ; preds = %for.end
  %inc9 = add nuw nsw i64 %j.0, 1
  br label %for.cond

for.end.10:                                       ; preds = %for.cond
  br label %end

end:                                              ; preds = %for.end.10, %if.then.6, %if.then
  ret void
}
