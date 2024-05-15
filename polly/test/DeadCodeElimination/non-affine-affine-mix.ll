; RUN: opt %loadPolly -polly-allow-nonaffine -polly-dce -polly-print-ast -disable-output < %s | FileCheck %s
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++)
; S1:    A[bar(i)] = i;
;      for (int i = 0; i < 1024; i++)
; S2:    A[i2] = i;
;    }

; We unfortunately do need to execute all iterations of S1, as we do not know
; the size of A and as a result S1 may write for example to A[1024], which
; is not overwritten by S2.

; CHECK: for (int c0 = 0; c0 <= 1023; c0 += 1)
; CHECK:   Stmt_S1(c0);
; CHECK: for (int c0 = 0; c0 <= 1023; c0 += 1)
; CHECK:   Stmt_S2(c0);

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

declare i32 @bar(i32) #1

define void @f(ptr %A) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %S1, label %next

S1:
  %nonaff = call i32 @bar(i32 %i.0)
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %nonaff
  store i32 %i.0, ptr %arrayidx, align 4
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

next:
 br label %for.cond.2

for.cond.2:
  %i.2 = phi i32 [ 0, %next ], [ %inc.2, %for.inc.2 ]
  %exitcond.2 = icmp ne i32 %i.2, 1024
  br i1 %exitcond.2, label %S2, label %for.end

S2:
  %arrayidx.2 = getelementptr inbounds i32, ptr %A, i32 %i.2
  store i32 %i.2, ptr %arrayidx.2, align 4
  br label %for.inc.2

for.inc.2:
  %inc.2 = add nsw i32 %i.2, 1
  br label %for.cond.2

for.end:
  ret void
}

attributes #1 = { nounwind readnone }

