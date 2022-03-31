; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; The constraint that r2 has to be bigger than r1 is implicitly containted in
; the domain, hence we do not want to see it explicitly.
;
; CHECK-NOT:  r2 >= 1 + r1
;
;    void wraps(int *A, int p, short q, char r1, char r2) {
;      for (char i = r1; i < r2; i++)
;        A[p + q] = A[(int)r1 + (int)r2];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @wraps(i32* %A, i32 %p, i16 signext %q, i8 signext %r1, i8 signext %r2) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i8 [ %r1, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i8 %i.0, %r2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %conv3 = sext i8 %r1 to i64
  %conv4 = sext i8 %r2 to i64
  %add = add nsw i64 %conv3, %conv4
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %add
  %tmp = load i32, i32* %arrayidx, align 4
  %conv5 = sext i16 %q to i32
  %add6 = add nsw i32 %conv5, %p
  %idxprom7 = sext i32 %add6 to i64
  %arrayidx8 = getelementptr inbounds i32, i32* %A, i64 %idxprom7
  store i32 %tmp, i32* %arrayidx8, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add i8 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
