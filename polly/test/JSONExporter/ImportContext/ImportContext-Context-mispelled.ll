; RUN: not --crash opt %loadPolly -polly-import-jscop -polly-ast -polly-ast-detect-parallel -disable-output < %s 2>&1 >/dev/null | FileCheck %s
;
; CHECK: JScop file has no key named 'context'.
;
; Verify if the JSONImporter check if there is a key name 'context'.
;
;    void ic(int *A, long n) {
;      for (long i = 0; i < 2 * n; i++)
; S0:    A[0] += i;
;      for (long i = 0; i < 2 * n; i++)
; S1:    A[i + 1] = 1;
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @ic(i32* %A, i32 %n) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %mul = shl nsw i32 %n, 1
  %cmp = icmp slt i32 %i.0, %mul
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br label %S0

S0:                                               ; preds = %for.body
  %tmp = load i32, i32* %A, align 4
  %add = add nsw i32 %tmp, %i.0
  store i32 %add, i32* %A, align 4
  br label %for.inc

for.inc:                                          ; preds = %S0
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc8, %for.end
  %i1.0 = phi i32 [ 0, %for.end ], [ %inc9, %for.inc8 ]
  %mul3 = shl nsw i32 %n, 1
  %cmp4 = icmp slt i32 %i1.0, %mul3
  br i1 %cmp4, label %for.body5, label %for.end10

for.body5:                                        ; preds = %for.cond2
  br label %S1

S1:                                               ; preds = %for.body5
  %add6 = add nsw i32 %i1.0, 1
  %arrayidx7 = getelementptr inbounds i32, i32* %A, i32 %add6
  store i32 1, i32* %arrayidx7, align 4
  br label %for.inc8

for.inc8:                                         ; preds = %S1
  %inc9 = add nsw i32 %i1.0, 1
  br label %for.cond2

for.end10:                                        ; preds = %for.cond2
  ret void
}

