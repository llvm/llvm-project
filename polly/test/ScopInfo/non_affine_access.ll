; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-scops -polly-delinearize -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-scops -polly-allow-nonaffine -analyze < %s | FileCheck %s -check-prefix=NONAFFINE
; RUN: opt %loadPolly -polly-scops -polly-delinearize -polly-allow-nonaffine -analyze < %s | FileCheck %s -check-prefix=NONAFFINE
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; void foo(long *A) {
;   for (i = 0; i < 1024; i++)
;     A[i * i] = i;
;

define void @foo(i64 *%A) nounwind uwtable ssp {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.body
  %indvar = phi i64 [ 0, %entry.split ], [ %indvar.next, %for.body ]
  %mul = mul nsw i64 %indvar, %indvar
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %mul
  store i64 %indvar, i64* %arrayidx, align 4
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

; CHECK-NOT: Stmt_for_body
; NONAFFINE: { Stmt_for_body[i0] -> MemRef_A[o0] : -1152921504606846976 <= o0 <= 1152921504606846975 };
