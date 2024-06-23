; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb '-passes=print<polly-function-scops>' -disable-output \
; RUN: -polly-invariant-load-hoisting=true < %s 2>&1 | FileCheck %s
;
; See http://research.microsoft.com/pubs/151917/divmodnote-letter.pdf
;
;    void f(long *A) {
;      for (long i = 0; i < 10; i++) {
;        A[8 / 3] = A[8 % 3];
;        A[8 / -3] = A[8 % -3];
;        A[-8 / 3] = A[-8 % 3];
;        A[-8 / -3] = A[-8 % -3];
;        A[1 / 2] = A[1 % 2];
;        A[1 / -2] = A[1 % -2];
;        A[-1 / 2] = A[-1 % 2];
;        A[-1 / -2] = A[-1 % -2];
;      }
;    }
;
; CHECK:   { Stmt_for_body[i0] -> MemRef_R[2] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_R[2] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_R[-2] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_R[-2] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_R[1] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_R[1] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_R[-1] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_R[-1] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_D[2] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_D[-2] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_D[-2] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_D[2] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_D[0] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_D[0] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_D[0] };
; CHECK:   { Stmt_for_body[i0] -> MemRef_D[0] };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %D, ptr %R) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i64 %i.0, 10
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %rem = srem i64 8, 3
  %arrayidx = getelementptr inbounds i64, ptr %R, i64 %rem
  %tmp = load i64, ptr %arrayidx, align 8
  %div = sdiv i64 8, 3
  %arrayidx1 = getelementptr inbounds i64, ptr %D, i64 %div
  store i64 %tmp, ptr %arrayidx1, align 8
  %rem2 = srem i64 8, -3
  %arrayidx3 = getelementptr inbounds i64, ptr %R, i64 %rem2
  %tmp1 = load i64, ptr %arrayidx3, align 8
  %div5 = sdiv i64 8, -3
  %arrayidx6 = getelementptr inbounds i64, ptr %D, i64 %div5
  store i64 %tmp1, ptr %arrayidx6, align 8
  %rem8 = srem i64 -8, 3
  %arrayidx9 = getelementptr inbounds i64, ptr %R, i64 %rem8
  %tmp2 = load i64, ptr %arrayidx9, align 8
  %div11 = sdiv i64 -8, 3
  %arrayidx12 = getelementptr inbounds i64, ptr %D, i64 %div11
  store i64 %tmp2, ptr %arrayidx12, align 8
  %rem15 = srem i64 -8, -3
  %arrayidx16 = getelementptr inbounds i64, ptr %R, i64 %rem15
  %tmp3 = load i64, ptr %arrayidx16, align 8
  %div19 = sdiv i64 -8, -3
  %arrayidx20 = getelementptr inbounds i64, ptr %D, i64 %div19
  store i64 %tmp3, ptr %arrayidx20, align 8
  %rem29 = srem i64 1, 2
  %arrayidx30 = getelementptr inbounds i64, ptr %R, i64 %rem29
  %tmp5 = load i64, ptr %arrayidx30, align 8
  %div31 = sdiv i64 1, 2
  %arrayidx32 = getelementptr inbounds i64, ptr %D, i64 %div31
  store i64 %tmp5, ptr %arrayidx32, align 8
  %rem34 = srem i64 1, -2
  %arrayidx35 = getelementptr inbounds i64, ptr %R, i64 %rem34
  %tmp6 = load i64, ptr %arrayidx35, align 8
  %div37 = sdiv i64 1, -2
  %arrayidx38 = getelementptr inbounds i64, ptr %D, i64 %div37
  store i64 %tmp6, ptr %arrayidx38, align 8
  %rem40 = srem i64 -1, 2
  %arrayidx41 = getelementptr inbounds i64, ptr %R, i64 %rem40
  %tmp7 = load i64, ptr %arrayidx41, align 8
  %div43 = sdiv i64 -1, 2
  %arrayidx44 = getelementptr inbounds i64, ptr %D, i64 %div43
  store i64 %tmp7, ptr %arrayidx44, align 8
  %rem47 = srem i64 -1, -2
  %arrayidx48 = getelementptr inbounds i64, ptr %R, i64 %rem47
  %tmp8 = load i64, ptr %arrayidx48, align 8
  %div51 = sdiv i64 -1, -2
  %arrayidx52 = getelementptr inbounds i64, ptr %D, i64 %div51
  store i64 %tmp8, ptr %arrayidx52, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
