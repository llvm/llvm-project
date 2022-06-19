; RUN: opt %loadPolly -polly-print-scops \
; RUN: -polly-invariant-load-hoisting=true -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -S -polly-codegen \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s --check-prefix=IR
;
;    void f(long *A, long *B, long *ptr, long val) {
;      for (long i = 0; i < 100; i++) {
;        long ptrV = ((long)(ptr + 1)) + 1;
;        long valP = (long)(((long *)(val + 1)) + 1);
;        A[ptrV] += B[valP];
;      }
;    }
;
; CHECK:        ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       [val, ptr] -> { Stmt_for_body[i0] -> MemRef_B[9 + val] };
; CHECK-NEXT:   Execution Context: [val, ptr] -> {  : -4097 <= val <= 4086 }
;
; CHECK:   ReadAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:       [val, ptr] -> { Stmt_for_body[i0] -> MemRef_A[9 + ptr] };
; CHECK-NEXT:   MustWriteAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:       [val, ptr] -> { Stmt_for_body[i0] -> MemRef_A[9 + ptr] };

; IR:      entry:
; IR-NEXT:   %ptr13 = ptrtoint i64* %ptr to i16
;
; IR:      polly.stmt.for.body:
; IR-NEXT:   %tmp4_p_scalar_ = load i64, i64* %scevgep, align 8, !alias.scope !3, !noalias !0
; IR-NEXT:   %p_add4 = add nsw i64 %tmp4_p_scalar_, %polly.preload.tmp3.merge
; IR-NEXT:   store i64 %p_add4, i64* %scevgep, align 8, !alias.scope !3, !noalias !0
; IR-NEXT:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; IR-NEXT:   %polly.loop_cond = icmp sle i64 %polly.indvar_next, 99
; IR-NEXT:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

; IR:      polly.loop_preheader:
; IR-NEXT:   %41 = add i16 %val, 1
; IR-NEXT:   %42 = add i16 %ptr13, 9
; IR-NEXT:   %scevgep = getelementptr i64, i64* %A, i16 %42
; IR-NEXT:   br label %polly.loop_header
;
target datalayout = "e-p:16:16:16-m:e-i64:64-f80:128-n8:16:16:64-S128"

define void @f(i64* %A, i64* %B, i64* %ptr, i16 %val) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add.ptr = getelementptr inbounds i64, i64* %ptr, i64 1
  %tmp = ptrtoint i64* %add.ptr to i16
  %add = add nsw i16 %tmp, 1
  %add1 = add nsw i16 %val, 1
  %tmp1 = inttoptr i16 %add1 to i64*
  %add.ptr2 = getelementptr inbounds i64, i64* %tmp1, i64 1
  %tmp2 = ptrtoint i64* %add.ptr2 to i16
  %arrayidx = getelementptr inbounds i64, i64* %B, i16 %tmp2
  %tmp3 = load i64, i64* %arrayidx
  %arrayidx3 = getelementptr inbounds i64, i64* %A, i16 %add
  %tmp4 = load i64, i64* %arrayidx3
  %add4 = add nsw i64 %tmp4, %tmp3
  store i64 %add4, i64* %arrayidx3
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
