; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | \
; RUN:     FileCheck %s -check-prefix=DETECT

; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | \
; RUN:     FileCheck %s -check-prefix=SCOP

; DETECT: Valid Region for Scop: loop => barrier
; DETECT-NEXT: Valid Region for Scop: branch => end

; SCOP: Statements {
; SCOP-NEXT: 	Stmt_then
; SCOP-NEXT:         Domain :=
; SCOP-NEXT:             [p_0] -> { Stmt_then[] : p_0 <= -2 or p_0 >= 0 };
; SCOP-NEXT:         Schedule :=
; SCOP-NEXT:             [p_0] -> { Stmt_then[] -> [] };
; SCOP-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:             [p_0] -> { Stmt_then[] -> MemRef_A[0] };
; SCOP-NEXT: }

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

define void @f(i16 %event, ptr %A) {
entry:
  br label %loop

loop:
  %indvar = phi i8 [ 0, %entry ], [ %indvar.next, %loop ]
  %indvar.next = add i8 %indvar, -1
  store float 1.0, ptr %A
  %cmp = icmp eq i8 %indvar.next, 0
  br i1 false, label %barrier, label %loop

barrier:
  fence seq_cst
  br label %branch

branch:
  br i1 %cmp, label %branch, label %then

then:
  store float 1.0, ptr %A
  br label %end

end:
  ret void
}
