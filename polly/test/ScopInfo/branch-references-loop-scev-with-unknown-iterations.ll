; RUN: opt %loadNPMPolly '-passes=print<polly-detect>,print<polly-function-scops>' -disable-output < %s 2>&1 | \
; RUN:     FileCheck %s -check-prefix=NONAFFINE
; RUN: opt %loadNPMPolly '-passes=print<polly-detect>,print<polly-function-scops>' -disable-output \
; RUN:     -polly-allow-nonaffine-branches=false < %s 2>&1 | \
; RUN:     FileCheck %s -check-prefix=NO-NONEAFFINE

; NONAFFINE-NOT: Statements

; NO-NONEAFFINE: Statements {
; NO-NONEAFFINE-NEXT: 	Stmt_then
; NO-NONEAFFINE-NEXT:         Domain :=
; NO-NONEAFFINE-NEXT:             [p_0] -> { Stmt_then[] : p_0 <= -2 or p_0 >= 0 };
; NO-NONEAFFINE-NEXT:         Schedule :=
; NO-NONEAFFINE-NEXT:             [p_0] -> { Stmt_then[] -> [] };
; NO-NONEAFFINE-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; NO-NONEAFFINE-NEXT:             [p_0] -> { Stmt_then[] -> MemRef_A[0] };
; NO-NONEAFFINE-NEXT: }

; Verify that this test case does not crash -polly-scops. The problem in
; this test case is that the branch instruction in %branch references
; a scalar evolution expression for which no useful value can be computed at the
; location %branch, as the loop %loop does not terminate. At some point, we
; did not identify the branch condition as non-affine during scop detection.
; This test verifies that we either model the branch condition as non-affine
; region (and return an empty scop) or only detect a smaller region if
; non-affine conditions are not allowed.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

define void @f(i16 %event, ptr %A) {
entry:
  br label %loop

loop:
  %indvar = phi i8 [ 0, %entry ], [ %indvar.next, %loop ]
  %indvar.next = add i8 %indvar, 1
  %cmp = icmp eq i8 %indvar.next, 0
  br i1 false, label %branch, label %loop

branch:
  br i1 %cmp, label %end, label %then

then:
  store float 1.0, ptr %A
  br label %end

end:
  ret void
}
