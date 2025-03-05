; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | \
; RUN:     FileCheck %s -check-prefix=NONAFFINE
; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb '-passes=print<polly-function-scops>' -disable-output \
; RUN:     -polly-allow-nonaffine-branches=false < %s 2>&1 | \
; RUN:     FileCheck %s -check-prefix=NO-NONEAFFINE

; NONAFFINE:      Statements {
; NONAFFINE-NEXT: 	Stmt_loop
; NONAFFINE-NEXT:         Domain :=
; NONAFFINE-NEXT:             [p] -> { Stmt_loop[0] : p = 100 };
; NONAFFINE-NEXT:         Schedule :=
; NONAFFINE-NEXT:             [p] -> { Stmt_loop[i0] -> [0, 0] };
; NONAFFINE-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; NONAFFINE-NEXT:             [p] -> { Stmt_loop[i0] -> MemRef_A[0] };
; NONAFFINE-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; NONAFFINE-NEXT:             [p] -> { Stmt_loop[i0] -> MemRef_cmp[] };
; NONAFFINE-NEXT: 	Stmt_branch__TO__end
; NONAFFINE-NEXT:         Domain :=
; NONAFFINE-NEXT:             [p] -> { Stmt_branch__TO__end[] : p = 100 };
; NONAFFINE-NEXT:         Schedule :=
; NONAFFINE-NEXT:             [p] -> { Stmt_branch__TO__end[] -> [1, 0] };
; NONAFFINE-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; NONAFFINE-NEXT:             [p] -> { Stmt_branch__TO__end[] -> MemRef_cmp[] };
; NONAFFINE-NEXT:         MayWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; NONAFFINE-NEXT:             [p] -> { Stmt_branch__TO__end[] -> MemRef_A[0] };
; NONAFFINE-NEXT: }

; NO-NONEAFFINE:      Statements {
; NO-NONEAFFINE-NEXT:    	Stmt_then
; NO-NONEAFFINE-NEXT:            Domain :=
; NO-NONEAFFINE-NEXT:                [p_0, p] -> { Stmt_then[] : p >= 2 + p_0 or p <= p_0 };
; NO-NONEAFFINE-NEXT:            Schedule :=
; NO-NONEAFFINE-NEXT:                [p_0, p] -> { Stmt_then[] -> [] };
; NO-NONEAFFINE-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; NO-NONEAFFINE-NEXT:                [p_0, p] -> { Stmt_then[] -> MemRef_A[0] };
; NO-NONEAFFINE-NEXT:    }

; NO-NONEAFFINE:      Statements {
; NO-NONEAFFINE-NEXT: 	Stmt_loop
; NO-NONEAFFINE-NEXT:         Domain :=
; NO-NONEAFFINE-NEXT:             [p] -> { Stmt_loop[0] : p = 100 };
; NO-NONEAFFINE-NEXT:         Schedule :=
; NO-NONEAFFINE-NEXT:             [p] -> { Stmt_loop[i0] -> [0] };
; NO-NONEAFFINE-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; NO-NONEAFFINE-NEXT:             [p] -> { Stmt_loop[i0] -> MemRef_A[0] };
; NO-NONEAFFINE-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; NO-NONEAFFINE-NEXT:             [p] -> { Stmt_loop[i0] -> MemRef_cmp[] };
; NO-NONEAFFINE-NEXT: }

; Verify that this test case does not crash -polly-scops. The problem in
; this test case is that the branch instruction in %branch references
; a scalar evolution expression for which no useful value can be computed at the
; location %branch, as the loop %loop does not terminate. At some point, we
; did not identify the branch condition as non-affine during scop detection.
; This test verifies that we either model the branch condition as non-affine
; region or only detect a smaller region if non-affine conditions are not
; allowed.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

define void @f(i16 %event, i8 %p, ptr %A) {
entry:
  br label %loop

loop:
  %indvar = phi i8 [ 0, %entry ], [ %indvar.next, %loop ]
  %indvar.next = add i8 %indvar, 1
  store float 1.0, ptr %A
  %cmp = icmp eq i8 %indvar.next, %p
  %possibly_infinite = icmp eq i8 100, %p
  br i1 %possibly_infinite, label %branch, label %loop

branch:
  br i1 %cmp, label %end, label %then

then:
  store float 1.0, ptr %A
  br label %end

end:
  ret void
}
