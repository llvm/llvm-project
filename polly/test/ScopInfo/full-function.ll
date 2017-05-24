; RUN: opt %loadPolly -polly-scops -analyze -polly-detect-full-functions < %s \
; RUN: | FileCheck %s -check-prefix=FULL
; RUN: opt %loadPolly -polly-scops -analyze < %s \
; RUN: | FileCheck %s -check-prefix=WITHOUT-FULL

; FULL:      Region: %bb---FunctionExit
; FULL:      Statements {
; FULL-NEXT: 	Stmt_loop_1
; FULL-NEXT:         Domain :=
; FULL-NEXT:             [p] -> { Stmt_loop_1[i0] : p = 42 and 0 <= i0 <= 1025 };
; FULL-NEXT:         Schedule :=
; FULL-NEXT:             [p] -> { Stmt_loop_1[i0] -> [1, i0] };
; FULL-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; FULL-NEXT:             [p] -> { Stmt_loop_1[i0] -> MemRef_A[0] };
; FULL-NEXT: 	Stmt_loop_2
; FULL-NEXT:         Domain :=
; FULL-NEXT:             [p] -> { Stmt_loop_2[i0] : 0 <= i0 <= 1025 and (p >= 43 or p <= 41) };
; FULL-NEXT:         Schedule :=
; FULL-NEXT:             [p] -> { Stmt_loop_2[i0] -> [0, i0] : p >= 43 or p <= 41 };
; FULL-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; FULL-NEXT:             [p] -> { Stmt_loop_2[i0] -> MemRef_A[0] };
; FULL-NEXT: }

; WITHOUT-FULL:        Region: %loop.2---%merge
; WITHOUT-FULL:        Statements {
; WITHOUT-FULL-NEXT:    	Stmt_loop_2
; WITHOUT-FULL-NEXT:            Domain :=
; WITHOUT-FULL-NEXT:                { Stmt_loop_2[i0] : 0 <= i0 <= 1025 };
; WITHOUT-FULL-NEXT:            Schedule :=
; WITHOUT-FULL-NEXT:                { Stmt_loop_2[i0] -> [i0] };
; WITHOUT-FULL-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; WITHOUT-FULL-NEXT:                { Stmt_loop_2[i0] -> MemRef_A[0] };
; WITHOUT-FULL-NEXT:    }

; WITHOUT-FULL:         Region: %loop.1---%merge
; WITHOUT-FULL:         Statements {
; WITHOUT-FULL-NEXT:    	Stmt_loop_1
; WITHOUT-FULL-NEXT:            Domain :=
; WITHOUT-FULL-NEXT:                { Stmt_loop_1[i0] : 0 <= i0 <= 1025 };
; WITHOUT-FULL-NEXT:            Schedule :=
; WITHOUT-FULL-NEXT:                { Stmt_loop_1[i0] -> [i0] };
; WITHOUT-FULL-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; WITHOUT-FULL-NEXT:                { Stmt_loop_1[i0] -> MemRef_A[0] };
; WITHOUT-FULL-NEXT:    }

define void @foo(float* %A, i32 %p) {
bb:
  %cmp = icmp eq i32 %p, 42
  br i1 %cmp, label %loop.1, label %loop.2

loop.1:
  %indvar.1 = phi i64 [0, %bb], [%indvar.next.1, %loop.1]
  %indvar.next.1 = add i64 %indvar.1, 1
  store float 42.0, float* %A
  %cmp.1 = icmp sle i64 %indvar.1, 1024
  br i1 %cmp.1, label %loop.1, label %merge

loop.2:
  %indvar.2 = phi i64 [0, %bb], [%indvar.next.2, %loop.2]
  %indvar.next.2 = add i64 %indvar.2, 1
  store float 42.0, float* %A
  %cmp.2 = icmp sle i64 %indvar.2, 1024
  br i1 %cmp.2, label %loop.2, label %merge

merge:
  ret void
}
