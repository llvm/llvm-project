; RUN: opt %loadPolly -polly-print-scops -disable-output < %s \
; RUN:  -polly-invariant-load-hoisting \
; RUN:  | FileCheck %s

; Verify that we canonicalize accesses even tough one of the accesses (even
; the canonical base) has a partial execution context. This is correct as
; the combined execution context still covers both accesses.

; CHECK:      Invariant Accesses: {
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_body2[i0] -> MemRef_A[0] };
; CHECK-NEXT:         Execution Context: {  :  }
; CHECK-NEXT: }

; CHECK:      Stmt_body1
; CHECK-NEXT:   Domain :=
; CHECK-NEXT:       { Stmt_body1[i0] : 0 <= i0 <= 1022 };
; CHECK-NEXT:   Schedule :=
; CHECK-NEXT:       { Stmt_body1[i0] -> [i0, 0] };
; CHECK-NEXT:   MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       { Stmt_body1[i0] -> MemRef_baseB[0] };
; CHECK-NEXT: Stmt_body2
; CHECK-NEXT:   Domain :=
; CHECK-NEXT:       { Stmt_body2[i0] : 0 <= i0 <= 510 };
; CHECK-NEXT:   Schedule :=
; CHECK-NEXT:       { Stmt_body2[i0] -> [i0, 1] };
; CHECK-NEXT:   MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       { Stmt_body2[i0] -> MemRef_baseB[0] };


define void @foo(float** %A) {
start:
  br label %loop

loop:
  %indvar = phi i64 [0, %start], [%indvar.next, %latch]
  %indvar.next = add nsw i64 %indvar, 1
  %icmp = icmp slt i64 %indvar.next, 1024
  br i1 %icmp, label %body1, label %exit

body1:
  %baseA = load float*, float** %A
  store float 42.0, float* %baseA
  %cmp = icmp slt i64 %indvar.next, 512
  br i1 %cmp, label %body2, label %latch

body2:
  %baseB = load float*, float** %A
  store float 42.0, float* %baseB
  br label %latch

latch:
  br label %loop

exit:
  ret void

}
