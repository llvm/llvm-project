; RUN: opt %loadPolly -polly-print-scops -disable-output < %s \
; RUN:  -polly-invariant-load-hoisting \
; RUN:  | FileCheck %s

; Verify that nested arrays with invariant base pointers are handled correctly.
; Specifically, we currently do not canonicalize arrays where some accesses are
; hoisted as invariant loads. If we would, we need to update the access function
; of the invariant loads as well. However, as this is not a very common
; situation, we leave this for now to avoid further complexity increases.
;
; In this test case the arrays baseA1 and baseA2 could be canonicalized to a
; single array, but there is also an invariant access to baseA1[0] through
; "%v0 = load float, float* %ptr" which prevents the canonicalization.

; CHECK:      Invariant Accesses: {
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_body2[i0] -> MemRef_A[0] };
; CHECK-NEXT:         Execution Context: {  :  }
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_body1[i0] -> MemRef_baseA1[0] };
; CHECK-NEXT:         Execution Context: {  :  }
; CHECK-NEXT: }

; CHECK:      Statements {
; CHECK-NEXT: 	Stmt_body1
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_body1[i0] : 0 <= i0 <= 1021 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_body1[i0] -> [i0, 0] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_body1[i0] -> MemRef_baseA1[1 + i0] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_body1[i0] -> MemRef_B[0] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_body1[i0] -> MemRef_B[0] };
; CHECK-NEXT: 	Stmt_body2
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_body2[i0] : 0 <= i0 <= 1021 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_body2[i0] -> [i0, 1] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_body2[i0] -> MemRef_baseA2[0] };
; CHECK-NEXT: }

define void @foo(float** %A, float* %B) {
start:
  br label %loop

loop:
  %indvar = phi i64 [1, %start], [%indvar.next, %latch]
  %indvar.next = add nsw i64 %indvar, 1
  %icmp = icmp slt i64 %indvar.next, 1024
  br i1 %icmp, label %body1, label %exit

body1:
  %baseA1 = load float*, float** %A
  %ptr = getelementptr inbounds float, float* %baseA1, i64 %indvar
  %v0 = load float, float* %ptr
  %v1 = load float, float* %baseA1
  store float %v0, float* %B
  store float %v1, float* %B
  br label %body2

body2:
  %baseA2 = load float*, float** %A
  store float undef, float* %baseA2
  br label %body3

body3:
  br label %latch

latch:
  br label %loop

exit:
  ret void

}
