; RUN: opt %loadPolly -polly-scops -analyze < %s \
; RUN:  -polly-invariant-load-hoisting \
; RUN:  | FileCheck %s

; Verify that a delinearized and a not delinearized access are not
; canonizalized.

; CHECK:      Stmt_body1
; CHECK-NEXT:   Domain :=
; CHECK-NEXT:       [n] -> { Stmt_body1[i0] : 0 <= i0 <= 1022 };
; CHECK-NEXT:   Schedule :=
; CHECK-NEXT:       [n] -> { Stmt_body1[i0] -> [i0, 0] };
; CHECK-NEXT:   MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       [n] -> { Stmt_body1[i0] -> MemRef_baseB[0] };
; CHECK-NEXT: Stmt_body2
; CHECK-NEXT:   Domain :=
; CHECK-NEXT:       [n] -> { Stmt_body2[i0] : 0 <= i0 <= 1022 };
; CHECK-NEXT:   Schedule :=
; CHECK-NEXT:       [n] -> { Stmt_body2[i0] -> [i0, 1] };
; CHECK-NEXT:   MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       [n] -> { Stmt_body2[i0] -> MemRef_baseA[i0, i0] };
; CHECK-NEXT: }


define void @foo(float** %A, i64 %n, i64 %m) {
start:
  br label %loop

loop:
  %indvar = phi i64 [0, %start], [%indvar.next, %latch]
  %indvar.next = add nsw i64 %indvar, 1
  %icmp = icmp slt i64 %indvar.next, 1024
  br i1 %icmp, label %body1, label %exit

body1:
  %baseB = load float*, float** %A
  store float 42.0, float* %baseB
  br label %body2

body2:
  %baseA = load float*, float** %A
  %offsetA = mul i64 %indvar, %n
  %offsetA2 = add i64 %offsetA, %indvar
  %ptrA = getelementptr float, float* %baseA, i64 %offsetA2
  store float 42.0, float* %ptrA
  br label %latch

latch:
  br label %loop

exit:
  ret void

}
