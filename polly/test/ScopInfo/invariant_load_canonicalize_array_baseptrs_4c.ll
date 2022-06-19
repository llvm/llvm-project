; RUN: opt %loadPolly -polly-print-scops -disable-output < %s \
; RUN:  -polly-invariant-load-hoisting \
; RUN:  | FileCheck %s

; Verify that arrays with different element types are not coalesced.

; CHECK:      Statements {
; CHECK-NEXT: 	Stmt_body1
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_body1[i0] : 0 <= i0 <= 1022 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_body1[i0] -> [i0, 0] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_body1[i0] -> MemRef_baseB[0] };
; CHECK-NEXT: 	Stmt_body2
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_body2[i0] : 0 <= i0 <= 1022 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_body2[i0] -> [i0, 1] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_body2[i0] -> MemRef_baseA[0] };
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
  %ptrcast = bitcast float* %baseA to i64*
  store i64 42, i64* %ptrcast
  br label %latch

latch:
  br label %loop

exit:
  ret void

}
