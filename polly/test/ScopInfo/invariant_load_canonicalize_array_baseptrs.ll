; RUN: opt %loadPolly -polly-scops -analyze < %s \
; RUN:  -polly-invariant-load-hoisting \
; RUN:  | FileCheck %s

; CHECK:      Stmt_body1
; CHECK-NEXT:       Domain :=
; CHECK-NEXT:           { Stmt_body1[i0] : 0 <= i0 <= 1022 };
; CHECK-NEXT:       Schedule :=
; CHECK-NEXT:           { Stmt_body1[i0] -> [i0, 0] };
; CHECK-NEXT:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:          { Stmt_body1[i0] -> MemRef_baseB[0] };
; CHECK-NEXT: Stmt_body2
; CHECK-NEXT:       Domain :=
; CHECK-NEXT:           { Stmt_body2[i0] : 0 <= i0 <= 1022 };
; CHECK-NEXT:       Schedule :=
; CHECK-NEXT:           { Stmt_body2[i0] -> [i0, 1] };
; CHECK-NEXT:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_body2[i0] -> MemRef_baseB[0] };

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
  br label %body2

body2:
  %baseB = load float*, float** %A
  store float 42.0, float* %baseB
  br label %latch

latch:
  br label %loop

exit:
  ret void

}
