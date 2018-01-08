; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s

; Verify that we allow scops containing uniform branch conditions, where all
; but one incoming block comes from an error condition.

; CHECK:         Statements {
; CHECK-NEXT:     	Stmt_A
; CHECK-NEXT:             Domain :=
; CHECK-NEXT:                 [p] -> { Stmt_A[] };
; CHECK-NEXT:             Schedule :=
; CHECK-NEXT:                 [p] -> { Stmt_A[] -> [1, 0] };
; CHECK-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [p] -> { Stmt_A[] -> MemRef_X[0] };
; CHECK-NEXT:     	Stmt_loop
; CHECK-NEXT:             Domain :=
; CHECK-NEXT:                 [p] -> { Stmt_loop[i0] : p >= 13 and 0 <= i0 <= 1025 };
; CHECK-NEXT:             Schedule :=
; CHECK-NEXT:                 [p] -> { Stmt_loop[i0] -> [0, i0] };
; CHECK-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [p] -> { Stmt_loop[i0] -> MemRef_X[0] };
; CHECK-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [p] -> { Stmt_loop[i0] -> MemRef_phi__phi[] };
; CHECK-NEXT:     }

declare void @bar()

define void @foo(float* %X, i64 %p) {
entry:
  br label %br

br:
  %cmp1 = icmp sle i64 %p, 12
  br i1 %cmp1, label %A, label %br2

br2:
  %cmp3 = icmp sle i64 %p, 12
  br i1 %cmp3, label %cond, label %loop

loop:
  %indvar = phi i64 [0, %br2], [%indvar.next, %loop]
  %indvar.next = add nsw i64 %indvar, 1
  store float 41.0, float* %X
  %cmp2 = icmp sle i64 %indvar, 1024
  br i1 %cmp2, label %loop, label %merge

cond:
  br label %cond2

cond2:
  call void @bar()
  br label %merge

merge:
  %phi = phi i1 [false, %cond2], [true, %loop]
  br i1 %phi, label %A, label %B

A:
  store float 42.0, float* %X
  br label %exit

B:
  call void @bar()
  store float 41.0, float* %X
  br label %exit

exit:
  ret void
}
