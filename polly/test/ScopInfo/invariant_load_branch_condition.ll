; RUN: opt %loadPolly -polly-scops -analyze \
; RUN:     -polly-invariant-load-hoisting < %s | FileCheck %s

; CHECK:      Invariant Accesses: {
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [val] -> { Stmt_next[] -> MemRef_ptr[0] };
; CHECK-NEXT:         Execution Context: [val] -> {  :  }
; CHECK-NEXT: }

; CHECK: Statements {
; CHECK-NEXT: 	Stmt_a
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [val] -> { Stmt_a[] : val = -1 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [val] -> { Stmt_a[] -> [1, 0] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [val] -> { Stmt_a[] -> MemRef_X[0] };
; CHECK-NEXT: 	Stmt_loop
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [val] -> { Stmt_loop[i0] : val = 0 and 0 <= i0 <= 1025 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [val] -> { Stmt_loop[i0] -> [0, i0] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [val] -> { Stmt_loop[i0] -> MemRef_X[0] };
; CHECK-NEXT: }

define void @foo(i1* %ptr, float* %X) {
entry:
  br label %next

next:
  %val = load i1, i1* %ptr
  br i1 %val, label %a, label %loop

a:
  store float 1.0, float* %X
  br label %merge

loop:
  %indvar = phi i64 [0, %next], [%indvar.next, %loop]
  store float 1.0, float* %X
  %indvar.next = add nsw i64 %indvar, 1
  %cmp = icmp sle i64 %indvar, 1024
  br i1 %cmp, label %loop, label %merge

merge:
  br label %exit

exit:
  ret void
}
