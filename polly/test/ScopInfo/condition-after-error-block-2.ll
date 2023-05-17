; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-print-scops -disable-output < %s | FileCheck %s

; Verify that we do not allow PHI nodes such as %phi, if they reference an error
; block and are used by anything else than a terminator instruction.

; CHECK:      Statements {
; CHECK-NEXT: 	Stmt_loop
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p] -> { Stmt_loop[i0] : p >= 13 and 0 <= i0 <= 1025 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p] -> { Stmt_loop[i0] -> [i0] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [p] -> { Stmt_loop[i0] -> MemRef_X[0] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p] -> { Stmt_loop[i0] -> MemRef_phi[] };
; CHECK-NEXT: }

declare void @bar()

define void @foo(ptr %X, i64 %p) {
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
  store float 41.0, ptr %X
  %cmp2 = icmp sle i64 %indvar, 1024
  br i1 %cmp2, label %loop, label %merge

cond:
  br label %cond2

cond2:
  call void @bar()
  br label %merge

merge:
  %phi = phi i1 [false, %cond2], [true, %loop]
  %add = add i1 %phi, 1
  br i1 %add, label %A, label %B

A:
  store float 42.0, ptr %X
  br label %exit

B:
  call void @bar()
  store float 41.0, ptr %X
  br label %exit

exit:
  ret void
}
