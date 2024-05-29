; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s

declare void @bar()

define void @foo(ptr %A, i64 %p) {
start:
   br label %next

next:
   %cmpA = icmp sle i64 %p, 0
   br i1 %cmpA, label %error, label %ok

error:
   call void @bar()
   br label %merge

ok:
   br label %merge

merge:
   %phi = phi i64 [0, %error], [1, %ok]
   store float 42.0, ptr %A
   %cmp = icmp eq i64 %phi, %p
   br i1 %cmp, label %loop, label %exit

loop:
   %indvar = phi i64 [0, %merge], [%indvar.next, %loop]
   store float 42.0, ptr %A
   %indvar.next = add i64 %indvar, 1
   %cmp2 = icmp sle i64 %indvar, 1024
   br i1 %cmp2, label %loop, label %exit

exit:
   ret void
}

; CHECK:      Statements {
; CHECK-NEXT: 	Stmt_ok
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p] -> { Stmt_ok[] : p > 0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p] -> { Stmt_ok[] -> [0, 0] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p] -> { Stmt_ok[] -> MemRef_phi__phi[] };
; CHECK-NEXT: 	Stmt_merge
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p] -> { Stmt_merge[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p] -> { Stmt_merge[] -> [1, 0] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p] -> { Stmt_merge[] -> MemRef_phi__phi[] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [p] -> { Stmt_merge[] -> MemRef_A[0] };
; CHECK-NEXT: 	Stmt_loop
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p] -> { Stmt_loop[i0] : p = 1 and 0 <= i0 <= 1025 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p] -> { Stmt_loop[i0] -> [2, i0] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [p] -> { Stmt_loop[i0] -> MemRef_A[0] };
; CHECK-NEXT: }
