; RUN: opt %loadPolly -polly-print-scops -disable-output \
; RUN:     < %s | FileCheck %s

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_loop__TO__backedge
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_loop__TO__backedge[i0] : 0 <= i0 <= 100 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_loop__TO__backedge[i0] -> [i0, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_loop__TO__backedge[i0] -> MemRef_merge__phi[] };
; CHECK-NEXT:     Stmt_backedge
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_backedge[i0] : 0 <= i0 <= 100 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_backedge[i0] -> [i0, 1] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_backedge[i0] -> MemRef_merge__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_backedge[i0] -> MemRef_A[0] };
; CHECK-NEXT: }

define void @foo(ptr %A, i1 %cond0, i1 %cond1) {
entry:
  br label %loop

loop:
  %indvar = phi i64 [0, %entry], [%indvar.next, %backedge]
  %val0 = fadd float 1.0, 2.0
  %val1 = fadd float 1.0, 2.0
  br i1 %cond0, label %branch1, label %backedge

branch1:
  %val2 = fadd float 1.0, 2.0
  br i1 %cond1, label %branch2, label %backedge

branch2:
  br label %backedge

backedge:
  %merge = phi float [%val0, %loop], [%val1, %branch1], [%val2, %branch2]
  %indvar.next = add i64 %indvar, 1
  store float %merge, ptr %A
  %cmp = icmp sle i64 %indvar.next, 100
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
