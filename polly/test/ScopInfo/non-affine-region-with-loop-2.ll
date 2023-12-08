; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-allow-nonaffine-loops -polly-print-scops -polly-codegen -disable-output < %s | FileCheck %s
;
; CHECK:    Stmt_loop3
; CHECK:            Domain :=
; CHECK:                [indvar] -> { Stmt_loop3[0] : indvar >= 101 or indvar <= 99 };
; CHECK:            Schedule :=
; CHECK:                [indvar] -> { Stmt_loop3[i0] -> [0, 0] : indvar >= 101 or indvar <= 99 };
; CHECK:    Stmt_loop2__TO__loop
; CHECK:            Domain :=
; CHECK:                [indvar] -> { Stmt_loop2__TO__loop[] : indvar >= 101 or indvar <= 99 };
; CHECK:            Schedule :=
; CHECK:                [indvar] -> { Stmt_loop2__TO__loop[] -> [1, 0] : indvar >= 101 or indvar <= 99 };
;
define void @foo(ptr %A, i64 %p) {
entry:
  br label %loop

loop:
  %indvar.3 = phi i64 [0, %entry], [%indvar.3, %loop], [%indvar.next.3, %next2], [%indvar.next.3, %cond]
  %indvar = phi i64 [0, %entry], [%indvar.next, %loop], [0, %next2], [0, %cond]
  %indvar.next = add i64 %indvar, 1
  fence seq_cst
  %cmp = icmp eq i64 %indvar, 100
  br i1 %cmp, label %next, label %loop

next:
  %indvar.next.3 = add i64 %indvar.3, 1
  %cmp.3 = icmp eq i64 %indvar, 100
  br i1 %cmp.3, label %loop3, label %exit

loop3:
  %indvar.6 = phi i64 [0, %next], [%indvar.next.6, %loop3]
  %indvar.next.6 = add i64 %indvar.6, 1
  %cmp.6 = icmp eq i64 %indvar.6, 100
  br i1 %cmp.3, label %loop3, label %loop2

loop2:
  %indvar.2 = phi i64 [0, %loop3], [%indvar.next.2, %loop2], [0, %cond]
  %indvar.next.2 = add i64 %indvar.2, 1
  %prod = mul i64 %indvar.2, %indvar.2
  store i64 %indvar, ptr %A
  %cmp.2 = icmp eq i64 %prod, 100
  br i1 %cmp.2, label %loop2, label %next2

next2:
  %cmp.4 = icmp eq i64 %p, 100
  br i1 %cmp.4, label %loop, label %cond

cond:
  br i1 false, label %loop, label %loop2

exit:
  ret void
}
