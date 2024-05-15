; RUN: opt %loadPolly -polly-allow-nonaffine-loops -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-allow-nonaffine-loops -polly-codegen -disable-output
;
; CHECK:      Domain :=
; CHECK-NEXT:   { Stmt_loop2__TO__loop[] };
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
  br i1 %cmp.3, label %loop2, label %exit

loop2:
  %indvar.2 = phi i64 [0, %next], [%indvar.next.2, %loop2], [0, %cond]
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
