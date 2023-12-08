; RUN: opt %loadPolly -polly-stmt-granularity=bb           -polly-use-llvm-names=0 -polly-print-scops -disable-output < %s | FileCheck %s -match-full-lines -check-prefix=IDX
; RUN: opt %loadPolly -polly-stmt-granularity=bb           -polly-use-llvm-names=1 -polly-print-scops -disable-output < %s | FileCheck %s -match-full-lines -check-prefix=BB
; RUN: opt %loadPolly -polly-stmt-granularity=scalar-indep -polly-use-llvm-names=0 -polly-print-scops -disable-output < %s | FileCheck %s -match-full-lines -check-prefix=IDX
; RUN: opt %loadPolly -polly-stmt-granularity=scalar-indep -polly-use-llvm-names=1 -polly-print-scops -disable-output < %s | FileCheck %s -match-full-lines -check-prefix=BB
;
; Check that the statement has the same name, regardless of how the
; basic block is split into multiple statements.
; Note that %unrelatedA and %unrelatedB can be put into separate
; statements, but are removed because those have no side-effects.
;
; for (int j = 0; j < n; j += 1) {
; body:
;   double unrelatedA = 21.0 + 21.0;
;   A[0] = 0.0;
;   double unrelatedB = 21.0 + 21.0;
; }
;
define void @func(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %unrelatedA = fadd double 21.0, 21.0
      store double 0.0, ptr %A
      %unrelatedB = fadd double 21.0, 21.0
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; IDX:      Statements {
; IDX-NEXT:     Stmt1

; BB:       Statements {
; BB-NEXT:      Stmt_body
