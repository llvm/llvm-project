; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb -polly-analyze-read-only-scalars=false '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb -polly-analyze-read-only-scalars=true  '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s -check-prefix=SCALARS

; CHECK-NOT: Memref_scalar

; SCALARS: float MemRef_scalar; // Element size 4

; SCALARS: ReadAccess :=  [Reduction Type: NONE] [Scalar: 1]
; SCALARS:     { Stmt_stmt1[i0] -> MemRef_scalar[] };
; SCALARS: ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; SCALARS:     { Stmt_stmt1[i0] -> MemRef_scalar2[] };


define void @foo(ptr noalias %A, ptr %B, float %scalar, float %scalar2) {
entry:
  br label %loop

loop:
  %indvar = phi i64 [0, %entry], [%indvar.next, %loop.backedge]
  br label %stmt1

stmt1:
  %val = load float, ptr %A
  %sum = fadd float %val, %scalar
  store float %sum, ptr %A
  store float %scalar2, ptr %B
  br label %loop.backedge

loop.backedge:
  %indvar.next = add i64 %indvar, 1
  %cond = icmp sle i64 %indvar, 100
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
