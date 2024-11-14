; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa -polly-parallel -polly-parallel-force -polly-invariant-load-hoisting=true '-passes=print<polly-ast>' -disable-output < %s | FileCheck %s -check-prefix=AST
; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa -polly-parallel -polly-parallel-force -polly-invariant-load-hoisting=true -passes=polly-codegen -S -verify-dom-info < %s | FileCheck %s -check-prefix=IR

; The interesting part of this test case is the instruction:
;   %tmp = bitcast i8* %call to i64**
; which is not part of the scop. In the SCEV based code generation not '%tmp',
; but %call is a parameter of the SCoP and we need to make sure its value is
; properly forwarded to the subfunction.

; AST: #pragma omp parallel for
; AST: for (int c0 = 0; c0 < cols; c0 += 1)
; AST:   Stmt_for_body(c0);
; AST: if (cols <= 0)
; AST:   Stmt_for_body(0);

; IR: @foo_polly_subfn

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i64 %cols, ptr noalias %call) {
entry:
  br label %for.body

for.body:
  %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
  %tmp1 = load ptr, ptr %call, align 8
  %arrayidx.2 = getelementptr inbounds i64, ptr %tmp1, i64 %indvar
  store i64 1, ptr %arrayidx.2, align 4
  %indvar.next = add nsw i64 %indvar, 1
  %cmp = icmp slt i64 %indvar.next, %cols
  br i1 %cmp, label %for.body, label %end

end:
  ret void
}

; Another variation of this test case, now with even more of the index
; expression defined outside of the scop.

; AST: #pragma omp parallel for
; AST: for (int c0 = 0; c0 < cols; c0 += 1)
; AST:   Stmt_for_body(c0);
; AST: if (cols <= 0)
; AST:   Stmt_for_body(0);

; IR: @bar_polly_subfn

define void @bar(i64 %cols, ptr noalias %call) {
entry:
  br label %for.body

for.body:
  %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
  %tmp1 = load ptr, ptr %call, align 8
  %arrayidx.2 = getelementptr inbounds i64, ptr %tmp1, i64 %indvar
  store i64 1, ptr %arrayidx.2, align 4
  %indvar.next = add nsw i64 %indvar, 1
  %cmp = icmp slt i64 %indvar.next, %cols
  br i1 %cmp, label %for.body, label %end

end:
  ret void
}
