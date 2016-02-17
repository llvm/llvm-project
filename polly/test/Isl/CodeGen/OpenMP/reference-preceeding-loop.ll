; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-ast -analyze < %s | FileCheck %s -check-prefix=AST
; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-codegen -S -verify-dom-info < %s | FileCheck %s -check-prefix=IR


; - Test the case where scalar evolution references a loop that is outside
;   of the scop, but does not contain the scop.

; AST:         {
; AST-NEXT:    #pragma simd
; AST-NEXT:    #pragma omp parallel for
; AST-NEXT:    for (int c0 = 0; c0 < -p_0 + symbol; c0 += 1)
; AST-NEXT:      Stmt_while_body(c0);
; AST-NEXT:    if (p_0 >= symbol)
; AST-NEXT:      Stmt_while_body(0);
; AST-NEXT:    }

; IR: @update_model_polly_subfn
; IR-NOT: @update_model_polly_subfn_1

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@cum_freq = external global [258 x i64], align 16

define void @update_model(i64 %symbol) {
entry:
  br label %for.one

for.one:
  %i.1 = phi i64 [ %dec07, %for.one ], [ %symbol, %entry ]
  %dec07 = add nsw i64 %i.1, -1
  br i1 undef, label %for.one, label %while.body

while.body:
  %indvar = phi i64 [ %sub42, %while.body ], [ %i.1, %for.one ]
  %sub42 = add nsw i64 %indvar, -1
  %arrayidx44 = getelementptr inbounds [258 x i64], [258 x i64]* @cum_freq, i64 0, i64 %sub42
  store i64 1, i64* %arrayidx44, align 4
  %cmp40 = icmp sgt i64 %sub42, 0
  br i1 %cmp40, label %while.body, label %while.end

while.end:
  ret void
}
