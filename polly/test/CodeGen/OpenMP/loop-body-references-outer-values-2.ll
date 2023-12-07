; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-print-ast -disable-output < %s | FileCheck %s -check-prefix=AST
; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-codegen -S -verify-dom-info < %s | FileCheck %s -check-prefix=IR

; AST: #pragma simd
; AST: #pragma omp parallel for
; AST: for (int c0 = 0; c0 <= 1023; c0 += 1)
; AST:   Stmt_for_i(c0);

; IR: getelementptr inbounds { ptr }, ptr %polly.par.userContext, i32 0, i32 0

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @kernel_trmm(ptr %B) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %extern = add i64 1, 0
  br label %for.i

for.i:
  %indvar.i = phi i64 [ %indvar.i.next, %for.i ], [ 0, %for.cond1.preheader ]
  %getelementptr = getelementptr [1024 x double], ptr %B, i64 %extern, i64 %indvar.i
  store double 0.000000e+00, ptr %getelementptr
  %indvar.i.next = add i64 %indvar.i, 1
  %exitcond.i = icmp ne i64 %indvar.i.next, 1024
  br i1 %exitcond.i, label %for.i, label %end

end:
  ret void
}
