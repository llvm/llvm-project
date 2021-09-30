; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-ast -analyze < %s | FileCheck %s -check-prefix=AST
; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-codegen -S -verify-dom-info < %s | FileCheck %s -check-prefix=IR

; This test case verifies that we create correct code even if two OpenMP loops
; share common outer variables.

; AST:   Stmt_for_body35(0);
; AST:   #pragma simd
; AST:   #pragma omp parallel for
; AST:   for (int c0 = 1; c0 < -p_0 + nj - 1; c0 += 1)
; AST:     Stmt_for_body35(c0);

; IR: @foo_polly_subfn
; IR-NOT: @foo_polly_subfn_1

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i64 %nj, [512 x double]* %R) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %k.014 = phi i64 [ %inc87, %for.inc86 ], [ 0, %entry ]
  %j.010 = add nsw i64 %k.014, 1
  br i1 undef, label %for.body35, label %for.inc86

for.body35:
  %j.012 = phi i64 [ %j.0, %for.body35 ], [ %j.010, %for.cond1.preheader ]
  %arrayidx39 = getelementptr inbounds [512 x double], [512 x double]* %R, i64 0, i64 %j.012
  store double 0.000000e+00, double* %arrayidx39
  %j.0 = add nsw i64 %j.012, 1
  %cmp34 = icmp slt i64 %j.0, %nj
  br i1 %cmp34, label %for.body35, label %for.inc86

for.inc86:
  %inc87 = add nsw i64 %k.014, 1
  br i1 undef, label %for.cond1.preheader, label %for.end88

for.end88:
  ret void
}
