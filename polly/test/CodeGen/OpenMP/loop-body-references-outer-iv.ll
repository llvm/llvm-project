; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-ast -analyze < %s | FileCheck %s -check-prefix=AST
; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-codegen -S -verify-dom-info < %s | FileCheck %s -check-prefix=IR

; This code has failed the scev based code generation as the scev in the scop
; contains an AddRecExpr of an outer loop. When generating code, we did not
; properly forward the value of this expression to the subfunction.

; AST: #pragma omp parallel for
; AST: for (int c0 = 0; c0 <= 1023; c0 += 1)
; AST:  Stmt_for_j(c0);

; IR: @single_parallel_loop_polly_subfn

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [1024 x float] zeroinitializer, align 16

define void @single_parallel_loop() nounwind {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ %indvar.i.next, %for.i.inc], [ 0, %entry ]
  br label %for.j

for.j:
  %indvar.j = phi i64 [ %indvar.j.next, %for.j], [ 0, %for.i ]
  %sum = add i64 %indvar.j, %indvar.i
  %scevgep = getelementptr [1024 x float], [1024 x float]* @A, i64 0, i64 %sum
  store float 0.0, float *%scevgep
  %indvar.j.next = add i64 %indvar.j, 1
  %exitcond.j = icmp slt i64 %indvar.j.next, 1024
  br i1 %exitcond.j, label %for.j, label %for.i.inc

for.i.inc:
  fence seq_cst
  %indvar.i.next = add i64 %indvar.i, 1
  %exitcond.i = icmp ne i64 %indvar.i.next, 1024
  br i1 %exitcond.i, label %for.i, label %exit

exit:
  ret void
}
