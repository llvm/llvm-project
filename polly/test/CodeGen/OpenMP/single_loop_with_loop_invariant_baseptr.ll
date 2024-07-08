; RUN: opt %loadNPMPolly -aa-pipeline=tbaa -polly-parallel -polly-parallel-force -polly-parallel-force -polly-invariant-load-hoisting=true '-passes=print<polly-ast>' -disable-output < %s | FileCheck %s -check-prefix=AST
; RUN: opt %loadNPMPolly -aa-pipeline=tbaa -polly-parallel -polly-parallel-force -polly-parallel-force -polly-invariant-load-hoisting=true -passes=polly-codegen -S -verify-dom-info < %s | FileCheck %s -check-prefix=IR

; #define N 1024
; float A[N];
;
; void single_parallel_loop(void) {
;   for (long i = 0; i < N; i++)
;     A[i] = 1;
; }

; AST: #pragma simd
; AST: #pragma omp parallel for
; AST: for (int c0 = 0; c0 <= 1023; c0 += 1)
; AST:   Stmt_S(c0);

; IR: @single_parallel_loop_polly_subfn
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @single_parallel_loop(float** %A) nounwind {
entry:
  br label %for.i

for.i:
  %indvar = phi i64 [ %indvar.next, %for.inc], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %S, label %exit

S:
  %ptr = load float*, float** %A,  !tbaa !2
  %scevgep = getelementptr float, float* %ptr, i64 %indvar
  %val = load float, float* %scevgep, !tbaa !6
  %sum = fadd float %val, 1.0
  store float %sum, float* %scevgep, !tbaa !6
  br label %for.inc

for.inc:
  %indvar.next = add i64 %indvar, 1
  br label %for.i

exit:
  ret void
}

!2 = !{!"float", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!6 = !{!"float *ptr", !3, i64 0}
