; RUN: opt %loadPolly -polly-codegen < %s -enable-polly-openmp -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; This test case implements the following code:
;
; for (i = 0; i < 1024; i++)
;   A[i] = A[i] * param
;
; The problem is that 'param' is not references in any subscript of loop
; bound, but it must still be forwarded to the OpenMP subfunction.

define void @foo(double %param, [1024 x double]* %A) {
entry:
  br label %for.preheader

for.preheader:
  br label %for.body

for.body:
  %indvar = phi i64 [ 0, %for.preheader ], [ %indvar.next, %for.inc ]
  %arrayidx = getelementptr [1024 x double]* %A, i64 0, i64 %indvar
  %val = load double* %arrayidx
  %mul = fmul double %param, %val
  store double %mul, double* %arrayidx, align 8
  br label %for.inc

for.inc:
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK: omp_subfn
