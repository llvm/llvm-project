; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true \
; RUN: -polly-target-throughput-vector-fma=1 \
; RUN: -polly-target-latency-vector-fma=8 \
; RUN: -analyze -polly-ast -polly-target-1st-cache-level-associativity=8 \
; RUN: -polly-target-2nd-cache-level-associativity=8 \
; RUN: -polly-target-1st-cache-level-size=32768 \
; RUN: -polly-target-vector-register-bitwidth=256 \
; RUN: -polly-target-2nd-cache-level-size=262144 < %s
;
; RUN: opt %loadPolly -analyze -polly-dependences < %s | FileCheck %s \
; RUN: --check-prefix=DEPENDENCES
;
;    /* C := A * B + C */
;    /* Elements of the matrices A, B, C have the char type. */
;    /* The type size of elements of the matrix multiplication operands is used
;       to determine the parameters of the code produced by the optimization
;       of the matrix multiplication (e.g. bounds of the loops of the loop
;       nest, the innermost loop body). This test checks the form of
;       the generated loop nest. See getMicroKernelParams and
;       getMacroKernelParams from lib/Transform/ScheduleOptimizer.cpp
;       for details.
;
;       This patch also checks that we can detect matrix multiplication
;       in case there are reduction dependencies and there are not RAW
;       dependencies. */
;    for (i = 0; i < _PB_NI; i++)
;      for (j = 0; j < _PB_NJ; j++)
;   for (k = 0; k < _PB_NK; ++k)
;     C[i][j] += A[i][k] * B[k][j];
;
; DEPENDENCES:  RAW dependences:
; DEPENDENCES-NEXT:    {  }
; DEPENDENCES-NEXT:  WAR dependences:
; DEPENDENCES-NEXT:    {  }
; DEPENDENCES-NEXT:  WAW dependences:
; DEPENDENCES-NEXT:    {  }
; DEPENDENCES-NEXT:  Reduction dependences:
; DEPENDENCES-NEXT:    { Stmt_for_body6[i0, i1, i2] -> Stmt_for_body6[i0, i1, 1 + i2] : 0 <= i0 <= 1023 and 0 <= i1 <= 1023 and 0 <= i2 <= 1022 }
; DEPENDENCES-NEXT:  Transitive closure of reduction dependences:
; DEPENDENCES-NEXT:    { Stmt_for_body6[i0, i1, i2] -> Stmt_for_body6[i0, i1, o2] : 0 <= i0 <= 1023 and 0 <= i1 <= 1023 and ((i2 >= 0 and i2 < o2 <= 1023) or (i2 <= 1023 and 0 <= o2 < i2)) }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define internal void @kernel_gemm(i32 %ni, i32 %nj, i32 %nk, i8 signext %alpha, i8 signext %beta, [1024 x i8]* %C, [1024 x i8]* %A, [1024 x i8]* %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc23, %entry.split
  %indvars.iv45 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next46, %for.inc23 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc20, %for.cond1.preheader
  %indvars.iv42 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next43, %for.inc20 ]
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.cond4.preheader
  %indvars.iv = phi i64 [ 0, %for.cond4.preheader ], [ %indvars.iv.next, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [1024 x i8], [1024 x i8]* %A, i64 %indvars.iv45, i64 %indvars.iv
  %tmp = load i8, i8* %arrayidx8, align 1
  %arrayidx12 = getelementptr inbounds [1024 x i8], [1024 x i8]* %B, i64 %indvars.iv, i64 %indvars.iv42
  %tmp1 = load i8, i8* %arrayidx12, align 1
  %mul = mul i8 %tmp1, %tmp
  %arrayidx17 = getelementptr inbounds [1024 x i8], [1024 x i8]* %C, i64 %indvars.iv45, i64 %indvars.iv42
  %tmp2 = load i8, i8* %arrayidx17, align 1
  %add = add i8 %mul, %tmp2
  store i8 %add, i8* %arrayidx17, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.body6, label %for.inc20

for.inc20:                                        ; preds = %for.body6
  %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1
  %exitcond44 = icmp ne i64 %indvars.iv.next43, 1024
  br i1 %exitcond44, label %for.cond4.preheader, label %for.inc23

for.inc23:                                        ; preds = %for.inc20
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %exitcond47 = icmp ne i64 %indvars.iv.next46, 1024
  br i1 %exitcond47, label %for.cond1.preheader, label %for.end25

for.end25:                                        ; preds = %for.inc23
  ret void
}
