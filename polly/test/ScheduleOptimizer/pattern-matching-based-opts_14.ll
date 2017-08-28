; RUN: opt %loadPolly -polly-import-jscop -polly-opt-isl  \
; RUN: -polly-target-throughput-vector-fma=1 \
; RUN: -polly-target-latency-vector-fma=8 \
; RUN: -polly-target-1st-cache-level-associativity=8 \
; RUN: -polly-target-2nd-cache-level-associativity=8 \
; RUN: -polly-target-1st-cache-level-size=32768 \
; RUN: -polly-target-vector-register-bitwidth=256 \
; RUN: -polly-target-2nd-cache-level-size=262144 \
; RUN: -polly-import-jscop-postfix=transformed -polly-codegen -S < %s \
; RUN: | FileCheck %s
;
; Check that we do not create different alias sets for locations represented by
; different raw pointers.
;
; Also check that we disable the Loop Vectorizer.
;
; CHECK-NOT: !76 = distinct !{!76, !5, !"second level alias metadata"}
; CHECK: !{!"llvm.loop.vectorize.enable", i1 false}
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define void @kernel_gemm(i32 %ni, i32 %nj, i32 %nk, [1024 x double]* %A, [1024 x double]* %B, [1024 x double]* %C, double* %C1) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.inc22, %entry.split
  %indvars.iv43 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next44, %for.inc22 ]
  br label %for.body3

for.body3:                                        ; preds = %for.inc19, %for.body
  %indvars.iv40 = phi i64 [ 0, %for.body ], [ %indvars.iv.next41, %for.inc19 ]
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body3
  %indvars.iv = phi i64 [ 0, %for.body3 ], [ %indvars.iv.next, %for.body6 ]
  %tmp = load double, double* %C1, align 8
  %arrayidx9 = getelementptr inbounds [1024 x double], [1024 x double]* %A, i64 %indvars.iv43, i64 %indvars.iv
  %tmp1 = load double, double* %arrayidx9, align 8
  %arrayidx13 = getelementptr inbounds [1024 x double], [1024 x double]* %B, i64 %indvars.iv, i64 %indvars.iv40
  %tmp2 = load double, double* %arrayidx13, align 8
  %mul = fmul double %tmp1, %tmp2
  %add = fadd double %tmp, %mul
  %arrayidx17 = getelementptr inbounds [1024 x double], [1024 x double]* %C, i64 %indvars.iv43, i64 %indvars.iv40
  %tmp3 = load double, double* %arrayidx17, align 8
  %add18 = fadd double %tmp3, %add
  store double %add18, double* %arrayidx17, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.body6, label %for.inc19

for.inc19:                                        ; preds = %for.body6
  %indvars.iv.next41 = add nuw nsw i64 %indvars.iv40, 1
  %exitcond42 = icmp ne i64 %indvars.iv.next41, 1024
  br i1 %exitcond42, label %for.body3, label %for.inc22

for.inc22:                                        ; preds = %for.inc19
  %indvars.iv.next44 = add nuw nsw i64 %indvars.iv43, 1
  %exitcond45 = icmp ne i64 %indvars.iv.next44, 1024
  br i1 %exitcond45, label %for.body, label %for.end24

for.end24:                                        ; preds = %for.inc22
  ret void
}
