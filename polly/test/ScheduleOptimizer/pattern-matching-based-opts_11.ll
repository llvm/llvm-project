; RUN: opt %loadPolly -polly-import-jscop \
; RUN: -polly-import-jscop-postfix=transformed \
; RUN: -polly-pattern-matching-based-opts=true \
; RUN: -polly-target-throughput-vector-fma=1 \
; RUN: -polly-target-latency-vector-fma=8 \
; RUN: -polly-target-1st-cache-level-associativity=8 \
; RUN: -polly-target-2nd-cache-level-associativity=8 \
; RUN: -polly-target-1st-cache-level-size=32768 \
; RUN: -polly-target-vector-register-bitwidth=256 \
; RUN: -polly-target-2nd-cache-level-size=262144 \
; RUN: -polly-opt-isl -debug \
; RUN: -polly-tc-opt=true -disable-output < %s 2>&1 \
; RUN: | FileCheck %s
; REQUIRES: asserts
;
; Check that the pattern matching detects the matrix multiplication pattern
; in case scalar memory accesses were replaced by accesses to newly created
; arrays.
;
; CHECK: The tensor contraction pattern was detected
; CHECK: The matrix multiplication pattern was detected
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define void @kernel_gemm(i32 %ni, i32 %nj, i32 %nk, double %A, [1024 x double]* %B, [1024 x double]* %C) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc16, %entry.split
  %indvars.iv35 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next36, %for.inc16 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc13, %for.cond1.preheader
  %indvars.iv32 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next33, %for.inc13 ]
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.cond4.preheader
  %indvars.iv = phi i64 [ 0, %for.cond4.preheader ], [ %indvars.iv.next, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [1024 x double], [1024 x double]* %B, i64 %indvars.iv, i64 %indvars.iv32
  %tmp = load double, double* %arrayidx8, align 8
  %mul = fmul double %tmp, %A
  %arrayidx12 = getelementptr inbounds [1024 x double], [1024 x double]* %C, i64 %indvars.iv35, i64 %indvars.iv32
  %tmp1 = load double, double* %arrayidx12, align 8
  %add = fadd double %tmp1, %mul
  store double %add, double* %arrayidx12, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.body6, label %for.inc13

for.inc13:                                        ; preds = %for.body6
  %indvars.iv.next33 = add nuw nsw i64 %indvars.iv32, 1
  %exitcond34 = icmp ne i64 %indvars.iv.next33, 1024
  br i1 %exitcond34, label %for.cond4.preheader, label %for.inc16

for.inc16:                                        ; preds = %for.inc13
  %indvars.iv.next36 = add nuw nsw i64 %indvars.iv35, 1
  %exitcond37 = icmp ne i64 %indvars.iv.next36, 1024
  br i1 %exitcond37, label %for.cond1.preheader, label %for.end18

for.end18:                                        ; preds = %for.inc16
  ret void
}
