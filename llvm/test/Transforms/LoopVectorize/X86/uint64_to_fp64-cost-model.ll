; RUN: opt < %s  -loop-vectorize -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx -S -debug-only=loop-vectorize 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"


; CHECK: cost of 4 for VF 1 For instruction:   %conv = uitofp i64 %tmp to double
; CHECK: cost of 5 for VF 2 For instruction:   %conv = uitofp i64 %tmp to double
; CHECK: cost of 10 for VF 4 For instruction:   %conv = uitofp i64 %tmp to double
define void @uint64_to_double_cost(i64* noalias nocapture %a, double* noalias nocapture readonly %b) nounwind {
entry:
  br label %for.body
for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 %indvars.iv
  %tmp = load i64, i64* %arrayidx, align 4
  %conv = uitofp i64 %tmp to double
  %arrayidx2 = getelementptr inbounds double, double* %b, i64 %indvars.iv
  store double %conv, double* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
