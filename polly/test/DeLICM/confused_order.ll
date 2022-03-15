; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-print-delicm -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-delicm -disable-output -pass-remarks-missed=polly-delicm < %s 2>&1 | FileCheck %s -check-prefix=REMARKS
;
; ForwardOptree changes the SCoP and may already map some accesses.
; DeLICM must be prepared to encounter implicit reads
; (isOriginalScalarKind()) that occur at the beginning of the SCoP
; to an array (isLatestArrayKind()). Otherwise it may confuse the
; MemoryAccess execution order.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @confused_order(double* nocapture %C, i32 %rows, i32 %cols) {
entry:
  %0 = sext i32 %cols to i64
  %1 = sext i32 %rows to i64
  %cmp108 = icmp slt i64 0, %0
  br i1 undef, label %for.body7.lr.ph, label %for.end103

for.body7.lr.ph:
  br label %for.end103

for.end103:
  %a_dot_b_domain.0.lcssa = phi double [ 0.000000e+00, %entry ], [ undef, %for.body7.lr.ph ]
  %arrayidx107 = getelementptr inbounds double, double* %C, i64 0
  store double %a_dot_b_domain.0.lcssa, double* %arrayidx107
  %cmp109 = icmp slt i64 0, %1
  %or.cond = and i1 %cmp108, %cmp109
  br i1 %or.cond, label %if.then110, label %for.inc116

if.then110:
  %arrayidx114 = getelementptr inbounds double, double* %C, i64 0
  store double %a_dot_b_domain.0.lcssa, double* %arrayidx114
  br label %for.inc116

for.inc116:
  ret void
}


; REMARKS-NOT: load after store of same element in same statement
; CHECK: No modification has been made
