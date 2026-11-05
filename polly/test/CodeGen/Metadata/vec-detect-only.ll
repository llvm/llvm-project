; RUN: opt %loadNPMPolly \
; RUN:   '-passes=polly-custom<detect;print-detect;scops;print-scops>' \
; RUN:   -plugin-arg=Polly,-polly-detect-for-vectorization \
; RUN:   -plugin-arg=Polly,-polly-process-unprofitable=false -disable-output \
; RUN:   < %s | FileCheck %s --check-prefix=DETECTED
; RUN: opt %loadNPMPolly \
; RUN:   '-passes=polly-custom<detect;print-detect;scops;print-scops>' \
; RUN:   -plugin-arg=Polly,-polly-process-unprofitable=false -disable-output \
; RUN:   < %s | FileCheck %s --check-prefix=NOT-DETECTED

; Verify that -polly-detect-for-vectorization bypasses profitability checks
; for a low-compute loop.
; DETECTED-LABEL: Detected Scops in Function simple_add
; DETECTED: Valid Region for Scop:
; DETECTED: Statements {

; The same loop is rejected by default profitability checks without the option.
; NOT-DETECTED-LABEL: Detected Scops in Function simple_add
; NOT-DETECTED-NOT: Valid Region for Scop:
; NOT-DETECTED-NOT: Statements {

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define void @simple_add(ptr noalias %A, ptr noalias %B, i64 %n) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %i = phi i64 [ 0, %entry ], [ %next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %i
  %v = load i32, ptr %arrayidx, align 4
  %inc = add nsw i32 %v, 1
  store i32 %inc, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %i
  store i32 1, ptr %arrayidx2, align 4
  %next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
