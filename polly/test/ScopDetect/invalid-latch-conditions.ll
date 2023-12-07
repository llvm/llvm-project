; RUN: opt %loadPolly                              -polly-process-unprofitable=false -polly-print-detect -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-allow-nonaffine-loops                                   -polly-print-detect -disable-output < %s | FileCheck %s --check-prefix=NALOOPS
; RUN: opt %loadPolly -polly-allow-nonaffine-loops -polly-process-unprofitable=false -polly-print-detect -disable-output < %s | FileCheck %s --check-prefix=PROFIT

; The latch conditions of the outer loop are not affine, thus the loop cannot
; handled by the domain generation and needs to be overapproximated.

; CHECK-NOT:  Valid
; NALOOPS:    Valid Region for Scop: for.body.6 => for.end.45
; PROFIT-NOT: Valid

; ModuleID = '/home/johannes/Downloads/bug.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @kernel_reg_detect(ptr %path) #0 {
entry:
  br label %for.body.6

for.body.6:                                       ; preds = %for.inc.43, %for.body.6, %entry
  %indvars.iv9 = phi i64 [ %indvars.iv.next10, %for.body.6 ], [ 0, %for.inc.43 ], [ 0, %entry ]
  %indvars.iv.next10 = add nuw nsw i64 %indvars.iv9, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next10 to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 6
  br i1 %exitcond, label %for.body.6, label %for.inc.40

for.inc.40:                                       ; preds = %for.inc.40, %for.body.6
  %tmp = load i32, ptr %path, align 4
  store i32 0, ptr %path, align 4
  %exitcond22 = icmp ne i64 0, 6
  br i1 %exitcond22, label %for.inc.40, label %for.inc.43

for.inc.43:                                       ; preds = %for.inc.40
  %exitcond23 = icmp ne i32 0, 10000
  br i1 %exitcond23, label %for.body.6, label %for.end.45

for.end.45:                                       ; preds = %for.inc.43
  ret void
}
