; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -disable-output < %s | \
; RUN: FileCheck %s -check-prefix=KERNEL-IR

; REQUIRES: pollyacc

; Ensure that no dead instructions are emitted between the store and the
; branch instruction of the ScopStmt. At some point, our dead-code-elimination
; did not remove code that was inserted to compute the old (unused) branch
; condition. This code referred to CPU registers and consequently resulted
; in invalid bitcode.

; KERNEL-IR: store i32 0, i32* %polly.access.MemRef_sum_c, align 4
; KERNEL-IR-NEXT: br label %polly.merge

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @kernel_dynprog([50 x [50 x i32]]* %sum_c) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry
  br label %for.body3

for.cond4.for.cond1.loopexit_crit_edge:           ; preds = %for.end
  br label %for.cond1.loopexit

for.cond1.loopexit:                               ; preds = %for.cond4.for.cond1.loopexit_crit_edge
  br i1 undef, label %for.body3, label %for.inc55

for.body3:                                        ; preds = %for.cond1.loopexit, %for.cond1.preheader
  %indvars.iv55 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next56, %for.cond1.loopexit ]
  %indvars.iv.next56 = add nuw nsw i64 %indvars.iv55, 1
  br label %for.body6

for.body6:                                        ; preds = %for.end, %for.body3
  %indvars.iv50 = phi i64 [ 0, %for.body3 ], [ %indvars.iv.next51, %for.end ]
  %arrayidx10 = getelementptr inbounds [50 x [50 x i32]], [50 x [50 x i32]]* %sum_c, i64 %indvars.iv55, i64 %indvars.iv50, i64 %indvars.iv55
  store i32 0, i32* %arrayidx10, align 4
  %cmp1334 = icmp slt i64 %indvars.iv.next56, %indvars.iv50
  br i1 %cmp1334, label %for.body14.lr.ph, label %for.end

for.body14.lr.ph:                                 ; preds = %for.body6
  br label %for.body14

for.body14:                                       ; preds = %for.body14, %for.body14.lr.ph
  %arrayidx32 = getelementptr inbounds [50 x [50 x i32]], [50 x [50 x i32]]* %sum_c, i64 %indvars.iv55, i64 %indvars.iv50, i64 0
  br i1 false, label %for.body14, label %for.cond12.for.end_crit_edge

for.cond12.for.end_crit_edge:                     ; preds = %for.body14
  br label %for.end

for.end:                                          ; preds = %for.cond12.for.end_crit_edge, %for.body6
  %indvars.iv.next51 = add nuw nsw i64 %indvars.iv50, 1
  %lftr.wideiv53 = trunc i64 %indvars.iv.next51 to i32
  %exitcond54 = icmp ne i32 %lftr.wideiv53, 50
  br i1 %exitcond54, label %for.body6, label %for.cond4.for.cond1.loopexit_crit_edge

for.inc55:                                        ; preds = %for.cond1.loopexit
  unreachable
}
