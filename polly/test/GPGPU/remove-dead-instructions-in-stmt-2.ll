; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -disable-output < %s | \
; RUN: FileCheck %s -check-prefix=KERNEL-IR

; REQUIRES: pollyacc

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; KERNEL-IR: store i32 0, ptr addrspace(1) %polly.access.MemRef_sum_c, align 4
; KERNEL-IR-NEXT: br label %polly.merge

define void @kernel_dynprog(ptr %sum_c) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry
  br label %for.body3

for.cond1.loopexit:                               ; preds = %for.end
  %indvars.iv.next49 = add nuw nsw i64 %indvars.iv48, 1
  %exitcond57 = icmp ne i64 %indvars.iv.next56, 49
  br i1 %exitcond57, label %for.body3, label %for.inc55

for.body3:                                        ; preds = %for.cond1.loopexit, %for.cond1.preheader
  %indvars.iv55 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next56, %for.cond1.loopexit ]
  %indvars.iv48 = phi i64 [ 1, %for.cond1.preheader ], [ %indvars.iv.next49, %for.cond1.loopexit ]
  %indvars.iv.next56 = add nuw nsw i64 %indvars.iv55, 1
  %arrayidx10 = getelementptr inbounds [50 x [50 x i32]], ptr %sum_c, i64 %indvars.iv55, i64 %indvars.iv48, i64 %indvars.iv55
  store i32 0, ptr %arrayidx10, align 4
  %cmp1334 = icmp slt i64 %indvars.iv.next56, %indvars.iv48
  br label %for.end

for.end:                                          ; preds = %for.body3
  br label %for.cond1.loopexit

for.inc55:                                        ; preds = %for.cond1.loopexit
  ret void
}
