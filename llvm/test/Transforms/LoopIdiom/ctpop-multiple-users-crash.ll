; RUN: opt -passes=loop-idiom -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios8.0.0"

; When we replace the precondition with a ctpop, we need to ensure
; that only the first branch reads the ctpop.  The store prior
; to that should continue to read from the original compare.

; CHECK: %tobool.5 = icmp ne i32 %num, 0
; CHECK: store i1 %tobool.5, ptr %ptr

define internal fastcc i32 @num_bits_set(i32 %num, ptr %ptr) #1 {
entry:
  %tobool.5 = icmp ne i32 %num, 0
  store i1 %tobool.5, ptr %ptr
  br i1 %tobool.5, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %count.07 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %num.addr.06 = phi i32 [ %num, %for.body.lr.ph ], [ %and, %for.body ]
  %sub = add i32 %num.addr.06, -1
  %and = and i32 %sub, %num.addr.06
  %inc = add nsw i32 %count.07, 1
  %tobool = icmp ne i32 %and, 0
  br i1 %tobool, label %for.body, label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %count.0.lcssa = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  ret i32 %count.0.lcssa
}
