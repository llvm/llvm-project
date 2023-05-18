; RUN: opt -loop-reduce -S < %s | FileCheck %s
; PR18000

target datalayout = "e-i64:64-f80:128-s:64-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = global i32 0, align 4
@b = common global i32 0, align 4
@e = common global i8 0, align 1
@d = common global i32 0, align 4
@c = common global i32 0, align 4
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

; Function Attrs: nounwind optsize uwtable
; CHECK-LABEL: foo
define i32 @foo() {
entry:
  %.pr = load i32, ptr @b, align 4
  %cmp10 = icmp slt i32 %.pr, 1
  br i1 %cmp10, label %for.cond1.preheader.lr.ph, label %entry.for.end9_crit_edge

entry.for.end9_crit_edge:                         ; preds = %entry
  %.pre = load i32, ptr @c, align 4
  br label %for.end9

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %0 = load i32, ptr @a, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %for.cond1.preheader.for.cond1.preheader.split_crit_edge, label %return.loopexit.split

for.cond1.preheader.for.cond1.preheader.split_crit_edge: ; preds = %for.cond1.preheader.lr.ph, %for.inc8
  %1 = phi i32 [ %inc, %for.inc8 ], [ %.pr, %for.cond1.preheader.lr.ph ]
  br label %if.end

; CHECK-LABEL: if.end
if.end:                                           ; preds = %if.end, %for.cond1.preheader.for.cond1.preheader.split_crit_edge

; CHECK: %lsr.iv = phi i32 [ %lsr.iv.next, %if.end ], [ 258, %for.cond1.preheader.for.cond1.preheader.split_crit_edge ]
  %indvars.iv = phi i32 [ 1, %for.cond1.preheader.for.cond1.preheader.split_crit_edge ], [ %indvars.iv.next, %if.end ]

  %2 = phi i8 [ 1, %for.cond1.preheader.for.cond1.preheader.split_crit_edge ], [ %dec, %if.end ]
  %conv7 = mul i32 %indvars.iv, 258
  %shl = and i32 %conv7, 510
  store i32 %shl, ptr @c, align 4

; CHECK: %lsr.iv.next = add nsw i32 %lsr.iv, -258
  %dec = add i8 %2, -1

  %cmp2 = icmp sgt i8 %dec, -1
  %indvars.iv.next = add i32 %indvars.iv, -1
  br i1 %cmp2, label %if.end, label %for.inc8

for.inc8:                                         ; preds = %if.end
  store i32 0, ptr @d, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr @b, align 4
  %cmp = icmp slt i32 %1, 0
  br i1 %cmp, label %for.cond1.preheader.for.cond1.preheader.split_crit_edge, label %for.cond.for.end9_crit_edge

for.cond.for.end9_crit_edge:                      ; preds = %for.inc8
  store i8 %dec, ptr @e, align 1
  br label %for.end9

for.end9:                                         ; preds = %entry.for.end9_crit_edge, %for.cond.for.end9_crit_edge
  %3 = phi i32 [ %.pre, %entry.for.end9_crit_edge ], [ %shl, %for.cond.for.end9_crit_edge ]
  %call = tail call i32 (ptr, ...) @printf(ptr @.str, i32 %3) #2
  br label %return

return.loopexit.split:                            ; preds = %for.cond1.preheader.lr.ph
  store i8 1, ptr @e, align 1
  store i32 0, ptr @d, align 4
  br label %return

return:                                           ; preds = %return.loopexit.split, %for.end9
  %retval.0 = phi i32 [ 0, %for.end9 ], [ 1, %return.loopexit.split ]
  ret i32 %retval.0
}

; Function Attrs: nounwind optsize
declare i32 @printf(ptr nocapture readonly, ...)

