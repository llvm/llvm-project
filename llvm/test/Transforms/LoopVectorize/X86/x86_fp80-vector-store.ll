; RUN: opt -O3 -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

@x = common global [1024 x x86_fp80] zeroinitializer, align 16

;CHECK-LABEL: @example(
;CHECK-NOT: bitcast x86_fp80* {{%[^ ]+}} to <{{[2-9][0-9]*}} x x86_fp80>*
;CHECK: store
;CHECK: ret void

define void @example() nounwind ssp uwtable {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %conv = sitofp i32 1 to x86_fp80
  %arrayidx = getelementptr inbounds [1024 x x86_fp80], [1024 x x86_fp80]* @x, i64 0, i64 %indvars.iv
  store x86_fp80 %conv, x86_fp80* %arrayidx, align 16
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
