; RUN: llc -mtriple=thumbv6m-eabi -verify-machineinstrs %s -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m--linux-gnueabi"

@d = internal unnamed_addr global i32 0, align 4
@c = internal global ptr null, align 4
@e = internal unnamed_addr global ptr null, align 4

; Function Attrs: nounwind optsize
define void @fn1(i32 %x, i32 %y, i32 %z) #0 {
entry:
; CHECK-LABEL: fn1:
; CHECK: stm r[[BASE:[0-9]]]!, {{.*}}
; CHECK-NOT: {{.*}} r[[BASE]]
  %g = alloca i32, align 4
  %h = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %x, ptr %i, align 4
  store i32 %y, ptr %h, align 4
  store i32 %z, ptr %g, align 4
  %.pr = load i32, ptr @d, align 4
  %cmp11 = icmp slt i32 %.pr, 1
  br i1 %cmp11, label %for.inc.lr.ph, label %for.body5

for.inc.lr.ph:                                    ; preds = %entry
  store i32 1, ptr @d, align 4
  br label %for.body5

for.body5:                                        ; preds = %entry, %for.inc.lr.ph, %for.body5
  %f.010 = phi i32 [ %inc7, %for.body5 ], [ 0, %for.inc.lr.ph ], [ 0, %entry ]
  store volatile ptr %g, ptr @c, align 4
  %inc7 = add nsw i32 %f.010, 1
  %exitcond = icmp eq i32 %inc7, 2
  br i1 %exitcond, label %for.end8, label %for.body5

for.end8:                                         ; preds = %for.body5
  store ptr %h, ptr @e, align 4
  ret void
}

attributes #0 = { nounwind optsize "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
