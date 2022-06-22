; RUN: llc %s -o - -enable-shrink-wrap=true | FileCheck %s

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i686-pc-linux"

@b = global i32 1, align 4
@a = common global i32 0, align 4

declare void @fn1() #0

; CHECK-LABEL: fn2:
define void @fn2() #0 {
entry:
  %0 = load i32, ptr @b, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end, label %lbl

lbl:                                              ; preds = %if.end, %entry
  store i32 0, ptr @b, align 4
  br label %if.end

if.end:                                           ; preds = %entry, %lbl
  tail call void @fn1()
  %1 = load i32, ptr @b, align 4
  %tobool1 = icmp eq i32 %1, 0
  br i1 %tobool1, label %if.end3, label %lbl

if.end3:                                          ; preds = %if.end
  ret void
}

attributes #0 = { norecurse nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
