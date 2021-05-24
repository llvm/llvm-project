; RUN: llc -mtriple=thumbv6m-eabi -verify-machineinstrs %s -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m--linux-gnueabi"

; Function Attrs: nounwind optsize
define void @foo(i32* nocapture readonly %A) #0 {
entry:
; CHECK-LABEL: foo:
; CHECK: ldm r[[BASE:[0-9]]]!,
; CHECK-NEXT: mov r[[BASE]],
  %0 = load i32, i32* %A, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i32 1
  %1 = load i32, i32* %arrayidx1, align 4
  %call = tail call i32 @bar(i32 %0, i32 %1, i32 %0, i32 %1) #2
  %call2 = tail call i32 @bar(i32 %0, i32 %1, i32 %0, i32 %1) #2
  ret void
}

; Function Attrs: optsize
declare i32 @bar(i32, i32, i32, i32) #1

attributes #0 = { nounwind optsize "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { optsize "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind optsize }
