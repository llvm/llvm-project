; RUN: llc -O2 < %s | FileCheck %s
; Look for four stores directly via r29.
; CHECK: memd(r29
; CHECK: memd(r29
; CHECK: memd(r29
; CHECK: memd(r29

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon"

; Function Attrs: nounwind
define void @foo() #0 {
entry:
  %t = alloca [4 x [2 x i32]], align 8
  call void @llvm.memset.p0.i32(ptr align 8 %t, i8 0, i32 32, i1 false)
  call void @bar(ptr %t) #1
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memset.p0.i32(ptr nocapture, i8, i32, i1) #1

declare void @bar(ptr) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
