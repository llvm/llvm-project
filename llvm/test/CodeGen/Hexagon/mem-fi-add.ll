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
  %0 = bitcast [4 x [2 x i32]]* %t to i8*
  call void @llvm.memset.p0i8.i32(i8* %0, i8 0, i32 32, i32 8, i1 false)
  %arraydecay = getelementptr inbounds [4 x [2 x i32]], [4 x [2 x i32]]* %t, i32 0, i32 0
  call void @bar([2 x i32]* %arraydecay) #1
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) #1

declare void @bar([2 x i32]*) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
