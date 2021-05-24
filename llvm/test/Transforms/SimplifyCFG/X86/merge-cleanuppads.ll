; RUN: opt -S -simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

; Function Attrs: uwtable
define void @test1() #0 personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @may_throw(i32 3)
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  tail call void @may_throw(i32 2) #2
  tail call void @may_throw(i32 1) #2
  ret void

ehcleanup:                                        ; preds = %entry
  %cp = cleanuppad within none []
  tail call void @may_throw(i32 2) #2 [ "funclet"(token %cp) ]
  cleanupret from %cp unwind label %ehcleanup2

ehcleanup2:
  %cp2 = cleanuppad within none []
  tail call void @may_throw(i32 1) #2 [ "funclet"(token %cp2) ]
  cleanupret from %cp2 unwind to caller
}

; CHECK-LABEL: define void @test1(
; CHECK: %[[cp:.*]] = cleanuppad within none []
; CHECK: tail call void @may_throw(i32 2) #2 [ "funclet"(token %[[cp]]) ]
; CHECK: tail call void @may_throw(i32 1) #2 [ "funclet"(token %[[cp]]) ]
; CHECK: cleanupret from %[[cp]] unwind to caller

declare void @may_throw(i32) #1

declare i32 @__CxxFrameHandler3(...)

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
