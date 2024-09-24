; ModuleID = 'bugpoint-reduced-instructions.bc'
; RUN: llc -O2 -o - %s -verify-machineinstrs | FileCheck %s
source_filename = "bugpoint-output-9ad75f8.bc"
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define hidden void @func() local_unnamed_addr #0 {
entry:
  br i1 undef, label %land.lhs.true, label %if.end

; CHECK: # %land.lhs.true
; Test updated due D63152 where any load/store prevents shrink-wrapping
; CHECK-NEXT: bclr
; CHECK-NEXT: # %if.end4
land.lhs.true:                                    ; preds = %entry
  br i1 undef, label %return, label %if.end4

if.end:                                           ; preds = %entry
  %cmp = icmp ne ptr @ptr1, null
  br i1 %cmp, label %if.end4, label %return

if.end4:                                          ; preds = %if.end, %land.lhs.true
  %call5 = tail call ptr @test_fun(ptr nonnull @ptr2, ptr null) #7
  unreachable

return:                                           ; preds = %if.end, %land.lhs.true
  ret void
}

declare extern_weak signext i32 @ptr1(ptr, ptr, ptr, ptr)
declare ptr @test_fun(ptr, ptr) noreturn

declare hidden void @ptr2(ptr nocapture readnone)

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="ppc64le" "target-features"="+altivec,+bpermd,+crypto,+direct-move,+extdiv,+power8-vector,+vsx" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nobuiltin nounwind }
