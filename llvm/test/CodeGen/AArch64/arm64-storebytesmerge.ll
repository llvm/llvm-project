; RUN: llc  -mtriple=aarch64-linux-gnu -enable-misched=false < %s | FileCheck %s

;target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
;target triple = "aarch64--linux-gnu"


; CHECK-LABEL: test
; CHECK: str     x30, [sp, #-16]!
; CHECK: adrp    x8, q   
; CHECK: ldr     x8, [x8, :lo12:q]
; CHECK: stp     xzr, xzr, [x8] 
; CHECK: bl f

@q = external dso_local unnamed_addr global ptr, align 8

; Function Attrs: nounwind
define void @test() local_unnamed_addr #0 {
entry:
  br label %for.body453.i

for.body453.i:                                    ; preds = %for.body453.i, %entry
  br i1 undef, label %for.body453.i, label %for.end705.i

for.end705.i:                                     ; preds = %for.body453.i
  %0 = load ptr, ptr @q, align 8
  store <2 x i16> zeroinitializer, ptr %0, align 2
  %1 = getelementptr i16, ptr %0, i64 2
  store <2 x i16> zeroinitializer, ptr %1, align 2
  %2 = getelementptr i16, ptr %0, i64 4
  store <2 x i16> zeroinitializer, ptr %2, align 2
  %3 = getelementptr i16, ptr %0, i64 6
  store <2 x i16> zeroinitializer, ptr %3, align 2
  call void @f() #2
  unreachable
}

declare void @f() local_unnamed_addr #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a57" "target-features"="+crc,+crypto,+fp-armv8,+neon" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a57" "target-features"="+crc,+crypto,+fp-armv8,+neon" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }
