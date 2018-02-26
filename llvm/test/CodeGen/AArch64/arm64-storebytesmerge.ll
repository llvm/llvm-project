; RUN: llc  -mtriple=aarch64-linux-gnu -enable-misched=false < %s | FileCheck %s

;target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
;target triple = "aarch64--linux-gnu"


; CHECK-LABEL: test
; CHECK: str     x30, [sp, #-16]!
; CHECK: adrp    x8, q   
; CHECK: ldr     x8, [x8, :lo12:q]
; CHECK: stp     xzr, xzr, [x8] 
; CHECK: bl f

@q = external unnamed_addr global i16*, align 8

; Function Attrs: nounwind
define void @test() local_unnamed_addr #0 {
entry:
  br label %for.body453.i

for.body453.i:                                    ; preds = %for.body453.i, %entry
  br i1 undef, label %for.body453.i, label %for.end705.i

for.end705.i:                                     ; preds = %for.body453.i
  %0 = load i16*, i16** @q, align 8
  %1 = getelementptr inbounds i16, i16* %0, i64 0
  %2 = bitcast i16* %1 to <2 x i16>*
  store <2 x i16> zeroinitializer, <2 x i16>* %2, align 2
  %3 = getelementptr i16, i16* %1, i64 2
  %4 = bitcast i16* %3 to <2 x i16>*
  store <2 x i16> zeroinitializer, <2 x i16>* %4, align 2
  %5 = getelementptr i16, i16* %1, i64 4
  %6 = bitcast i16* %5 to <2 x i16>*
  store <2 x i16> zeroinitializer, <2 x i16>* %6, align 2
  %7 = getelementptr i16, i16* %1, i64 6
  %8 = bitcast i16* %7 to <2 x i16>*
  store <2 x i16> zeroinitializer, <2 x i16>* %8, align 2
  call void @f() #2
  unreachable
}

declare void @f() local_unnamed_addr #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a57" "target-features"="+crc,+crypto,+fp-armv8,+neon" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a57" "target-features"="+crc,+crypto,+fp-armv8,+neon" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }
