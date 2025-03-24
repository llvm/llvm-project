; RUN: llc -mtriple=hexagon -verify-machineinstrs < %s | FileCheck %s
; CHECK: extractu(r1,#31,#0)

; In the IR this was an extract of 31 bits starting at position 32 in r1:0.
; When mapping it to an extract from r1, the offset was not reset to 0, and
; we had "extractu(r1,#31,#32)".

target triple = "hexagon"

@g0 = global double zeroinitializer, align 8

define hidden i32 @fred(ptr %a0, i32 %a1, ptr %a2, ptr %a3) #0 {
b4:
  br label %b5

b5:                                               ; preds = %b5, %b4
  %v6 = call double @fabs(double undef) #1
  store double %v6, ptr @g0, align 8
  br label %b5
}

declare double @fabs(double) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="-hvx,-long-calls" }
attributes #1 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="-hvx,-long-calls" }
