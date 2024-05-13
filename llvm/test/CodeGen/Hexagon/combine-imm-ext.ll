; RUN: llc -O2 -march=hexagon < %s | FileCheck %s

target triple = "hexagon"

; Function Attrs: nounwind
define i32 @f0(ptr %a0, ptr %a1) #0 {
b0:
; We want to see a #-22 in combine, not ##-22.
; CHECK: combine(#5,#-22)
  %v0 = tail call i32 @f1(ptr %a0, ptr %a1, i32 -22, i32 5) #0
  ret i32 %v0
}

declare i32 @f1(...)

attributes #0 = { nounwind }
