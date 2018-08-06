; RUN: llc -O2 -march=hexagon < %s | FileCheck %s

target triple = "hexagon"

; Function Attrs: nounwind
define i32 @f0(i32* %a0, i32* %a1) #0 {
b0:
; We want to see a ##24576 in combine, not #24576.
; CHECK: combine(#5,##24576)
  %v0 = tail call i32 bitcast (i32 (...)* @f1 to i32 (i32*, i32*, i16, i16)*)(i32* %a0, i32* %a1, i16 24576, i16 5) #0
  ret i32 %v0
}

declare i32 @f1(...)

attributes #0 = { nounwind }
