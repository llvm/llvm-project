; RUN: llc -mtriple=hexagon -debug < %s
; REQUIRES: asserts

target triple = "hexagon"

; Function Attrs: nounwind readnone
define i32 @f0(i1 zeroext %a0) #0 {
b0:
  %v0 = select i1 %a0, i32 1, i32 2
  ret i32 %v0
}

attributes #0 = { nounwind readnone }
