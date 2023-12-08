; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: combine(r{{[0-9]+}}.l,r{{[0-9]+}}.h)

target triple = "hexagon"

; Function Attrs: nounwind readnone
define i32 @f0(i64 %a0) #0 {
b0:
  %v0 = lshr i64 %a0, 16
  %v1 = trunc i64 %v0 to i32
  ret i32 %v1
}

attributes #0 = { nounwind readnone }
