; RUN: llc -mtriple=hexagon -hexagon-small-data-threshold=0 < %s | FileCheck %s
;
; Check that no CONST64's are emitted for a -G0, mv5 compile
; CHECK-NOT: CONST

; Function Attrs: nounwind readnone
define double @f0(double %a0) #0 {
b0:
  %v0 = fmul double %a0, 0x400921FB53C8D4F1
  %v1 = fmul double %v0, %a0
  ret double %v1
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv55" }
