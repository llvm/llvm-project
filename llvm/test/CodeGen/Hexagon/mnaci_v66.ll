; RUN: llc -march=hexagon < %s | FileCheck %s
; This test validates the generation of v66 only instruction M2_mnaci
; CHECK: r{{[0-9]+}} -= mpyi(r{{[0-9]+}},r{{[0-9]+}})

target triple = "hexagon-unknown--elf"

; Function Attrs: norecurse nounwind readnone
define i32 @_Z4testiii(i32 %a, i32 %b, i32 %c) #0 {
entry:
  %mul = mul nsw i32 %c, %b
  %sub = sub nsw i32 %a, %mul
  ret i32 %sub
}

attributes #0 = { norecurse nounwind readnone "target-cpu"="hexagonv66" "target-features"="-hvx,-long-calls" }
