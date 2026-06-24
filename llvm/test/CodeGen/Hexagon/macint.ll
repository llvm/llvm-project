; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; Check that we generate integer multiply accumulate.

; CHECK: r{{[0-9]+}} {{\+|\-}}= mpyi(r{{[0-9]+}},

define i32 @f0(ptr %a0, ptr %a1) #0 {
b0:
  %v0 = load i32, ptr %a0, align 4
  %v1 = udiv i32 %v0, 10000
  %v2 = urem i32 %v1, 10
  store i32 %v2, ptr %a1, align 4
  ret i32 0
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
