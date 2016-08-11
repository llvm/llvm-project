; RUN: llc -march=hexagon -O0 < %s | FileCheck %s

; CHECK-LABEL: danny:
; CHECK: mux(p0, r1, ##1065353216)

define float @danny(i32 %x, float %f) #0 {
  %t = icmp sgt i32 %x, 0
  %u = select i1 %t, float %f, float 1.0
  ret float %u
}

; CHECK-LABEL: sammy:
; CHECK: mux(p0, ##1069547520, r1)

define float @sammy(i32 %x, float %f) #0 {
  %t = icmp sgt i32 %x, 0
  %u = select i1 %t, float 1.5, float %f
  ret float %u
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }

