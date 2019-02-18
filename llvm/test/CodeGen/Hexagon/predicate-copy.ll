; RUN: llc -march=hexagon -O3 < %s | FileCheck %s

; CHECK: r{{[0-9]+}} = p{{[0-9]+}}

define i1 @f0() #0 {
b0:
  ret i1 false
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
