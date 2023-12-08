; RUN: llc  -march=hexagon  -O3 -hexagon-small-data-threshold=0 -disable-hexagon-misched < %s | FileCheck %s
; CHECK-LABEL: f0
; CHECK-DAG: [[REG:r[0-9]+]] = add
; CHECK-DAG: memw(##g0) = [[REG]].new

@g0 = external global i32, align 8

; Function Attrs: nounwind
define void @f0(i32 %a0) #0 {
b0:
  %v0 = add i32 %a0, 1
  store i32 %v0, ptr @g0, align 4
  ret void
}

attributes #0 = { nounwind }
