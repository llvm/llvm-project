; RUN: llc -march=hexagon < %s | FileCheck %s
; Check for integer abs instruction.
; CHECK: r{{[0-9]+}} = abs

; Function Attrs: nounwind readnone
define i32 @f0(i32 %a0) #0 {
b0:
  %v0 = icmp slt i32 %a0, 0
  %v1 = sub nsw i32 0, %a0
  %v2 = select i1 %v0, i32 %v1, i32 %a0
  ret i32 %v2
}

attributes #0 = { nounwind readnone }
