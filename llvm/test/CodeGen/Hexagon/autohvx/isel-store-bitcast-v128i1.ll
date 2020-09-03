; RUN: llc -march=hexagon < %s | FileCheck %s

; Primarily check if this compiles without failing.

; CHECK-LABEL: fred:
; CHECK: memd
define void @fred(<128 x i8> %a0, <128 x i8> %a1, i128* %a2) #0 {
  %v0 = icmp eq <128 x i8> %a0, %a1
  %v1 = bitcast <128 x i1> %v0 to i128
  store i128 %v1, i128* %a2, align 16
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv66" "target-features"="+v66,+hvx,+hvxv66,+hvx-length128b" }

