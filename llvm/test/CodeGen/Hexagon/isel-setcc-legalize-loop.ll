; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that we scalarize the comparison. This testcase used to loop forever
; due to the repeated split-widen operations in legalizing SETCC.

; CHECK: fred:
; CHECK: sfcmp.gt
; CHECK: vinsert

define <32 x i32> @fred(<32 x i32> %a0, <32 x i32> %a1) #0 {
b0:
  %v0 = bitcast <32 x i32> %a0 to <32 x float>
  %v1 = bitcast <32 x i32> %a1 to <32 x float>
  %v2 = fcmp ogt <32 x float> %v0, %v1
  %v3 = select <32 x i1> %v2, <32 x float> zeroinitializer, <32 x float> %v0
  %v4 = bitcast <32 x float> %v3 to <32 x i32>
  ret <32 x i32> %v4
}

attributes #0 = { nounwind "target-cpu"="hexagonv66" "target-features"="+hvxv66,+hvx-length128b" }
