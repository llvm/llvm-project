; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; Rudimentary test of fcmp/icmp on vectors returning vector of bool

; CHECK: @ffoo
; CHECK: fcmp olt <4 x float> %a, %b
define <4 x i1> @ffoo(<4 x float> %a, <4 x float> %b) nounwind {
entry:
  %cmp = fcmp olt <4 x float> %a, %b		; <4 x i1> [#uses=1]
  ret <4 x i1> %cmp
}

; CHECK: @ifoo
; CHECK: icmp slt <4 x i32> %a, %b
define <4 x i1> @ifoo(<4 x i32> %a, <4 x i32> %b) nounwind {
entry:
  %cmp = icmp slt <4 x i32> %a, %b		; <4 x i1> [#uses=1]
  ret <4 x i1> %cmp
}
