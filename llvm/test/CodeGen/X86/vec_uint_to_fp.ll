; RUN: llc < %s -march=x86 -mcpu=corei7-avx | FileCheck %s

; Test that we are not lowering uinttofp to scalars
define <4 x float> @test1(<4 x i32> %A) nounwind {
; CHECK: test1:
; CHECK-NOT: cvtsd2ss
; CHECK: ret
  %C = uitofp <4 x i32> %A to <4 x float>
  ret <4 x float> %C
}

