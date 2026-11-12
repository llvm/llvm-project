; RUN: opt < %s -passes=reassociate -S | FileCheck %s

define <4 x float> @test1() {
; CHECK-LABEL: test1
; CHECK-NEXT: %tmp1 = fsub <4 x float> zeroinitializer, zeroinitializer
; CHECK-NEXT: %tmp2 = fmul <4 x float> %tmp1, zeroinitializer
; CHECK-NEXT: ret <4 x float> %tmp2

  %tmp1 = fsub <4 x float> zeroinitializer, zeroinitializer
  %tmp2 = fmul <4 x float> zeroinitializer, %tmp1
  ret <4 x float> %tmp2
}
