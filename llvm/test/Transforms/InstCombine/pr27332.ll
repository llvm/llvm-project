; RUN: opt -passes=instcombine -S -o - < %s | FileCheck %s
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)

define <4 x i1> @test1(<4 x float> %V) {
entry:
  %abs = call <4 x float> @llvm.fabs.v4f32(<4 x float> %V)
  %cmp = fcmp olt <4 x float> %abs, zeroinitializer
  ret <4 x i1> %cmp
}
; CHECK-LABEL: define <4 x i1> @test1(
; CHECK:   ret <4 x i1> zeroinitializer

declare float @fabsf()

define i1 @test2() {
  %call = call float @fabsf()
  %cmp = fcmp olt float %call, 0.000000e+00
  ret i1 %cmp
}
; CHECK-LABEL: define i1 @test2(
; CHECK:  %[[call:.*]] = call float @fabsf()
; CHECK:  %[[cmp:.*]] = fcmp olt float %[[call]], 0.000000e+00
; CHECK:  ret i1 %[[cmp]]
