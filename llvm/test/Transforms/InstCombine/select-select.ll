; RUN: opt -instcombine -S < %s | FileCheck %s

; CHECK: @foo1
define float @foo1(float %a) #0 {
; CHECK-NOT: xor
  %b = fcmp ogt float %a, 0.000000e+00
  %c = select i1 %b, float %a, float 0.000000e+00
  %d = fcmp olt float %c, 1.000000e+00
  %f = select i1 %d, float %c, float 1.000000e+00
  ret float %f
}

; CHECK: @foo2
define float @foo2(float %a) #0 {
; CHECK-NOT: xor
  %b = fcmp ogt float %a, 0.000000e+00
  %c = select i1 %b, float %a, float 0.000000e+00
  %d = fcmp olt float %c, 1.000000e+00
  %e = select i1 %b, float %a, float 0.000000e+00
  %f = select i1 %d, float %e, float 1.000000e+00
  ret float %f
}

; CHECK-LABEL: @foo3
define <2 x i32> @foo3(<2 x i1> %vec_bool, i1 %bool, <2 x i32> %V) {
; CHECK: %[[sel0:.*]] = select <2 x i1> %vec_bool, <2 x i32> zeroinitializer, <2 x i32> %V
; CHECK: %[[sel1:.*]] = select i1 %bool, <2 x i32> %[[sel0]], <2 x i32> %V
; CHECK: ret <2 x i32> %[[sel1]]
  %sel0 = select <2 x i1> %vec_bool, <2 x i32> zeroinitializer, <2 x i32> %V
  %sel1 = select i1 %bool, <2 x i32> %sel0, <2 x i32> %V
  ret <2 x i32> %sel1
}

attributes #0 = { nounwind readnone ssp uwtable }
