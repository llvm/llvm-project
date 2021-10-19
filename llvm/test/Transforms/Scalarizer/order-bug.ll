; RUN: opt %s -scalarizer -S -o - | FileCheck %s
; RUN: opt %s -passes='function(scalarizer)' -S -o - | FileCheck %s

; This input caused the scalarizer to replace & erase gathered results when 
; future gathered results depended on them being alive

define dllexport spir_func <4 x i32> @main(float %a) {
entry:
  %i = insertelement <4 x float> undef, float %a, i32 0
  br label %z

y:
; CHECK: %f.upto0 = insertelement <4 x i32> poison, i32 %b.i0, i32 0
; CHECK: %f.upto1 = insertelement <4 x i32> %f.upto0, i32 %b.i0, i32 1
; CHECK: %f.upto2 = insertelement <4 x i32> %f.upto1, i32 %b.i0, i32 2
; CHECK: %f = insertelement <4 x i32> %f.upto2, i32 %b.i0, i32 3
  %f = shufflevector <4 x i32> %b, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %f

z:
; CHECK: %b.i0 = bitcast float %a to i32
  %b = bitcast <4 x float> %i to <4 x i32>
  br label %y
}
