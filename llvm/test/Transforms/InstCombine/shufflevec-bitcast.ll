; RUN: opt < %s -instcombine -S | FileCheck %s

define void @test(<16 x i8> %w, i32* %o1, float* %o2) {

; CHECK:       %v.bc = bitcast <16 x i8> %w to <4 x i32>
; CHECK-NEXT:  %v.extract = extractelement <4 x i32> %v.bc, i32 3
; CHECK-NEXT:  %v.bc{{[0-9]*}} = bitcast <16 x i8> %w to <4 x float>
; CHECK-NEXT:  %v.extract{{[0-9]*}} = extractelement <4 x float> %v.bc{{[0-9]*}}, i32 3

  %v = shufflevector <16 x i8> %w, <16 x i8> undef, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  %f = bitcast <4 x i8> %v to float
  %i = bitcast <4 x i8> %v to i32
  store i32 %i, i32* %o1, align 4
  store float %f, float* %o2, align 4
  ret void
}
