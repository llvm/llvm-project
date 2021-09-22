; RUN: opt -instcombine -S < %s | FileCheck %s

define void @test(<4 x float> *%in_ptr, <4 x float> *%out_ptr) {
  %A = load <4 x float>, <4 x float>* %in_ptr, align 16
  %B = shufflevector <4 x float> %A, <4 x float> undef, <4 x i32> <i32 0, i32 0, i32 undef, i32 undef>
  %C = shufflevector <4 x float> %B, <4 x float> %A, <4 x i32> <i32 0, i32 1, i32 4, i32 undef>
  %D = shufflevector <4 x float> %C, <4 x float> %A, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
; CHECK:  %D = shufflevector <4 x float> %A, <4 x float> poison, <4 x i32> zeroinitializer
  store <4 x float> %D, <4 x float> *%out_ptr
  ret void
}
