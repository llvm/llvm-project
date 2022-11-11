; RUN: opt -passes=instcombine -S < %s | FileCheck %s

define void @test(ptr %in_ptr, ptr %out_ptr) {
  %A = load <4 x float>, ptr %in_ptr, align 16
  %B = shufflevector <4 x float> %A, <4 x float> undef, <4 x i32> <i32 0, i32 0, i32 undef, i32 undef>
  %C = shufflevector <4 x float> %B, <4 x float> %A, <4 x i32> <i32 0, i32 1, i32 4, i32 undef>
  %D = shufflevector <4 x float> %C, <4 x float> %A, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
; CHECK:  %D = shufflevector <4 x float> %A, <4 x float> poison, <4 x i32> zeroinitializer
  store <4 x float> %D, ptr %out_ptr
  ret void
}
