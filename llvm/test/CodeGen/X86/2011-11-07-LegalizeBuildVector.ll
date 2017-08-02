; RUN: llc < %s -mtriple=i686-- -mattr=+avx | FileCheck %s

; We don't really care what this outputs; just make sure it's somewhat sane.
; CHECK: legalize_test
; CHECK: vmovups
define void @legalize_test(i32 %x, <8 x i32>* %p) nounwind {
entry:
  %t1 = insertelement <8 x i32> <i32 undef, i32 undef, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>, i32 %x, i32 0
  %t2 = shufflevector <8 x i32> %t1, <8 x i32> zeroinitializer, <8 x i32> <i32 0, i32 9, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %int2float = sitofp <8 x i32> %t2 to <8 x float>
  %blendAsInt.i821 = bitcast <8 x float> %int2float to <8 x i32>
  store <8 x i32> %blendAsInt.i821, <8 x i32>* %p, align 4
  ret void
}
