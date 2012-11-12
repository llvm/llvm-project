; RUN: llc < %s -march=x86 -mattr=+avx | FileCheck %s

; CHECK: test
; Test case when creating pshufhw, we incorrectly set the higher order bit
; for an undef,
define void @test(<8 x i16>* %dest, <8 x i16> %in) nounwind {
entry:
; CHECK-NOT: vmovaps
; CHECK: vmovlpd
; CHECK: vpshufhw        $-95
  %0 = load <8 x i16>* %dest
  %1 = shufflevector <8 x i16> %0, <8 x i16> %in, <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 13, i32 undef, i32 14, i32 14>
  store <8 x i16> %1, <8 x i16>* %dest
  ret void
}

; CHECK: test2
; A test case where we shouldn't generate a punpckldq but a pshufd and a pslldq
define void @test2(<4 x i32>* %dest, <4 x i32> %in) nounwind {
entry:
; CHECK-NOT: pslldq
; CHECK: shufps
  %0 = shufflevector <4 x i32> %in, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, <4 x i32> < i32 undef, i32 5, i32 undef, i32 2>
  store <4 x i32> %0, <4 x i32>* %dest
  ret void
}
