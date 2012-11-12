
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s
; CHECK: build_vector_again
define <4 x i8> @build_vector_again(<16 x i8> %in) nounwind readnone {
entry:
  %out = shufflevector <16 x i8> %in, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK: pmovzxbd
  ret <4 x i8> %out
; CHECK: ret
}
