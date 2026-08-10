; RUN: llc < %s -mtriple=x86_64-- -mcpu=corei7 -debug

; REQUIRES: asserts

; Test that the dag combiner doesn't assert if we try to replace a sequence of two
; v4f32 X86ISD::PSHUFD nodes with a single PSHUFD.


define <4 x float> @test(<4 x float> %V) {
  %1 = shufflevector <4 x float> %V, <4 x float> undef, <4 x i32> <i32 3, i32 0, i32 2, i32 1>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 3, i32 0, i32 2, i32 1>
  ret <4 x float> %2
}

