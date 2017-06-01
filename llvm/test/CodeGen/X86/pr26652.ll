; RUN: llc < %s -march=x86
; PR26652

define <2 x i32> @test(<4 x i32> %a, <4 x i32> %b) {
entry:
  %0 = or <4 x i32> %a, %b
  %1 = shufflevector <4 x i32> %0, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  ret <2 x i32> %1
}
