; RUN: llc < %s -mtriple=i686-- -mattr=+sse
; PR2598

define <2 x float> @a(<2 x i32> %i) nounwind {
  %r = sitofp <2 x i32> %i to <2 x float> 
  ret <2 x float> %r
}

define <2 x i32> @b(<2 x float> %i) nounwind {
  %r = fptosi <2 x float> %i to <2 x i32> 
  ret <2 x i32> %r
}

