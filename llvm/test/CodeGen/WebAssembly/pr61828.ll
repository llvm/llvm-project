; RUN: llc < %s -mattr=+simd128 -mtriple=wasm64

define void @foo(i64 %i0, i64 %i1, ptr %p) {
  %B4 = urem i64 %i0, %i0
  %B5 = udiv i64 %i1, %B4
  %I = insertelement <4 x float> <float 0.25, float 0.25, float 0.25, float 0.25>, float 0.5, i64 %B5
  store <4 x float> %I, ptr %p
  ret void
}
