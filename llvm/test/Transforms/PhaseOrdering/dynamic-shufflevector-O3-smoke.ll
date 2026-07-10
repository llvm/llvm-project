; RUN: opt -O3 -S -disable-output < %s
; Exercises dynamic-mask shuffles through the full O3 pipeline; this test
; passes if opt does not crash.
define <8 x i32> @f(<4 x i32> %a, <4 x i32> %b, <8 x i8> %m, <4 x i32> %c) {
  %s1 = shufflevector <4 x i32> %a, <4 x i32> %b, <8 x i8> %m
  %s2 = shufflevector <4 x i32> %c, <4 x i32> poison, <8 x i8> %m
  %r = add <8 x i32> %s1, %s2
  ret <8 x i32> %r
}
define <4 x i32> @same_len(<4 x i32> %a, <4 x i32> %b, <4 x i64> %m) {
  %s = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i64> %m
  ret <4 x i32> %s
}
define <vscale x 4 x i32> @scalable(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %m) {
  %s = shufflevector <vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %m
  ret <vscale x 4 x i32> %s
}
