; RUN: llc < %s -mtriple=x86_64-linux -mcpu=core2
; RUN: llc < %s -mtriple=x86_64-win32 -mcpu=core2

define <8 x i32> @a(<8 x i16> %a) nounwind {
  %c = sext <8 x i16> %a to <8 x i32>
  ret <8 x i32> %c
}

;define <3 x i32> @b(<3 x i16> %a) nounwind {
;  %c = sext <3 x i16> %a to <3 x i32>
;  ret <3 x i32> %c
;}

define <1 x i32> @c(<1 x i16> %a) nounwind {
  %c = sext <1 x i16> %a to <1 x i32>
  ret <1 x i32> %c
}

define <8 x i32> @d(<8 x i16> %a) nounwind {
  %c = zext <8 x i16> %a to <8 x i32>
  ret <8 x i32> %c
}

;define <3 x i32> @e(<3 x i16> %a) nounwind {
;  %c = zext <3 x i16> %a to <3 x i32>
;  ret <3 x i32> %c
;}

define <1 x i32> @f(<1 x i16> %a) nounwind {
  %c = zext <1 x i16> %a to <1 x i32>
  ret <1 x i32> %c
}

define <8 x i16> @g(<8 x i32> %a) nounwind {
  %c = trunc <8 x i32> %a to <8 x i16>
  ret <8 x i16> %c
}

define <3 x i16> @h(<3 x i32> %a) nounwind {
  %c = trunc <3 x i32> %a to <3 x i16>
  ret <3 x i16> %c
}

define <1 x i16> @i(<1 x i32> %a) nounwind {
  %c = trunc <1 x i32> %a to <1 x i16>
  ret <1 x i16> %c
}

; PR6438
define void @__OpenCL_math_kernel4_kernel() nounwind {
  %tmp12.i = and <4 x i32> zeroinitializer, <i32 2139095040, i32 2139095040, i32 2139095040, i32 2139095040> ; <<4 x i32>> [#uses=1]
  %cmp13.i = icmp eq <4 x i32> %tmp12.i, <i32 2139095040, i32 2139095040, i32 2139095040, i32 2139095040> ; <<4 x i1>> [#uses=2]
  %cmp.ext14.i = sext <4 x i1> %cmp13.i to <4 x i32> ; <<4 x i32>> [#uses=0]
  %tmp2110.i = and <4 x i1> %cmp13.i, zeroinitializer ; <<4 x i1>> [#uses=0]
  ret void
}
