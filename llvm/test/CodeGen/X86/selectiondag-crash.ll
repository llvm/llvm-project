; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=corei7 < %s

; Check that llc doesn't crash in the attempt to fold a shuffle with
; a splat mask into a constant build_vector.

define <8 x i8> @autogen_SD26299(i8) {
BB:
  %Shuff = shufflevector <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32> zeroinitializer, <8 x i32> <i32 2, i32 undef, i32 6, i32 8, i32 undef, i32 12, i32 14, i32 0>
  %Shuff14 = shufflevector <8 x i32> %Shuff, <8 x i32> %Shuff, <8 x i32> <i32 7, i32 9, i32 11, i32 undef, i32 undef, i32 1, i32 3, i32 5>
  %Shuff35 = shufflevector <8 x i32> %Shuff14, <8 x i32> %Shuff, <8 x i32> <i32 undef, i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13>
  %I42 = insertelement <8 x i32> %Shuff35, i32 88608, i32 0
  %Shuff48 = shufflevector <8 x i32> %Shuff35, <8 x i32> %I42, <8 x i32> <i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 0, i32 2>
  %Tr59 = trunc <8 x i32> %Shuff48 to <8 x i8>
  ret <8 x i8> %Tr59
}
