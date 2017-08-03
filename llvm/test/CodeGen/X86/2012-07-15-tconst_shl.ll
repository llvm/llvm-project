; RUN: llc < %s -march=x86-64 -mcpu=corei7 -mattr=+avx2
; make sure that we are not crashing.

define <16 x i32> @autogen_SD34717() {
BB:
  %Shuff7 = shufflevector <16 x i32> zeroinitializer, <16 x i32> zeroinitializer, <16 x i32> <i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 undef, i32 22, i32 24, i32 26, i32 28, i32 30, i32 undef>
  %B9 = lshr <16 x i32> zeroinitializer, %Shuff7
  ret <16 x i32> %B9
}
