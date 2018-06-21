; RUN: llc < %s
target triple = "armv7-none-linux-gnueabi"

define <3 x i64> @shuffle(i1 %dec1, i1 %dec0, <3 x i64> %b) {
entry:
  %.sink = select i1 %dec1, <3 x i64> %b, <3 x i64> zeroinitializer
  %.sink15 = select i1 %dec0, <3 x i64> %b, <3 x i64> zeroinitializer
  %vecinit7 = shufflevector <3 x i64> %.sink, <3 x i64> %.sink15, <3 x i32> <i32 0, i32 4, i32 undef>
  ret <3 x i64> %vecinit7
}
