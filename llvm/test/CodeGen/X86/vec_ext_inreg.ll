; RUN: llc < %s -march=x86-64 

define <8 x i32> @a(<8 x i32> %a) nounwind {
  %b = trunc <8 x i32> %a to <8 x i16>
  %c = sext <8 x i16> %b to <8 x i32>
  ret <8 x i32> %c
}

define <3 x i32> @b(<3 x i32> %a) nounwind {
  %b = trunc <3 x i32> %a to <3 x i16>
  %c = sext <3 x i16> %b to <3 x i32>
  ret <3 x i32> %c
}

define <1 x i32> @c(<1 x i32> %a) nounwind {
  %b = trunc <1 x i32> %a to <1 x i16>
  %c = sext <1 x i16> %b to <1 x i32>
  ret <1 x i32> %c
}

define <8 x i32> @d(<8 x i32> %a) nounwind {
  %b = trunc <8 x i32> %a to <8 x i16>
  %c = zext <8 x i16> %b to <8 x i32>
  ret <8 x i32> %c
}

define <3 x i32> @e(<3 x i32> %a) nounwind {
  %b = trunc <3 x i32> %a to <3 x i16>
  %c = zext <3 x i16> %b to <3 x i32>
  ret <3 x i32> %c
}

define <1 x i32> @f(<1 x i32> %a) nounwind {
  %b = trunc <1 x i32> %a to <1 x i16>
  %c = zext <1 x i16> %b to <1 x i32>
  ret <1 x i32> %c
}
