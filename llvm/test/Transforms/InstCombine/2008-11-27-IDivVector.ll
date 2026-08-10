; RUN: opt < %s -passes=instcombine -S | not grep div

define <2 x i8> @f(<2 x i8> %x) {
  %A = udiv <2 x i8> %x, <i8 1, i8 1>
  ret <2 x i8> %A
}

define <2 x i8> @g(<2 x i8> %x) {
  %A = sdiv <2 x i8> %x, <i8 1, i8 1>
  ret <2 x i8> %A
}
