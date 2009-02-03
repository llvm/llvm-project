; RUN: llvm-as < %s | opt -instcombine -disable-output

define <3 x i8> @f(<3 x i8> %i) {
  %A = sdiv <3 x i8> %i, %i
  ret <3 x i8> %A
}
