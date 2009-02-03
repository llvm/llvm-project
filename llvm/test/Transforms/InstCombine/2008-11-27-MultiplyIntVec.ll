; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep mul

define <2 x i8> @f(<2 x i8> %x) {
  %A = mul <2 x i8> %x, <i8 1, i8 1>
  ret <2 x i8> %A
}

define <2 x i8> @g(<2 x i8> %x) {
  %A = mul <2 x i8> %x, <i8 -1, i8 -1>
  ret <2 x i8> %A
}
