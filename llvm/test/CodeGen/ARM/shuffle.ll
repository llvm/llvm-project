; RUN: llc < %s -mtriple=thumbv7-apple-darwin -relocation-model=pic -disable-fp-elim | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin"

define <8 x i8> @shuf(<8 x i8> %a) nounwind readnone optsize ssp {
entry:
; CHECK: vtbl
  %shuffle = shufflevector <8 x i8> %a, <8 x i8> undef, <8 x i32> <i32 3, i32 1, i32 2, i32 0, i32 4, i32 4, i32 5, i32 0>
  ret <8 x i8> %shuffle
}

define <8 x i8> @shuf2(<8 x i8> %a, <8 x i8> %b) nounwind readnone optsize ssp {
entry:
; CHECK: vtbl
  %shuffle = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 3, i32 1, i32 2, i32 0, i32 4, i32 4, i32 5, i32 8>
  ret <8 x i8> %shuffle
}
