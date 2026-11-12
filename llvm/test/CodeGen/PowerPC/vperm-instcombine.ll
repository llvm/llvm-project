; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

define <16 x i8> @foo() nounwind ssp {
; CHECK: @foo
;; Arguments are {0,1,...,15},{16,17,...,31},{30,28,26,...,0}
  %1 = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> <i32 50462976, i32 117835012, i32 185207048, i32 252579084>, <4 x i32> <i32 319951120, i32 387323156, i32 454695192, i32 522067228>, <16 x i8> <i8 30, i8 28, i8 26, i8 24, i8 22, i8 20, i8 18, i8 16, i8 14, i8 12, i8 10, i8 8, i8 6, i8 4, i8 2, i8 0>)
  %2 = bitcast <4 x i32> %1 to <16 x i8>
  ret <16 x i8> %2
;; Revised arguments are {16,17,...31},{0,1,...,15},{1,3,5,...,31}
;; optimized into the following:
; CHECK: ret <16 x i8> <i8 17, i8 19, i8 21, i8 23, i8 25, i8 27, i8 29, i8 31, i8 1, i8 3, i8 5, i8 7, i8 9, i8 11, i8 13, i8 15>
}

declare <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32>, <4 x i32>, <16 x i8>)
