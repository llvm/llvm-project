; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

define <8 x i32> @test1(<8 x i32> %A, <8 x i32> %B) nounwind {
; CHECK: test1:
; CHECK: vpalignr $4
  %C = shufflevector <8 x i32> %A, <8 x i32> %B, <8 x i32> <i32 1, i32 2, i32 3, i32 8, i32 5, i32 6, i32 7, i32 12>
  ret <8 x i32> %C
}

define <8 x i32> @test2(<8 x i32> %A, <8 x i32> %B) nounwind {
; CHECK: test2:
; CHECK: vpalignr $4
  %C = shufflevector <8 x i32> %A, <8 x i32> %B, <8 x i32> <i32 1, i32 2, i32 3, i32 8, i32 5, i32 6, i32 undef, i32 12>
  ret <8 x i32> %C
}

define <8 x i32> @test3(<8 x i32> %A, <8 x i32> %B) nounwind {
; CHECK: test3:
; CHECK: vpalignr $4
  %C = shufflevector <8 x i32> %A, <8 x i32> %B, <8 x i32> <i32 1, i32 undef, i32 3, i32 8, i32 5, i32 6, i32 7, i32 12>
  ret <8 x i32> %C
}
;
define <8 x i32> @test4(<8 x i32> %A, <8 x i32> %B) nounwind {
; CHECK: test4:
; CHECK: vpalignr $8
  %C = shufflevector <8 x i32> %A, <8 x i32> %B, <8 x i32> <i32 10, i32 11, i32 undef, i32 1, i32 14, i32 15, i32 4, i32 5>
  ret <8 x i32> %C
}

define <16 x i16> @test5(<16 x i16> %A, <16 x i16> %B) nounwind {
; CHECK: test5:
; CHECK: vpalignr $6
  %C = shufflevector <16 x i16> %A, <16 x i16> %B, <16 x i32> <i32 3, i32 4, i32 undef, i32 6, i32 7, i32 16, i32 17, i32 18, i32 11, i32 12, i32 13, i32 undef, i32 15, i32 24, i32 25, i32 26>
  ret <16 x i16> %C
}

define <16 x i16> @test6(<16 x i16> %A, <16 x i16> %B) nounwind {
; CHECK: test6:
; CHECK: vpalignr $6
  %C = shufflevector <16 x i16> %A, <16 x i16> %B, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 11, i32 12, i32 13, i32 undef, i32 15, i32 24, i32 25, i32 26>
  ret <16 x i16> %C
}

define <16 x i16> @test7(<16 x i16> %A, <16 x i16> %B) nounwind {
; CHECK: test7:
; CHECK: vpalignr $6
  %C = shufflevector <16 x i16> %A, <16 x i16> %B, <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i16> %C
}

define <32 x i8> @test8(<32 x i8> %A, <32 x i8> %B) nounwind {
; CHECK: test8:
; CHECK: palignr $5
  %C = shufflevector <32 x i8> %A, <32 x i8> %B, <32 x i32> <i32 5, i32 6, i32 7, i32 undef, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 36, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50, i32 51, i32 52>
  ret <32 x i8> %C
}
