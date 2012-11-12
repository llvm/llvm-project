; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vmovddup %ymm
define <4 x i64> @A(<4 x i64> %a) {
  %c = shufflevector <4 x i64> %a, <4 x i64> undef, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  ret <4 x i64> %c
}

; CHECK: vmovddup (%
define <4 x i64> @B(<4 x i64>* %ptr) {
  %a = load <4 x i64>* %ptr
  %c = shufflevector <4 x i64> %a, <4 x i64> undef, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  ret <4 x i64> %c
}
