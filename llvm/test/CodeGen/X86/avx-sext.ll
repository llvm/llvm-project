; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

define <8 x i32> @sext_8i16_to_8i32(<8 x i16> %A) nounwind uwtable readnone ssp {
;CHECK: sext_8i16_to_8i32
;CHECK: vpmovsxwd

  %B = sext <8 x i16> %A to <8 x i32>
  ret <8 x i32>%B
}

define <4 x i64> @sext_4i32_to_4i64(<4 x i32> %A) nounwind uwtable readnone ssp {
;CHECK: sext_4i32_to_4i64
;CHECK: vpmovsxdq

  %B = sext <4 x i32> %A to <4 x i64>
  ret <4 x i64>%B
}
