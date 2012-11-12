; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

define <8 x i32> @zext_8i16_to_8i32(<8 x i16> %A) nounwind uwtable readnone ssp {
;CHECK: zext_8i16_to_8i32
;CHECK: vpunpckhwd
;CHECK: ret

  %B = zext <8 x i16> %A to <8 x i32>
  ret <8 x i32>%B
}

define <4 x i64> @zext_4i32_to_4i64(<4 x i32> %A) nounwind uwtable readnone ssp {
;CHECK: zext_4i32_to_4i64
;CHECK: vpunpckhdq
;CHECK: ret

  %B = zext <4 x i32> %A to <4 x i64>
  ret <4 x i64>%B
}


define <8 x i32> @zext_8i8_to_8i32(<8 x i8> %z) {
;CHECK: zext_8i8_to_8i32
;CHECK: vpunpckhwd
;CHECK: vpunpcklwd
;CHECK: vinsertf128
;CHECK: ret
  %t = zext <8 x i8> %z to <8 x i32>
  ret <8 x i32> %t
}
