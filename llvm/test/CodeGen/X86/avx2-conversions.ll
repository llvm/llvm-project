; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; CHECK: trunc4
; CHECK: vpermd
; CHECK-NOT: vinsert
; CHECK: ret
define <4 x i32> @trunc4(<4 x i64> %A) nounwind {
  %B = trunc <4 x i64> %A to <4 x i32>
  ret <4 x i32>%B
}

; CHECK: trunc8
; CHECK: vpshufb
; CHECK-NOT: vinsert
; CHECK: ret

define <8 x i16> @trunc8(<8 x i32> %A) nounwind {
  %B = trunc <8 x i32> %A to <8 x i16>
  ret <8 x i16>%B
}

; CHECK: sext4
; CHECK: vpmovsxdq
; CHECK-NOT: vinsert
; CHECK: ret
define <4 x i64> @sext4(<4 x i32> %A) nounwind {
  %B = sext <4 x i32> %A to <4 x i64>
  ret <4 x i64>%B
}

; CHECK: sext8
; CHECK: vpmovsxwd
; CHECK-NOT: vinsert
; CHECK: ret
define <8 x i32> @sext8(<8 x i16> %A) nounwind {
  %B = sext <8 x i16> %A to <8 x i32>
  ret <8 x i32>%B
}

; CHECK: zext4
; CHECK: vpmovzxdq
; CHECK-NOT: vinsert
; CHECK: ret
define <4 x i64> @zext4(<4 x i32> %A) nounwind {
  %B = zext <4 x i32> %A to <4 x i64>
  ret <4 x i64>%B
}

; CHECK: zext8
; CHECK: vpmovzxwd
; CHECK-NOT: vinsert
; CHECK: ret
define <8 x i32> @zext8(<8 x i16> %A) nounwind {
  %B = zext <8 x i16> %A to <8 x i32>
  ret <8 x i32>%B
}
; CHECK: zext_8i8_8i32
; CHECK: vpmovzxwd
; CHECK: vpand
; CHECK: ret
define <8 x i32> @zext_8i8_8i32(<8 x i8> %A) nounwind {
  %B = zext <8 x i8> %A to <8 x i32>  
  ret <8 x i32>%B
}




