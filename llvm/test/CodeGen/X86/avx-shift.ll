; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

;;; Shift left
; CHECK: vpslld
; CHECK: vpslld
define <8 x i32> @vshift00(<8 x i32> %a) nounwind readnone {
  %s = shl <8 x i32> %a, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32
2>
  ret <8 x i32> %s
}

; CHECK: vpsllw
; CHECK: vpsllw
define <16 x i16> @vshift01(<16 x i16> %a) nounwind readnone {
  %s = shl <16 x i16> %a, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  ret <16 x i16> %s
}

; CHECK: vpsllq
; CHECK: vpsllq
define <4 x i64> @vshift02(<4 x i64> %a) nounwind readnone {
  %s = shl <4 x i64> %a, <i64 2, i64 2, i64 2, i64 2>
  ret <4 x i64> %s
}

;;; Logical Shift right
; CHECK: vpsrld
; CHECK: vpsrld
define <8 x i32> @vshift03(<8 x i32> %a) nounwind readnone {
  %s = lshr <8 x i32> %a, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32
2>
  ret <8 x i32> %s
}

; CHECK: vpsrlw
; CHECK: vpsrlw
define <16 x i16> @vshift04(<16 x i16> %a) nounwind readnone {
  %s = lshr <16 x i16> %a, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  ret <16 x i16> %s
}

; CHECK: vpsrlq
; CHECK: vpsrlq
define <4 x i64> @vshift05(<4 x i64> %a) nounwind readnone {
  %s = lshr <4 x i64> %a, <i64 2, i64 2, i64 2, i64 2>
  ret <4 x i64> %s
}

;;; Arithmetic Shift right
; CHECK: vpsrad
; CHECK: vpsrad
define <8 x i32> @vshift06(<8 x i32> %a) nounwind readnone {
  %s = ashr <8 x i32> %a, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32
2>
  ret <8 x i32> %s
}

; CHECK: vpsraw
; CHECK: vpsraw
define <16 x i16> @vshift07(<16 x i16> %a) nounwind readnone {
  %s = ashr <16 x i16> %a, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  ret <16 x i16> %s
}

; CHECK: vpsrlw
; CHECK: pand
; CHECK: pxor
; CHECK: psubb
; CHECK: vpsrlw
; CHECK: pand
; CHECK: pxor
; CHECK: psubb
define <32 x i8> @vshift09(<32 x i8> %a) nounwind readnone {
  %s = ashr <32 x i8> %a, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
  ret <32 x i8> %s
}

; CHECK: pxor
; CHECK: pcmpgtb
; CHECK: pcmpgtb
define <32 x i8> @vshift10(<32 x i8> %a) nounwind readnone {
  %s = ashr <32 x i8> %a, <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  ret <32 x i8> %s
}

; CHECK: vpsrlw
; CHECK: pand
; CHECK: vpsrlw
; CHECK: pand
define <32 x i8> @vshift11(<32 x i8> %a) nounwind readnone {
  %s = lshr <32 x i8> %a, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
  ret <32 x i8> %s
}

; CHECK: vpsllw
; CHECK: pand
; CHECK: vpsllw
; CHECK: pand
define <32 x i8> @vshift12(<32 x i8> %a) nounwind readnone {
  %s = shl <32 x i8> %a, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
  ret <32 x i8> %s
}

;;; Support variable shifts
; CHECK: _vshift08
; CHECK: vextractf128 $1
; CHECK: vpslld $23
; CHECK: vextractf128 $1
; CHECK: vpslld $23
define <8 x i32> @vshift08(<8 x i32> %a) nounwind {
  %bitop = shl <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, %a
  ret <8 x i32> %bitop
}

;;; Uses shifts for sign extension
; CHECK: _sext_v16i16
; CHECK: vpsllw
; CHECK: vpsraw
; CHECK: vpsllw
; CHECK: vpsraw
; CHECK: vinsertf128
define <16 x i16> @sext_v16i16(<16 x i16> %a) nounwind {
  %b = trunc <16 x i16> %a to <16 x i8>
  %c = sext <16 x i8> %b to <16 x i16>
  ret <16 x i16> %c
}

; CHECK: _sext_v8i32
; CHECK: vpslld
; CHECK: vpsrad
; CHECK: vpslld
; CHECK: vpsrad
; CHECK: vinsertf128
define <8 x i32> @sext_v8i32(<8 x i32> %a) nounwind {
  %b = trunc <8 x i32> %a to <8 x i16>
  %c = sext <8 x i16> %b to <8 x i32>
  ret <8 x i32> %c
}
