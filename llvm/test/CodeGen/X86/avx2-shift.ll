; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; CHECK: variable_shl0
; CHECK: psllvd
; CHECK: ret
define <4 x i32> @variable_shl0(<4 x i32> %x, <4 x i32> %y) {
  %k = shl <4 x i32> %x, %y
  ret <4 x i32> %k
}
; CHECK: variable_shl1
; CHECK: psllvd
; CHECK: ret
define <8 x i32> @variable_shl1(<8 x i32> %x, <8 x i32> %y) {
  %k = shl <8 x i32> %x, %y
  ret <8 x i32> %k
}
; CHECK: variable_shl2
; CHECK: psllvq
; CHECK: ret
define <2 x i64> @variable_shl2(<2 x i64> %x, <2 x i64> %y) {
  %k = shl <2 x i64> %x, %y
  ret <2 x i64> %k
}
; CHECK: variable_shl3
; CHECK: psllvq
; CHECK: ret
define <4 x i64> @variable_shl3(<4 x i64> %x, <4 x i64> %y) {
  %k = shl <4 x i64> %x, %y
  ret <4 x i64> %k
}
; CHECK: variable_srl0
; CHECK: psrlvd
; CHECK: ret
define <4 x i32> @variable_srl0(<4 x i32> %x, <4 x i32> %y) {
  %k = lshr <4 x i32> %x, %y
  ret <4 x i32> %k
}
; CHECK: variable_srl1
; CHECK: psrlvd
; CHECK: ret
define <8 x i32> @variable_srl1(<8 x i32> %x, <8 x i32> %y) {
  %k = lshr <8 x i32> %x, %y
  ret <8 x i32> %k
}
; CHECK: variable_srl2
; CHECK: psrlvq
; CHECK: ret
define <2 x i64> @variable_srl2(<2 x i64> %x, <2 x i64> %y) {
  %k = lshr <2 x i64> %x, %y
  ret <2 x i64> %k
}
; CHECK: variable_srl3
; CHECK: psrlvq
; CHECK: ret
define <4 x i64> @variable_srl3(<4 x i64> %x, <4 x i64> %y) {
  %k = lshr <4 x i64> %x, %y
  ret <4 x i64> %k
}

; CHECK: variable_sra0
; CHECK: vpsravd
; CHECK: ret
define <4 x i32> @variable_sra0(<4 x i32> %x, <4 x i32> %y) {
  %k = ashr <4 x i32> %x, %y
  ret <4 x i32> %k
}
; CHECK: variable_sra1
; CHECK: vpsravd
; CHECK: ret
define <8 x i32> @variable_sra1(<8 x i32> %x, <8 x i32> %y) {
  %k = ashr <8 x i32> %x, %y
  ret <8 x i32> %k
}

;;; Shift left
; CHECK: vpslld
define <8 x i32> @vshift00(<8 x i32> %a) nounwind readnone {
  %s = shl <8 x i32> %a, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32
2>
  ret <8 x i32> %s
}

; CHECK: vpsllw
define <16 x i16> @vshift01(<16 x i16> %a) nounwind readnone {
  %s = shl <16 x i16> %a, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  ret <16 x i16> %s
}

; CHECK: vpsllq
define <4 x i64> @vshift02(<4 x i64> %a) nounwind readnone {
  %s = shl <4 x i64> %a, <i64 2, i64 2, i64 2, i64 2>
  ret <4 x i64> %s
}

;;; Logical Shift right
; CHECK: vpsrld
define <8 x i32> @vshift03(<8 x i32> %a) nounwind readnone {
  %s = lshr <8 x i32> %a, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32
2>
  ret <8 x i32> %s
}

; CHECK: vpsrlw
define <16 x i16> @vshift04(<16 x i16> %a) nounwind readnone {
  %s = lshr <16 x i16> %a, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  ret <16 x i16> %s
}

; CHECK: vpsrlq
define <4 x i64> @vshift05(<4 x i64> %a) nounwind readnone {
  %s = lshr <4 x i64> %a, <i64 2, i64 2, i64 2, i64 2>
  ret <4 x i64> %s
}

;;; Arithmetic Shift right
; CHECK: vpsrad
define <8 x i32> @vshift06(<8 x i32> %a) nounwind readnone {
  %s = ashr <8 x i32> %a, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32
2>
  ret <8 x i32> %s
}

; CHECK: vpsraw
define <16 x i16> @vshift07(<16 x i16> %a) nounwind readnone {
  %s = ashr <16 x i16> %a, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  ret <16 x i16> %s
}

; CHECK: variable_sra0_load
; CHECK: vpsravd (%
; CHECK: ret
define <4 x i32> @variable_sra0_load(<4 x i32> %x, <4 x i32>* %y) {
  %y1 = load <4 x i32>* %y
  %k = ashr <4 x i32> %x, %y1
  ret <4 x i32> %k
}

; CHECK: variable_sra1_load
; CHECK: vpsravd (%
; CHECK: ret
define <8 x i32> @variable_sra1_load(<8 x i32> %x, <8 x i32>* %y) {
  %y1 = load <8 x i32>* %y
  %k = ashr <8 x i32> %x, %y1
  ret <8 x i32> %k
}

; CHECK: variable_shl0_load
; CHECK: vpsllvd (%
; CHECK: ret
define <4 x i32> @variable_shl0_load(<4 x i32> %x, <4 x i32>* %y) {
  %y1 = load <4 x i32>* %y
  %k = shl <4 x i32> %x, %y1
  ret <4 x i32> %k
}
; CHECK: variable_shl1_load
; CHECK: vpsllvd (%
; CHECK: ret
define <8 x i32> @variable_shl1_load(<8 x i32> %x, <8 x i32>* %y) {
  %y1 = load <8 x i32>* %y
  %k = shl <8 x i32> %x, %y1
  ret <8 x i32> %k
}
; CHECK: variable_shl2_load
; CHECK: vpsllvq (%
; CHECK: ret
define <2 x i64> @variable_shl2_load(<2 x i64> %x, <2 x i64>* %y) {
  %y1 = load <2 x i64>* %y
  %k = shl <2 x i64> %x, %y1
  ret <2 x i64> %k
}
; CHECK: variable_shl3_load
; CHECK: vpsllvq (%
; CHECK: ret
define <4 x i64> @variable_shl3_load(<4 x i64> %x, <4 x i64>* %y) {
  %y1 = load <4 x i64>* %y
  %k = shl <4 x i64> %x, %y1
  ret <4 x i64> %k
}
; CHECK: variable_srl0_load
; CHECK: vpsrlvd (%
; CHECK: ret
define <4 x i32> @variable_srl0_load(<4 x i32> %x, <4 x i32>* %y) {
  %y1 = load <4 x i32>* %y
  %k = lshr <4 x i32> %x, %y1
  ret <4 x i32> %k
}
; CHECK: variable_srl1_load
; CHECK: vpsrlvd (%
; CHECK: ret
define <8 x i32> @variable_srl1_load(<8 x i32> %x, <8 x i32>* %y) {
  %y1 = load <8 x i32>* %y
  %k = lshr <8 x i32> %x, %y1
  ret <8 x i32> %k
}
; CHECK: variable_srl2_load
; CHECK: vpsrlvq (%
; CHECK: ret
define <2 x i64> @variable_srl2_load(<2 x i64> %x, <2 x i64>* %y) {
  %y1 = load <2 x i64>* %y
  %k = lshr <2 x i64> %x, %y1
  ret <2 x i64> %k
}
; CHECK: variable_srl3_load
; CHECK: vpsrlvq (%
; CHECK: ret
define <4 x i64> @variable_srl3_load(<4 x i64> %x, <4 x i64>* %y) {
  %y1 = load <4 x i64>* %y
  %k = lshr <4 x i64> %x, %y1
  ret <4 x i64> %k
}

define <32 x i8> @shl9(<32 x i8> %A) nounwind {
  %B = shl <32 x i8> %A, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <32 x i8> %B
; CHECK: shl9:
; CHECK: vpsllw $3
; CHECK: vpand
; CHECK: ret
}

define <32 x i8> @shr9(<32 x i8> %A) nounwind {
  %B = lshr <32 x i8> %A, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <32 x i8> %B
; CHECK: shr9:
; CHECK: vpsrlw $3
; CHECK: vpand
; CHECK: ret
}

define <32 x i8> @sra_v32i8_7(<32 x i8> %A) nounwind {
  %B = ashr <32 x i8> %A, <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  ret <32 x i8> %B
; CHECK: sra_v32i8_7:
; CHECK: vpxor
; CHECK: vpcmpgtb
; CHECK: ret
}

define <32 x i8> @sra_v32i8(<32 x i8> %A) nounwind {
  %B = ashr <32 x i8> %A, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <32 x i8> %B
; CHECK: sra_v32i8:
; CHECK: vpsrlw $3
; CHECK: vpand
; CHECK: vpxor
; CHECK: vpsubb
; CHECK: ret
}

; CHECK: _sext_v16i16
; CHECK: vpsllw
; CHECK: vpsraw
; CHECK-NOT: vinsertf128
define <16 x i16> @sext_v16i16(<16 x i16> %a) nounwind {
  %b = trunc <16 x i16> %a to <16 x i8>
  %c = sext <16 x i8> %b to <16 x i16>
  ret <16 x i16> %c
}

; CHECK: _sext_v8i32
; CHECK: vpslld
; CHECK: vpsrad
; CHECK-NOT: vinsertf128
define <8 x i32> @sext_v8i32(<8 x i32> %a) nounwind {
  %b = trunc <8 x i32> %a to <8 x i16>
  %c = sext <8 x i16> %b to <8 x i32>
  ret <8 x i32> %c
}
