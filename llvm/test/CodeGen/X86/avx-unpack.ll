; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vunpckhps
define <8 x float> @unpackhips(<8 x float> %src1, <8 x float> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x float> %src1, <8 x float> %src2, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x float> %shuffle.i
}

; CHECK: vunpckhpd
define <4 x double> @unpackhipd(<4 x double> %src1, <4 x double> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x double> %src1, <4 x double> %src2, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x double> %shuffle.i
}

; CHECK: vunpcklps
define <8 x float> @unpacklops(<8 x float> %src1, <8 x float> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x float> %src1, <8 x float> %src2, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x float> %shuffle.i
}

; CHECK: vunpcklpd
define <4 x double> @unpacklopd(<4 x double> %src1, <4 x double> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x double> %src1, <4 x double> %src2, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x double> %shuffle.i
}

; CHECK-NOT: vunpcklps %ymm
define <8 x float> @unpacklops-not(<8 x float> %src1, <8 x float> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x float> %src1, <8 x float> %src2, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x float> %shuffle.i
}

; CHECK-NOT: vunpcklpd %ymm
define <4 x double> @unpacklopd-not(<4 x double> %src1, <4 x double> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x double> %src1, <4 x double> %src2, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x double> %shuffle.i
}

; CHECK-NOT: vunpckhps %ymm
define <8 x float> @unpackhips-not(<8 x float> %src1, <8 x float> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x float> %src1, <8 x float> %src2, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13>
  ret <8 x float> %shuffle.i
}

; CHECK-NOT: vunpckhpd %ymm
define <4 x double> @unpackhipd-not(<4 x double> %src1, <4 x double> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x double> %src1, <4 x double> %src2, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ret <4 x double> %shuffle.i
}

;;;;
;;;; Unpack versions using the fp unit for int unpacking
;;;;

; CHECK: vunpckhps
define <8 x i32> @unpackhips1(<8 x i32> %src1, <8 x i32> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x i32> %src1, <8 x i32> %src2, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i32> %shuffle.i
}

; CHECK: vunpckhps (%
define <8 x i32> @unpackhips2(<8 x i32>* %src1, <8 x i32>* %src2) nounwind uwtable readnone ssp {
entry:
  %a = load <8 x i32>* %src1
  %b = load <8 x i32>* %src2
  %shuffle.i = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i32> %shuffle.i
}

; CHECK: vunpckhpd
define <4 x i64> @unpackhipd1(<4 x i64> %src1, <4 x i64> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x i64> %src1, <4 x i64> %src2, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x i64> %shuffle.i
}

; CHECK: vunpckhpd (%
define <4 x i64> @unpackhipd2(<4 x i64>* %src1, <4 x i64>* %src2) nounwind uwtable readnone ssp {
entry:
  %a = load <4 x i64>* %src1
  %b = load <4 x i64>* %src2
  %shuffle.i = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x i64> %shuffle.i
}

; CHECK: vunpcklps
define <8 x i32> @unpacklops1(<8 x i32> %src1, <8 x i32> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x i32> %src1, <8 x i32> %src2, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x i32> %shuffle.i
}

; CHECK: vunpcklps (%
define <8 x i32> @unpacklops2(<8 x i32>* %src1, <8 x i32>* %src2) nounwind uwtable readnone ssp {
entry:
  %a = load <8 x i32>* %src1
  %b = load <8 x i32>* %src2
  %shuffle.i = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x i32> %shuffle.i
}

; CHECK: vunpcklpd
define <4 x i64> @unpacklopd1(<4 x i64> %src1, <4 x i64> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x i64> %src1, <4 x i64> %src2, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x i64> %shuffle.i
}

; CHECK: vunpcklpd (%
define <4 x i64> @unpacklopd2(<4 x i64>* %src1, <4 x i64>* %src2) nounwind uwtable readnone ssp {
entry:
  %a = load <4 x i64>* %src1
  %b = load <4 x i64>* %src2
  %shuffle.i = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x i64> %shuffle.i
}

; CHECK: vpunpckhwd
; CHECK: vpunpckhwd
; CHECK: vinsertf128
define <16 x i16> @unpackhwd_undef(<16 x i16> %src1) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <16 x i16> %src1, <16 x i16> %src1, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i16> %shuffle.i
}

; CHECK: vpunpcklwd
; CHECK: vpunpcklwd
; CHECK: vinsertf128
define <16 x i16> @unpacklwd_undef(<16 x i16> %src1) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <16 x i16> %src1, <16 x i16> %src1, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  ret <16 x i16> %shuffle.i
}

; CHECK: vpunpckhbw
; CHECK: vpunpckhbw
; CHECK: vinsertf128
define <32 x i8> @unpackhbw_undef(<32 x i8> %src1, <32 x i8> %src2) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <32 x i8> %src1, <32 x i8> %src1, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  ret <32 x i8> %shuffle.i
}

; CHECK: vpunpcklbw
; CHECK: vpunpcklbw
; CHECK: vinsertf128
define <32 x i8> @unpacklbw_undef(<32 x i8> %src1) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <32 x i8> %src1, <32 x i8> %src1, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55>
  ret <32 x i8> %shuffle.i
}
