; RUN: opt < %s -mtriple=aarch64--linux-gnu -passes="print<cost-model>" 2>&1 -disable-output | FileCheck %s --check-prefix=COST
; RUN: llc < %s -mtriple=aarch64--linux-gnu | FileCheck %s --check-prefix=CODE

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; COST-LABEL: trn1.v8i8
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <8 x i8> %v0, <8 x i8> %v1, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
; CODE-LABEL: trn1.v8i8
; CODE:       trn1 v0.8b, v0.8b, v1.8b
define <8 x i8> @trn1.v8i8(<8 x i8> %v0, <8 x i8> %v1) {
  %tmp0 = shufflevector <8 x i8> %v0, <8 x i8> %v1, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x i8> %tmp0
}

; COST-LABEL: trn2.v8i8
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <8 x i8> %v0, <8 x i8> %v1, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
; CODE-LABEL: trn2.v8i8
; CODE:       trn2 v0.8b, v0.8b, v1.8b
define <8 x i8> @trn2.v8i8(<8 x i8> %v0, <8 x i8> %v1) {
  %tmp0 = shufflevector <8 x i8> %v0, <8 x i8> %v1, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x i8> %tmp0
}

; COST-LABEL: trn1.v16i8
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <16 x i8> %v0, <16 x i8> %v1, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
; CODE-LABEL: trn1.v16i8
; CODE:       trn1 v0.16b, v0.16b, v1.16b
define <16 x i8> @trn1.v16i8(<16 x i8> %v0, <16 x i8> %v1) {
  %tmp0 = shufflevector <16 x i8> %v0, <16 x i8> %v1, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
  ret <16 x i8> %tmp0
}

; COST-LABEL: trn2.v16i8
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <16 x i8> %v0, <16 x i8> %v1, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
; CODE-LABEL: trn2.v16i8
; CODE:       trn2 v0.16b, v0.16b, v1.16b
define <16 x i8> @trn2.v16i8(<16 x i8> %v0, <16 x i8> %v1) {
  %tmp0 = shufflevector <16 x i8> %v0, <16 x i8> %v1, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
  ret <16 x i8> %tmp0
}

; COST-LABEL: trn1.v4i16
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <4 x i16> %v0, <4 x i16> %v1, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
; CODE-LABEL: trn1.v4i16
; CODE:       trn1 v0.4h, v0.4h, v1.4h
define <4 x i16> @trn1.v4i16(<4 x i16> %v0, <4 x i16> %v1) {
  %tmp0 = shufflevector <4 x i16> %v0, <4 x i16> %v1, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x i16> %tmp0
}

; COST-LABEL: trn2.v4i16
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <4 x i16> %v0, <4 x i16> %v1, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
; CODE-LABEL: trn2.v4i16
; CODE:       trn2 v0.4h, v0.4h, v1.4h
define <4 x i16> @trn2.v4i16(<4 x i16> %v0, <4 x i16> %v1) {
  %tmp0 = shufflevector <4 x i16> %v0, <4 x i16> %v1, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x i16> %tmp0
}

; COST-LABEL: trn1.v8i16
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <8 x i16> %v0, <8 x i16> %v1, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
; CODE-LABEL: trn1.v8i16
; CODE:       trn1 v0.8h, v0.8h, v1.8h
define <8 x i16> @trn1.v8i16(<8 x i16> %v0, <8 x i16> %v1) {
  %tmp0 = shufflevector <8 x i16> %v0, <8 x i16> %v1, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x i16> %tmp0
}

; COST-LABEL: trn2.v8i16
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <8 x i16> %v0, <8 x i16> %v1, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
; CODE-LABEL: trn2.v8i16
; CODE:       trn2 v0.8h, v0.8h, v1.8h
define <8 x i16> @trn2.v8i16(<8 x i16> %v0, <8 x i16> %v1) {
  %tmp0 = shufflevector <8 x i16> %v0, <8 x i16> %v1, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x i16> %tmp0
}

; COST-LABEL: trn1.v2i32
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <2 x i32> %v0, <2 x i32> %v1, <2 x i32> <i32 0, i32 2>
; CODE-LABEL: trn1.v2i32
; CODE:       zip1 v0.2s, v0.2s, v1.2s
define <2 x i32> @trn1.v2i32(<2 x i32> %v0, <2 x i32> %v1) {
  %tmp0 = shufflevector <2 x i32> %v0, <2 x i32> %v1, <2 x i32> <i32 0, i32 2>
  ret <2 x i32> %tmp0
}

; COST-LABEL: trn2.v2i32
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <2 x i32> %v0, <2 x i32> %v1, <2 x i32> <i32 1, i32 3>
; CODE-LABEL: trn2.v2i32
; CODE:       zip2 v0.2s, v0.2s, v1.2s
define <2 x i32> @trn2.v2i32(<2 x i32> %v0, <2 x i32> %v1) {
  %tmp0 = shufflevector <2 x i32> %v0, <2 x i32> %v1, <2 x i32> <i32 1, i32 3>
  ret <2 x i32> %tmp0
}

; COST-LABEL: trn1.v4i32
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
; CODE-LABEL: trn1.v4i32
; CODE:       trn1 v0.4s, v0.4s, v1.4s
define <4 x i32> @trn1.v4i32(<4 x i32> %v0, <4 x i32> %v1) {
  %tmp0 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x i32> %tmp0
}

; COST-LABEL: trn2.v4i32
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
; CODE-LABEL: trn2.v4i32
; CODE:       trn2 v0.4s, v0.4s, v1.4s
define <4 x i32> @trn2.v4i32(<4 x i32> %v0, <4 x i32> %v1) {
  %tmp0 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x i32> %tmp0
}

; COST-LABEL: trn1.v2i64
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <2 x i64> %v0, <2 x i64> %v1, <2 x i32> <i32 0, i32 2>
; CODE-LABEL: trn1.v2i64
; CODE:       zip1 v0.2d, v0.2d, v1.2d
define <2 x i64> @trn1.v2i64(<2 x i64> %v0, <2 x i64> %v1) {
  %tmp0 = shufflevector <2 x i64> %v0, <2 x i64> %v1, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %tmp0
}

; COST-LABEL: trn2.v2i64
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <2 x i64> %v0, <2 x i64> %v1, <2 x i32> <i32 1, i32 3>
; CODE-LABEL: trn2.v2i64
; CODE:       zip2 v0.2d, v0.2d, v1.2d
define <2 x i64> @trn2.v2i64(<2 x i64> %v0, <2 x i64> %v1) {
  %tmp0 = shufflevector <2 x i64> %v0, <2 x i64> %v1, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %tmp0
}

; COST-LABEL: trn1.v2f32
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <2 x float> %v0, <2 x float> %v1, <2 x i32> <i32 0, i32 2>
; CODE-LABEL: trn1.v2f32
; CODE:       zip1 v0.2s, v0.2s, v1.2s
define <2 x float> @trn1.v2f32(<2 x float> %v0, <2 x float> %v1) {
  %tmp0 = shufflevector <2 x float> %v0, <2 x float> %v1, <2 x i32> <i32 0, i32 2>
  ret <2 x float> %tmp0
}

; COST-LABEL: trn2.v2f32
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <2 x float> %v0, <2 x float> %v1, <2 x i32> <i32 1, i32 3>
; CODE-LABEL: trn2.v2f32
; CODE:       zip2 v0.2s, v0.2s, v1.2s
define <2 x float> @trn2.v2f32(<2 x float> %v0, <2 x float> %v1) {
  %tmp0 = shufflevector <2 x float> %v0, <2 x float> %v1, <2 x i32> <i32 1, i32 3>
  ret <2 x float> %tmp0
}

; COST-LABEL: trn1.v4f32
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <4 x float> %v0, <4 x float> %v1, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
; CODE-LABEL: trn1.v4f32
; CODE:       trn1 v0.4s, v0.4s, v1.4s
define <4 x float> @trn1.v4f32(<4 x float> %v0, <4 x float> %v1) {
  %tmp0 = shufflevector <4 x float> %v0, <4 x float> %v1, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x float> %tmp0
}

; COST-LABEL: trn2.v4f32
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <4 x float> %v0, <4 x float> %v1, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
; CODE-LABEL: trn2.v4f32
; CODE:       trn2 v0.4s, v0.4s, v1.4s
define <4 x float> @trn2.v4f32(<4 x float> %v0, <4 x float> %v1) {
  %tmp0 = shufflevector <4 x float> %v0, <4 x float> %v1, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x float> %tmp0
}

; COST-LABEL: trn1.v2f64
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <2 x double> %v0, <2 x double> %v1, <2 x i32> <i32 0, i32 2>
; CODE-LABEL: trn1.v2f64
; CODE:       zip1 v0.2d, v0.2d, v1.2d
define <2 x double> @trn1.v2f64(<2 x double> %v0, <2 x double> %v1) {
  %tmp0 = shufflevector <2 x double> %v0, <2 x double> %v1, <2 x i32> <i32 0, i32 2>
  ret <2 x double> %tmp0
}

; COST-LABEL: trn2.v2f64
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <2 x double> %v0, <2 x double> %v1, <2 x i32> <i32 1, i32 3>
; CODE-LABEL: trn2.v2f64
; CODE:       zip2 v0.2d, v0.2d, v1.2d
define <2 x double> @trn2.v2f64(<2 x double> %v0, <2 x double> %v1) {
  %tmp0 = shufflevector <2 x double> %v0, <2 x double> %v1, <2 x i32> <i32 1, i32 3>
  ret <2 x double> %tmp0
}
