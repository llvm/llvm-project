; RUN: opt < %s -mtriple=aarch64--linux-gnu -cost-model -analyze | FileCheck %s --check-prefix=COST
; RUN: llc < %s -mtriple=aarch64--linux-gnu | FileCheck %s --check-prefix=CODE

; COST-LABEL: sel.v8i8
; COST:       Found an estimated cost of 42 for instruction: %tmp0 = shufflevector <8 x i8> %v0, <8 x i8> %v1, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 15>
; CODE-LABEL: sel.v8i8
; CODE:       tbl v0.8b, { v0.16b }, v2.8b
define <8 x i8> @sel.v8i8(<8 x i8> %v0, <8 x i8> %v1) {
  %tmp0 = shufflevector <8 x i8> %v0, <8 x i8> %v1, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 15>
  ret <8 x i8> %tmp0
}

; COST-LABEL: sel.v16i8
; COST:       Found an estimated cost of 90 for instruction: %tmp0 = shufflevector <16 x i8> %v0, <16 x i8> %v1, <16 x i32> <i32 0, i32 17, i32 2, i32 19, i32 4, i32 21, i32 6, i32 23, i32 8, i32 25, i32 10, i32 27, i32 12, i32 29, i32 14, i32 31>
; CODE-LABEL: sel.v16i8
; CODE:       tbl v0.16b, { v0.16b, v1.16b }, v2.16b
define <16 x i8> @sel.v16i8(<16 x i8> %v0, <16 x i8> %v1) {
  %tmp0 = shufflevector <16 x i8> %v0, <16 x i8> %v1, <16 x i32> <i32 0, i32 17, i32 2, i32 19, i32 4, i32 21, i32 6, i32 23, i32 8, i32 25, i32 10, i32 27, i32 12, i32 29, i32 14, i32 31>
  ret <16 x i8> %tmp0
}

; COST-LABEL: sel.v4i16
; COST:       Found an estimated cost of 18 for instruction: %tmp0 = shufflevector <4 x i16> %v0, <4 x i16> %v1, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
; CODE-LABEL: sel.v4i16
; CODE:       rev32 v0.4h, v0.4h
; CODE:       trn2 v0.4h, v0.4h, v1.4h
define <4 x i16> @sel.v4i16(<4 x i16> %v0, <4 x i16> %v1) {
  %tmp0 = shufflevector <4 x i16> %v0, <4 x i16> %v1, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x i16> %tmp0
}

; COST-LABEL: sel.v8i16
; COST:       Found an estimated cost of 42 for instruction: %tmp0 = shufflevector <8 x i16> %v0, <8 x i16> %v1, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 15>
; CODE-LABEL: sel.v8i16
; CODE:       tbl v0.16b, { v0.16b, v1.16b }, v2.16b
define <8 x i16> @sel.v8i16(<8 x i16> %v0, <8 x i16> %v1) {
  %tmp0 = shufflevector <8 x i16> %v0, <8 x i16> %v1, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 15>
  ret <8 x i16> %tmp0
}

; COST-LABEL: sel.v2i32
; COST:        Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <2 x i32> %v0, <2 x i32> %v1, <2 x i32> <i32 0, i32 3>
; CODE-LABEL: sel.v2i32
; CODE:       mov v0.s[1], v1.s[1]
define <2 x i32> @sel.v2i32(<2 x i32> %v0, <2 x i32> %v1) {
  %tmp0 = shufflevector <2 x i32> %v0, <2 x i32> %v1, <2 x i32> <i32 0, i32 3>
  ret <2 x i32> %tmp0
}

; COST-LABEL: sel.v4i32
; COST:       Found an estimated cost of 2 for instruction: %tmp0 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
; CODE-LABEL: sel.v4i32
; CODE:       rev64 v0.4s, v0.4s
; CODE:       trn2 v0.4s, v0.4s, v1.4s
define <4 x i32> @sel.v4i32(<4 x i32> %v0, <4 x i32> %v1) {
  %tmp0 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x i32> %tmp0
}

; COST-LABEL: sel.v2i64
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <2 x i64> %v0, <2 x i64> %v1, <2 x i32> <i32 0, i32 3>
; CODE-LABEL: sel.v2i64
; CODE:       mov v0.d[1], v1.d[1]
define <2 x i64> @sel.v2i64(<2 x i64> %v0, <2 x i64> %v1) {
  %tmp0 = shufflevector <2 x i64> %v0, <2 x i64> %v1, <2 x i32> <i32 0, i32 3>
  ret <2 x i64> %tmp0
}

; COST-LABEL: sel.v2f32
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <2 x float> %v0, <2 x float> %v1, <2 x i32> <i32 0, i32 3>
; CODE-LABEL: sel.v2f32
; CODE:       mov v0.s[1], v1.s[1]
define <2 x float> @sel.v2f32(<2 x float> %v0, <2 x float> %v1) {
  %tmp0 = shufflevector <2 x float> %v0, <2 x float> %v1, <2 x i32> <i32 0, i32 3>
  ret <2 x float> %tmp0
}

; COST-LABEL: sel.v4f32
; COST:       Found an estimated cost of 2 for instruction: %tmp0 = shufflevector <4 x float> %v0, <4 x float> %v1, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
; CODE-LABEL: sel.v4f32
; CODE:       rev64 v0.4s, v0.4s
; CODE:       trn2 v0.4s, v0.4s, v1.4s
define <4 x float> @sel.v4f32(<4 x float> %v0, <4 x float> %v1) {
  %tmp0 = shufflevector <4 x float> %v0, <4 x float> %v1, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x float> %tmp0
}

; COST-LABEL: sel.v2f64
; COST:       Found an estimated cost of 1 for instruction: %tmp0 = shufflevector <2 x double> %v0, <2 x double> %v1, <2 x i32> <i32 0, i32 3>
; CODE-LABEL: sel.v2f64
; CODE:       mov v0.d[1], v1.d[1]
define <2 x double> @sel.v2f64(<2 x double> %v0, <2 x double> %v1) {
  %tmp0 = shufflevector <2 x double> %v0, <2 x double> %v1, <2 x i32> <i32 0, i32 3>
  ret <2 x double> %tmp0
}
