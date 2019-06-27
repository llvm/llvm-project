; Test vector loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8 loads.
define <16 x i8> @f1(<16 x i8> *%ptr) {
; CHECK-LABEL: f1:
; CHECK: vl %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <16 x i8>, <16 x i8> *%ptr
  ret <16 x i8> %ret
}

; Test v8i16 loads.
define <8 x i16> @f2(<8 x i16> *%ptr) {
; CHECK-LABEL: f2:
; CHECK: vl %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <8 x i16>, <8 x i16> *%ptr
  ret <8 x i16> %ret
}

; Test v4i32 loads.
define <4 x i32> @f3(<4 x i32> *%ptr) {
; CHECK-LABEL: f3:
; CHECK: vl %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <4 x i32>, <4 x i32> *%ptr
  ret <4 x i32> %ret
}

; Test v2i64 loads.
define <2 x i64> @f4(<2 x i64> *%ptr) {
; CHECK-LABEL: f4:
; CHECK: vl %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <2 x i64>, <2 x i64> *%ptr
  ret <2 x i64> %ret
}

; Test v4f32 loads.
define <4 x float> @f5(<4 x float> *%ptr) {
; CHECK-LABEL: f5:
; CHECK: vl %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <4 x float>, <4 x float> *%ptr
  ret <4 x float> %ret
}

; Test v2f64 loads.
define <2 x double> @f6(<2 x double> *%ptr) {
; CHECK-LABEL: f6:
; CHECK: vl %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <2 x double>, <2 x double> *%ptr
  ret <2 x double> %ret
}

; Test the highest aligned in-range offset.
define <16 x i8> @f7(<16 x i8> *%base) {
; CHECK-LABEL: f7:
; CHECK: vl %v24, 4080(%r2)
; CHECK: br %r14
  %ptr = getelementptr <16 x i8>, <16 x i8> *%base, i64 255
  %ret = load <16 x i8>, <16 x i8> *%ptr
  ret <16 x i8> %ret
}

; Test the highest unaligned in-range offset.
define <16 x i8> @f8(i8 *%base) {
; CHECK-LABEL: f8:
; CHECK: vl %v24, 4095(%r2)
; CHECK: br %r14
  %addr = getelementptr i8, i8 *%base, i64 4095
  %ptr = bitcast i8 *%addr to <16 x i8> *
  %ret = load <16 x i8>, <16 x i8> *%ptr, align 1
  ret <16 x i8> %ret
}

; Test the next offset up, which requires separate address logic,
define <16 x i8> @f9(<16 x i8> *%base) {
; CHECK-LABEL: f9:
; CHECK: aghi %r2, 4096
; CHECK: vl %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr <16 x i8>, <16 x i8> *%base, i64 256
  %ret = load <16 x i8>, <16 x i8> *%ptr
  ret <16 x i8> %ret
}

; Test negative offsets, which also require separate address logic,
define <16 x i8> @f10(<16 x i8> *%base) {
; CHECK-LABEL: f10:
; CHECK: aghi %r2, -16
; CHECK: vl %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr <16 x i8>, <16 x i8> *%base, i64 -1
  %ret = load <16 x i8>, <16 x i8> *%ptr
  ret <16 x i8> %ret
}

; Check that indexes are allowed.
define <16 x i8> @f11(i8 *%base, i64 %index) {
; CHECK-LABEL: f11:
; CHECK: vl %v24, 0(%r3,%r2)
; CHECK: br %r14
  %addr = getelementptr i8, i8 *%base, i64 %index
  %ptr = bitcast i8 *%addr to <16 x i8> *
  %ret = load <16 x i8>, <16 x i8> *%ptr, align 1
  ret <16 x i8> %ret
}

; Test v2i8 loads.
define <2 x i8> @f12(<2 x i8> *%ptr) {
; CHECK-LABEL: f12:
; CHECK: vlreph %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <2 x i8>, <2 x i8> *%ptr
  ret <2 x i8> %ret
}

; Test v4i8 loads.
define <4 x i8> @f13(<4 x i8> *%ptr) {
; CHECK-LABEL: f13:
; CHECK: vlrepf %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <4 x i8>, <4 x i8> *%ptr
  ret <4 x i8> %ret
}

; Test v8i8 loads.
define <8 x i8> @f14(<8 x i8> *%ptr) {
; CHECK-LABEL: f14:
; CHECK: vlrepg %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <8 x i8>, <8 x i8> *%ptr
  ret <8 x i8> %ret
}

; Test v2i16 loads.
define <2 x i16> @f15(<2 x i16> *%ptr) {
; CHECK-LABEL: f15:
; CHECK: vlrepf %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <2 x i16>, <2 x i16> *%ptr
  ret <2 x i16> %ret
}

; Test v4i16 loads.
define <4 x i16> @f16(<4 x i16> *%ptr) {
; CHECK-LABEL: f16:
; CHECK: vlrepg %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <4 x i16>, <4 x i16> *%ptr
  ret <4 x i16> %ret
}

; Test v2i32 loads.
define <2 x i32> @f17(<2 x i32> *%ptr) {
; CHECK-LABEL: f17:
; CHECK: vlrepg %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <2 x i32>, <2 x i32> *%ptr
  ret <2 x i32> %ret
}

; Test v2f32 loads.
define <2 x float> @f18(<2 x float> *%ptr) {
; CHECK-LABEL: f18:
; CHECK: vlrepg %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <2 x float>, <2 x float> *%ptr
  ret <2 x float> %ret
}
