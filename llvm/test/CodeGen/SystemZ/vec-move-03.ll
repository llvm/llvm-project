; Test vector stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8 stores.
define void @f1(<16 x i8> %val, <16 x i8> *%ptr) {
; CHECK-LABEL: f1:
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  store <16 x i8> %val, <16 x i8> *%ptr
  ret void
}

; Test v8i16 stores.
define void @f2(<8 x i16> %val, <8 x i16> *%ptr) {
; CHECK-LABEL: f2:
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  store <8 x i16> %val, <8 x i16> *%ptr
  ret void
}

; Test v4i32 stores.
define void @f3(<4 x i32> %val, <4 x i32> *%ptr) {
; CHECK-LABEL: f3:
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  store <4 x i32> %val, <4 x i32> *%ptr
  ret void
}

; Test v2i64 stores.
define void @f4(<2 x i64> %val, <2 x i64> *%ptr) {
; CHECK-LABEL: f4:
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  store <2 x i64> %val, <2 x i64> *%ptr
  ret void
}

; Test v4f32 stores.
define void @f5(<4 x float> %val, <4 x float> *%ptr) {
; CHECK-LABEL: f5:
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  store <4 x float> %val, <4 x float> *%ptr
  ret void
}

; Test v2f64 stores.
define void @f6(<2 x double> %val, <2 x double> *%ptr) {
; CHECK-LABEL: f6:
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  store <2 x double> %val, <2 x double> *%ptr
  ret void
}

; Test the highest aligned in-range offset.
define void @f7(<16 x i8> %val, <16 x i8> *%base) {
; CHECK-LABEL: f7:
; CHECK: vst %v24, 4080(%r2), 3
; CHECK: br %r14
  %ptr = getelementptr <16 x i8>, <16 x i8> *%base, i64 255
  store <16 x i8> %val, <16 x i8> *%ptr
  ret void
}

; Test the highest unaligned in-range offset.
define void @f8(<16 x i8> %val, i8 *%base) {
; CHECK-LABEL: f8:
; CHECK: vst %v24, 4095(%r2)
; CHECK: br %r14
  %addr = getelementptr i8, i8 *%base, i64 4095
  %ptr = bitcast i8 *%addr to <16 x i8> *
  store <16 x i8> %val, <16 x i8> *%ptr, align 1
  ret void
}

; Test the next offset up, which requires separate address logic,
define void @f9(<16 x i8> %val, <16 x i8> *%base) {
; CHECK-LABEL: f9:
; CHECK: aghi %r2, 4096
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  %ptr = getelementptr <16 x i8>, <16 x i8> *%base, i64 256
  store <16 x i8> %val, <16 x i8> *%ptr
  ret void
}

; Test negative offsets, which also require separate address logic,
define void @f10(<16 x i8> %val, <16 x i8> *%base) {
; CHECK-LABEL: f10:
; CHECK: aghi %r2, -16
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  %ptr = getelementptr <16 x i8>, <16 x i8> *%base, i64 -1
  store <16 x i8> %val, <16 x i8> *%ptr
  ret void
}

; Check that indexes are allowed.
define void @f11(<16 x i8> %val, i8 *%base, i64 %index) {
; CHECK-LABEL: f11:
; CHECK: vst %v24, 0(%r3,%r2)
; CHECK: br %r14
  %addr = getelementptr i8, i8 *%base, i64 %index
  %ptr = bitcast i8 *%addr to <16 x i8> *
  store <16 x i8> %val, <16 x i8> *%ptr, align 1
  ret void
}

; Test v2i8 stores.
define void @f12(<2 x i8> %val, <2 x i8> *%ptr) {
; CHECK-LABEL: f12:
; CHECK: vsteh %v24, 0(%r2), 0
; CHECK: br %r14
  store <2 x i8> %val, <2 x i8> *%ptr
  ret void
}

; Test v4i8 stores.
define void @f13(<4 x i8> %val, <4 x i8> *%ptr) {
; CHECK-LABEL: f13:
; CHECK: vstef %v24, 0(%r2)
; CHECK: br %r14
  store <4 x i8> %val, <4 x i8> *%ptr
  ret void
}

; Test v8i8 stores.
define void @f14(<8 x i8> %val, <8 x i8> *%ptr) {
; CHECK-LABEL: f14:
; CHECK: vsteg %v24, 0(%r2)
; CHECK: br %r14
  store <8 x i8> %val, <8 x i8> *%ptr
  ret void
}

; Test v2i16 stores.
define void @f15(<2 x i16> %val, <2 x i16> *%ptr) {
; CHECK-LABEL: f15:
; CHECK: vstef %v24, 0(%r2), 0
; CHECK: br %r14
  store <2 x i16> %val, <2 x i16> *%ptr
  ret void
}

; Test v4i16 stores.
define void @f16(<4 x i16> %val, <4 x i16> *%ptr) {
; CHECK-LABEL: f16:
; CHECK: vsteg %v24, 0(%r2)
; CHECK: br %r14
  store <4 x i16> %val, <4 x i16> *%ptr
  ret void
}

; Test v2i32 stores.
define void @f17(<2 x i32> %val, <2 x i32> *%ptr) {
; CHECK-LABEL: f17:
; CHECK: vsteg %v24, 0(%r2), 0
; CHECK: br %r14
  store <2 x i32> %val, <2 x i32> *%ptr
  ret void
}

; Test v2f32 stores.
define void @f18(<2 x float> %val, <2 x float> *%ptr) {
; CHECK-LABEL: f18:
; CHECK: vsteg %v24, 0(%r2), 0
; CHECK: br %r14
  store <2 x float> %val, <2 x float> *%ptr
  ret void
}

; Test quadword-aligned stores.
define void @f19(<16 x i8> %val, <16 x i8> *%ptr) {
; CHECK-LABEL: f19:
; CHECK: vst %v24, 0(%r2), 4
; CHECK: br %r14
  store <16 x i8> %val, <16 x i8> *%ptr, align 16
  ret void
}

