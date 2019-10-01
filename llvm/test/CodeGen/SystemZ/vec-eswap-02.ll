; Test stores of element-swapped vector elements.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

; Test v16i8 stores.
define void @f1(<16 x i8> %val, <16 x i8> *%ptr) {
; CHECK-LABEL: f1:
; CHECK: vstbrq %v24, 0(%r2)
; CHECK: br %r14
  %swap = shufflevector <16 x i8> %val, <16 x i8> undef,
                        <16 x i32> <i32 15, i32 14, i32 13, i32 12,
                                    i32 11, i32 10, i32 9, i32 8,
                                    i32 7, i32 6, i32 5, i32 4,
                                    i32 3, i32 2, i32 1, i32 0>
  store <16 x i8> %swap, <16 x i8> *%ptr
  ret void
}

; Test v8i16 stores.
define void @f2(<8 x i16> %val, <8 x i16> *%ptr) {
; CHECK-LABEL: f2:
; CHECK: vsterh %v24, 0(%r2)
; CHECK: br %r14
  %swap = shufflevector <8 x i16> %val, <8 x i16> undef,
                        <8 x i32> <i32 7, i32 6, i32 5, i32 4,
                                   i32 3, i32 2, i32 1, i32 0>
  store <8 x i16> %swap, <8 x i16> *%ptr
  ret void
}

; Test v4i32 stores.
define void @f3(<4 x i32> %val, <4 x i32> *%ptr) {
; CHECK-LABEL: f3:
; CHECK: vsterf %v24, 0(%r2)
; CHECK: br %r14
  %swap = shufflevector <4 x i32> %val, <4 x i32> undef,
                        <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  store <4 x i32> %swap, <4 x i32> *%ptr
  ret void
}

; Test v2i64 stores.
define void @f4(<2 x i64> %val, <2 x i64> *%ptr) {
; CHECK-LABEL: f4:
; CHECK: vsterg %v24, 0(%r2)
; CHECK: br %r14
  %swap = shufflevector <2 x i64> %val, <2 x i64> undef,
                        <2 x i32> <i32 1, i32 0>
  store <2 x i64> %swap, <2 x i64> *%ptr
  ret void
}

; Test v4f32 stores.
define void @f5(<4 x float> %val, <4 x float> *%ptr) {
; CHECK-LABEL: f5:
; CHECK: vsterf %v24, 0(%r2)
; CHECK: br %r14
  %swap = shufflevector <4 x float> %val, <4 x float> undef,
                        <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  store <4 x float> %swap, <4 x float> *%ptr
  ret void
}

; Test v2f64 stores.
define void @f6(<2 x double> %val, <2 x double> *%ptr) {
; CHECK-LABEL: f6:
; CHECK: vsterg %v24, 0(%r2)
; CHECK: br %r14
  %swap = shufflevector <2 x double> %val, <2 x double> undef,
                        <2 x i32> <i32 1, i32 0>
  store <2 x double> %swap, <2 x double> *%ptr
  ret void
}

; Test the highest aligned in-range offset.
define void @f7(<4 x i32> %val, <4 x i32> *%base) {
; CHECK-LABEL: f7:
; CHECK: vsterf %v24, 4080(%r2)
; CHECK: br %r14
  %ptr = getelementptr <4 x i32>, <4 x i32> *%base, i64 255
  %swap = shufflevector <4 x i32> %val, <4 x i32> undef,
                        <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  store <4 x i32> %swap, <4 x i32> *%ptr
  ret void
}

; Test the highest unaligned in-range offset.
define void @f8(<4 x i32> %val, i8 *%base) {
; CHECK-LABEL: f8:
; CHECK: vsterf %v24, 4095(%r2)
; CHECK: br %r14
  %addr = getelementptr i8, i8 *%base, i64 4095
  %ptr = bitcast i8 *%addr to <4 x i32> *
  %swap = shufflevector <4 x i32> %val, <4 x i32> undef,
                        <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  store <4 x i32> %swap, <4 x i32> *%ptr, align 1
  ret void
}

; Test the next offset up, which requires separate address logic,
define void @f9(<4 x i32> %val, <4 x i32> *%base) {
; CHECK-LABEL: f9:
; CHECK: aghi %r2, 4096
; CHECK: vsterf %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr <4 x i32>, <4 x i32> *%base, i64 256
  %swap = shufflevector <4 x i32> %val, <4 x i32> undef,
                        <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  store <4 x i32> %swap, <4 x i32> *%ptr
  ret void
}

; Test negative offsets, which also require separate address logic,
define void @f10(<4 x i32> %val, <4 x i32> *%base) {
; CHECK-LABEL: f10:
; CHECK: aghi %r2, -16
; CHECK: vsterf %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr <4 x i32>, <4 x i32> *%base, i64 -1
  %swap = shufflevector <4 x i32> %val, <4 x i32> undef,
                        <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  store <4 x i32> %swap, <4 x i32> *%ptr
  ret void
}

; Check that indexes are allowed.
define void @f11(<4 x i32> %val, i8 *%base, i64 %index) {
; CHECK-LABEL: f11:
; CHECK: vsterf %v24, 0(%r3,%r2)
; CHECK: br %r14
  %addr = getelementptr i8, i8 *%base, i64 %index
  %ptr = bitcast i8 *%addr to <4 x i32> *
  %swap = shufflevector <4 x i32> %val, <4 x i32> undef,
                        <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  store <4 x i32> %swap, <4 x i32> *%ptr, align 1
  ret void
}

