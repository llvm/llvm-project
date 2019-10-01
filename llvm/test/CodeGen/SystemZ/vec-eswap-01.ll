; Test loads of byte-swapped vector elements.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

; Test v16i8 loads.
define <16 x i8> @f1(<16 x i8> *%ptr) {
; CHECK-LABEL: f1:
; CHECK: vlbrq %v24, 0(%r2)
; CHECK: br %r14
  %load = load <16 x i8>, <16 x i8> *%ptr
  %ret = shufflevector <16 x i8> %load, <16 x i8> undef,
                       <16 x i32> <i32 15, i32 14, i32 13, i32 12,
                                   i32 11, i32 10, i32 9, i32 8,
                                   i32 7, i32 6, i32 5, i32 4,
                                   i32 3, i32 2, i32 1, i32 0>
  ret <16 x i8> %ret
}

; Test v8i16 loads.
define <8 x i16> @f2(<8 x i16> *%ptr) {
; CHECK-LABEL: f2:
; CHECK: vlerh %v24, 0(%r2)
; CHECK: br %r14
  %load = load <8 x i16>, <8 x i16> *%ptr
  %ret = shufflevector <8 x i16> %load, <8 x i16> undef,
                       <8 x i32> <i32 7, i32 6, i32 5, i32 4,
                                  i32 3, i32 2, i32 1, i32 0>
  ret <8 x i16> %ret
}

; Test v4i32 loads.
define <4 x i32> @f3(<4 x i32> *%ptr) {
; CHECK-LABEL: f3:
; CHECK: vlerf %v24, 0(%r2)
; CHECK: br %r14
  %load = load <4 x i32>, <4 x i32> *%ptr
  %ret = shufflevector <4 x i32> %load, <4 x i32> undef,
                       <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %ret
}

; Test v2i64 loads.
define <2 x i64> @f4(<2 x i64> *%ptr) {
; CHECK-LABEL: f4:
; CHECK: vlerg %v24, 0(%r2)
; CHECK: br %r14
  %load = load <2 x i64>, <2 x i64> *%ptr
  %ret = shufflevector <2 x i64> %load, <2 x i64> undef,
                       <2 x i32> <i32 1, i32 0>
  ret <2 x i64> %ret
}

; Test v4f32 loads.
define <4 x float> @f5(<4 x float> *%ptr) {
; CHECK-LABEL: f5:
; CHECK: vlerf %v24, 0(%r2)
; CHECK: br %r14
  %load = load <4 x float>, <4 x float> *%ptr
  %ret = shufflevector <4 x float> %load, <4 x float> undef,
                       <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x float> %ret
}

; Test v2f64 loads.
define <2 x double> @f6(<2 x double> *%ptr) {
; CHECK-LABEL: f6:
; CHECK: vlerg %v24, 0(%r2)
; CHECK: br %r14
  %load = load <2 x double>, <2 x double> *%ptr
  %ret = shufflevector <2 x double> %load, <2 x double> undef,
                       <2 x i32> <i32 1, i32 0>
  ret <2 x double> %ret
}

; Test the highest aligned in-range offset.
define <4 x i32> @f7(<4 x i32> *%base) {
; CHECK-LABEL: f7:
; CHECK: vlerf %v24, 4080(%r2)
; CHECK: br %r14
  %ptr = getelementptr <4 x i32>, <4 x i32> *%base, i64 255
  %load = load <4 x i32>, <4 x i32> *%ptr
  %ret = shufflevector <4 x i32> %load, <4 x i32> undef,
                       <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %ret
}

; Test the highest unaligned in-range offset.
define <4 x i32> @f8(i8 *%base) {
; CHECK-LABEL: f8:
; CHECK: vlerf %v24, 4095(%r2)
; CHECK: br %r14
  %addr = getelementptr i8, i8 *%base, i64 4095
  %ptr = bitcast i8 *%addr to <4 x i32> *
  %load = load <4 x i32>, <4 x i32> *%ptr
  %ret = shufflevector <4 x i32> %load, <4 x i32> undef,
                       <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %ret
}

; Test the next offset up, which requires separate address logic,
define <4 x i32> @f9(<4 x i32> *%base) {
; CHECK-LABEL: f9:
; CHECK: aghi %r2, 4096
; CHECK: vlerf %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr <4 x i32>, <4 x i32> *%base, i64 256
  %load = load <4 x i32>, <4 x i32> *%ptr
  %ret = shufflevector <4 x i32> %load, <4 x i32> undef,
                       <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %ret
}

; Test negative offsets, which also require separate address logic,
define <4 x i32> @f10(<4 x i32> *%base) {
; CHECK-LABEL: f10:
; CHECK: aghi %r2, -16
; CHECK: vlerf %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr <4 x i32>, <4 x i32> *%base, i64 -1
  %load = load <4 x i32>, <4 x i32> *%ptr
  %ret = shufflevector <4 x i32> %load, <4 x i32> undef,
                       <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %ret
}

; Check that indexes are allowed.
define <4 x i32> @f11(i8 *%base, i64 %index) {
; CHECK-LABEL: f11:
; CHECK: vlerf %v24, 0(%r3,%r2)
; CHECK: br %r14
  %addr = getelementptr i8, i8 *%base, i64 %index
  %ptr = bitcast i8 *%addr to <4 x i32> *
  %load = load <4 x i32>, <4 x i32> *%ptr
  %ret = shufflevector <4 x i32> %load, <4 x i32> undef,
                       <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %ret
}

