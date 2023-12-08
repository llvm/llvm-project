; Test vector sign extensions.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i1->v16i8 extension.
define <16 x i8> @f1(<16 x i8> %val) {
; CHECK-LABEL: f1:
; CHECK: veslb [[REG:%v[0-9]+]], %v24, 7
; CHECK: vesrab %v24, [[REG]], 7
; CHECK: br %r14
  %trunc = trunc <16 x i8> %val to <16 x i1>
  %ret = sext <16 x i1> %trunc to <16 x i8>
  ret <16 x i8> %ret
}

; Test a v8i1->v8i16 extension.
define <8 x i16> @f2(<8 x i16> %val) {
; CHECK-LABEL: f2:
; CHECK: veslh [[REG:%v[0-9]+]], %v24, 15
; CHECK: vesrah %v24, [[REG]], 15
; CHECK: br %r14
  %trunc = trunc <8 x i16> %val to <8 x i1>
  %ret = sext <8 x i1> %trunc to <8 x i16>
  ret <8 x i16> %ret
}

; Test a v8i8->v8i16 extension.
define <8 x i16> @f3(<8 x i16> %val) {
; CHECK-LABEL: f3:
; CHECK: veslh [[REG:%v[0-9]+]], %v24, 8
; CHECK: vesrah %v24, [[REG]], 8
; CHECK: br %r14
  %trunc = trunc <8 x i16> %val to <8 x i8>
  %ret = sext <8 x i8> %trunc to <8 x i16>
  ret <8 x i16> %ret
}

; Test a v4i1->v4i32 extension.
define <4 x i32> @f4(<4 x i32> %val) {
; CHECK-LABEL: f4:
; CHECK: veslf [[REG:%v[0-9]+]], %v24, 31
; CHECK: vesraf %v24, [[REG]], 31
; CHECK: br %r14
  %trunc = trunc <4 x i32> %val to <4 x i1>
  %ret = sext <4 x i1> %trunc to <4 x i32>
  ret <4 x i32> %ret
}

; Test a v4i8->v4i32 extension.
define <4 x i32> @f5(<4 x i32> %val) {
; CHECK-LABEL: f5:
; CHECK: veslf [[REG:%v[0-9]+]], %v24, 24
; CHECK: vesraf %v24, [[REG]], 24
; CHECK: br %r14
  %trunc = trunc <4 x i32> %val to <4 x i8>
  %ret = sext <4 x i8> %trunc to <4 x i32>
  ret <4 x i32> %ret
}

; Test a v4i16->v4i32 extension.
define <4 x i32> @f6(<4 x i32> %val) {
; CHECK-LABEL: f6:
; CHECK: veslf [[REG:%v[0-9]+]], %v24, 16
; CHECK: vesraf %v24, [[REG]], 16
; CHECK: br %r14
  %trunc = trunc <4 x i32> %val to <4 x i16>
  %ret = sext <4 x i16> %trunc to <4 x i32>
  ret <4 x i32> %ret
}

; Test a v2i1->v2i64 extension.
define <2 x i64> @f7(<2 x i64> %val) {
; CHECK-LABEL: f7:
; CHECK: veslg [[REG:%v[0-9]+]], %v24, 63
; CHECK: vesrag %v24, [[REG]], 63
; CHECK: br %r14
  %trunc = trunc <2 x i64> %val to <2 x i1>
  %ret = sext <2 x i1> %trunc to <2 x i64>
  ret <2 x i64> %ret
}

; Test a v2i8->v2i64 extension.
define <2 x i64> @f8(<2 x i64> %val) {
; CHECK-LABEL: f8:
; CHECK: vsegb %v24, %v24
; CHECK: br %r14
  %trunc = trunc <2 x i64> %val to <2 x i8>
  %ret = sext <2 x i8> %trunc to <2 x i64>
  ret <2 x i64> %ret
}

; Test a v2i16->v2i64 extension.
define <2 x i64> @f9(<2 x i64> %val) {
; CHECK-LABEL: f9:
; CHECK: vsegh %v24, %v24
; CHECK: br %r14
  %trunc = trunc <2 x i64> %val to <2 x i16>
  %ret = sext <2 x i16> %trunc to <2 x i64>
  ret <2 x i64> %ret
}

; Test a v2i32->v2i64 extension.
define <2 x i64> @f10(<2 x i64> %val) {
; CHECK-LABEL: f10:
; CHECK: vsegf %v24, %v24
; CHECK: br %r14
  %trunc = trunc <2 x i64> %val to <2 x i32>
  %ret = sext <2 x i32> %trunc to <2 x i64>
  ret <2 x i64> %ret
}

; Test an alternative v2i8->v2i64 extension.
define <2 x i64> @f11(<2 x i64> %val) {
; CHECK-LABEL: f11:
; CHECK: vsegb %v24, %v24
; CHECK: br %r14
  %shl = shl <2 x i64> %val, <i64 56, i64 56>
  %ret = ashr <2 x i64> %shl, <i64 56, i64 56>
  ret <2 x i64> %ret
}

; Test an alternative v2i16->v2i64 extension.
define <2 x i64> @f12(<2 x i64> %val) {
; CHECK-LABEL: f12:
; CHECK: vsegh %v24, %v24
; CHECK: br %r14
  %shl = shl <2 x i64> %val, <i64 48, i64 48>
  %ret = ashr <2 x i64> %shl, <i64 48, i64 48>
  ret <2 x i64> %ret
}

; Test an alternative v2i32->v2i64 extension.
define <2 x i64> @f13(<2 x i64> %val) {
; CHECK-LABEL: f13:
; CHECK: vsegf %v24, %v24
; CHECK: br %r14
  %shl = shl <2 x i64> %val, <i64 32, i64 32>
  %ret = ashr <2 x i64> %shl, <i64 32, i64 32>
  ret <2 x i64> %ret
}

; Test an extraction-based v2i8->v2i64 extension.
define <2 x i64> @f14(<16 x i8> %val) {
; CHECK-LABEL: f14:
; CHECK: vsegb %v24, %v24
; CHECK: br %r14
  %elt0 = extractelement <16 x i8> %val, i32 7
  %elt1 = extractelement <16 x i8> %val, i32 15
  %ext0 = sext i8 %elt0 to i64
  %ext1 = sext i8 %elt1 to i64
  %vec0 = insertelement <2 x i64> undef, i64 %ext0, i32 0
  %vec1 = insertelement <2 x i64> %vec0, i64 %ext1, i32 1
  ret <2 x i64> %vec1
}

; Test an extraction-based v2i16->v2i64 extension.
define <2 x i64> @f15(<16 x i16> %val) {
; CHECK-LABEL: f15:
; CHECK: vsegh %v24, %v24
; CHECK: br %r14
  %elt0 = extractelement <16 x i16> %val, i32 3
  %elt1 = extractelement <16 x i16> %val, i32 7
  %ext0 = sext i16 %elt0 to i64
  %ext1 = sext i16 %elt1 to i64
  %vec0 = insertelement <2 x i64> undef, i64 %ext0, i32 0
  %vec1 = insertelement <2 x i64> %vec0, i64 %ext1, i32 1
  ret <2 x i64> %vec1
}

; Test an extraction-based v2i32->v2i64 extension.
define <2 x i64> @f16(<16 x i32> %val) {
; CHECK-LABEL: f16:
; CHECK: vsegf %v24, %v24
; CHECK: br %r14
  %elt0 = extractelement <16 x i32> %val, i32 1
  %elt1 = extractelement <16 x i32> %val, i32 3
  %ext0 = sext i32 %elt0 to i64
  %ext1 = sext i32 %elt1 to i64
  %vec0 = insertelement <2 x i64> undef, i64 %ext0, i32 0
  %vec1 = insertelement <2 x i64> %vec0, i64 %ext1, i32 1
  ret <2 x i64> %vec1
}
