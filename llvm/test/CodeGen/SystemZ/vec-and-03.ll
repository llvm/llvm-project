; Test vector zero extensions, which need to be implemented as ANDs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i1->v16i8 extension.
define <16 x i8> @f1(<16 x i8> %val) {
; CHECK-LABEL: f1:
; CHECK: vrepib [[REG:%v[0-9]+]], 1
; CHECK: vn %v24, %v24, [[REG]]
; CHECK: br %r14
  %trunc = trunc <16 x i8> %val to <16 x i1>
  %ret = zext <16 x i1> %trunc to <16 x i8>
  ret <16 x i8> %ret
}

; Test a v8i1->v8i16 extension.
define <8 x i16> @f2(<8 x i16> %val) {
; CHECK-LABEL: f2:
; CHECK: vrepih [[REG:%v[0-9]+]], 1
; CHECK: vn %v24, %v24, [[REG]]
; CHECK: br %r14
  %trunc = trunc <8 x i16> %val to <8 x i1>
  %ret = zext <8 x i1> %trunc to <8 x i16>
  ret <8 x i16> %ret
}

; Test a v8i8->v8i16 extension.
define <8 x i16> @f3(<8 x i16> %val) {
; CHECK-LABEL: f3:
; CHECK: vgbm [[REG:%v[0-9]+]], 21845
; CHECK: vn %v24, %v24, [[REG]]
; CHECK: br %r14
  %trunc = trunc <8 x i16> %val to <8 x i8>
  %ret = zext <8 x i8> %trunc to <8 x i16>
  ret <8 x i16> %ret
}

; Test a v4i1->v4i32 extension.
define <4 x i32> @f4(<4 x i32> %val) {
; CHECK-LABEL: f4:
; CHECK: vrepif [[REG:%v[0-9]+]], 1
; CHECK: vn %v24, %v24, [[REG]]
; CHECK: br %r14
  %trunc = trunc <4 x i32> %val to <4 x i1>
  %ret = zext <4 x i1> %trunc to <4 x i32>
  ret <4 x i32> %ret
}

; Test a v4i8->v4i32 extension.
define <4 x i32> @f5(<4 x i32> %val) {
; CHECK-LABEL: f5:
; CHECK: vgbm [[REG:%v[0-9]+]], 4369
; CHECK: vn %v24, %v24, [[REG]]
; CHECK: br %r14
  %trunc = trunc <4 x i32> %val to <4 x i8>
  %ret = zext <4 x i8> %trunc to <4 x i32>
  ret <4 x i32> %ret
}

; Test a v4i16->v4i32 extension.
define <4 x i32> @f6(<4 x i32> %val) {
; CHECK-LABEL: f6:
; CHECK: vgbm [[REG:%v[0-9]+]], 13107
; CHECK: vn %v24, %v24, [[REG]]
; CHECK: br %r14
  %trunc = trunc <4 x i32> %val to <4 x i16>
  %ret = zext <4 x i16> %trunc to <4 x i32>
  ret <4 x i32> %ret
}

; Test a v2i1->v2i64 extension.
define <2 x i64> @f7(<2 x i64> %val) {
; CHECK-LABEL: f7:
; CHECK: vrepig [[REG:%v[0-9]+]], 1
; CHECK: vn %v24, %v24, [[REG]]
; CHECK: br %r14
  %trunc = trunc <2 x i64> %val to <2 x i1>
  %ret = zext <2 x i1> %trunc to <2 x i64>
  ret <2 x i64> %ret
}

; Test a v2i8->v2i64 extension.
define <2 x i64> @f8(<2 x i64> %val) {
; CHECK-LABEL: f8:
; CHECK: vgbm [[REG:%v[0-9]+]], 257
; CHECK: vn %v24, %v24, [[REG]]
; CHECK: br %r14
  %trunc = trunc <2 x i64> %val to <2 x i8>
  %ret = zext <2 x i8> %trunc to <2 x i64>
  ret <2 x i64> %ret
}

; Test a v2i16->v2i64 extension.
define <2 x i64> @f9(<2 x i64> %val) {
; CHECK-LABEL: f9:
; CHECK: vgbm [[REG:%v[0-9]+]], 771
; CHECK: vn %v24, %v24, [[REG]]
; CHECK: br %r14
  %trunc = trunc <2 x i64> %val to <2 x i16>
  %ret = zext <2 x i16> %trunc to <2 x i64>
  ret <2 x i64> %ret
}

; Test a v2i32->v2i64 extension.
define <2 x i64> @f10(<2 x i64> %val) {
; CHECK-LABEL: f10:
; CHECK: vgbm [[REG:%v[0-9]+]], 3855
; CHECK: vn %v24, %v24, [[REG]]
; CHECK: br %r14
  %trunc = trunc <2 x i64> %val to <2 x i32>
  %ret = zext <2 x i32> %trunc to <2 x i64>
  ret <2 x i64> %ret
}
