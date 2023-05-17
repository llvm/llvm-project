; Test vector zero-extending loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i1->v16i8 extension.
define <16 x i8> @f1(ptr %ptr) {
; No expected output, but must compile.
  %val = load <16 x i1>, ptr %ptr
  %ret = zext <16 x i1> %val to <16 x i8>
  ret <16 x i8> %ret
}

; Test a v8i1->v8i16 extension.
define <8 x i16> @f2(ptr %ptr) {
; No expected output, but must compile.
  %val = load <8 x i1>, ptr %ptr
  %ret = zext <8 x i1> %val to <8 x i16>
  ret <8 x i16> %ret
}

; Test a v8i8->v8i16 extension.
define <8 x i16> @f3(ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: vlrepg [[REG1:%v[0-9]+]], 0(%r2)
; CHECK: vuplhb %v24, [[REG1]]
; CHECK: br %r14
  %val = load <8 x i8>, ptr %ptr
  %ret = zext <8 x i8> %val to <8 x i16>
  ret <8 x i16> %ret
}

; Test a v4i1->v4i32 extension.
define <4 x i32> @f4(ptr %ptr) {
; No expected output, but must compile.
  %val = load <4 x i1>, ptr %ptr
  %ret = zext <4 x i1> %val to <4 x i32>
  ret <4 x i32> %ret
}

; Test a v4i8->v4i32 extension.
define <4 x i32> @f5(ptr %ptr) {
; CHECK-LABEL: f5:
; CHECK: larl	%r1, .LCPI4_0
; CHECK: vlrepf [[REG1:%v[0-9]+]], 0(%r2)
; CHECK: vl	%v1, 0(%r1), 3
; CHECK: vperm	%v24, %v1, [[REG1]], %v1
; CHECK: br %r14
  %val = load <4 x i8>, ptr %ptr
  %ret = zext <4 x i8> %val to <4 x i32>
  ret <4 x i32> %ret
}

; Test a v4i16->v4i32 extension.
define <4 x i32> @f6(ptr %ptr) {
; CHECK-LABEL: f6:
; CHECK: vlrepg [[REG1:%v[0-9]+]], 0(%r2)
; CHECK: vuplhh %v24, [[REG1]]
; CHECK: br %r14
  %val = load <4 x i16>, ptr %ptr
  %ret = zext <4 x i16> %val to <4 x i32>
  ret <4 x i32> %ret
}

; Test a v2i1->v2i64 extension.
define <2 x i64> @f7(ptr %ptr) {
; No expected output, but must compile.
  %val = load <2 x i1>, ptr %ptr
  %ret = zext <2 x i1> %val to <2 x i64>
  ret <2 x i64> %ret
}

; Test a v2i8->v2i64 extension.
define <2 x i64> @f8(ptr %ptr) {
; CHECK-LABEL: f8:
; CHECK: larl	%r1, .LCPI7_0
; CHECK: vlreph	[[REG1:%v[0-9]+]], 0(%r2)
; CHECK: vl	%v1, 0(%r1), 3
; CHECK: vperm	%v24, %v1, [[REG1]], %v1
; CHECK: br %r14
  %val = load <2 x i8>, ptr %ptr
  %ret = zext <2 x i8> %val to <2 x i64>
  ret <2 x i64> %ret
}

; Test a v2i16->v2i64 extension.
define <2 x i64> @f9(ptr %ptr) {
; CHECK-LABEL: f9:
; CHECK: larl	%r1, .LCPI8_0
; CHECK: vlrepf	[[REG1:%v[0-9]+]], 0(%r2)
; CHECK: vl	%v1, 0(%r1), 3
; CHECK: vperm	%v24, %v1, [[REG1]], %v1
; CHECK: br %r14
  %val = load <2 x i16>, ptr %ptr
  %ret = zext <2 x i16> %val to <2 x i64>
  ret <2 x i64> %ret
}

; Test a v2i32->v2i64 extension.
define <2 x i64> @f10(ptr %ptr) {
; CHECK-LABEL: f10:
; CHECK: vlrepg [[REG1:%v[0-9]+]], 0(%r2)
; CHECK: vuplhf %v24, [[REG1]]
; CHECK: br %r14
  %val = load <2 x i32>, ptr %ptr
  %ret = zext <2 x i32> %val to <2 x i64>
  ret <2 x i64> %ret
}
