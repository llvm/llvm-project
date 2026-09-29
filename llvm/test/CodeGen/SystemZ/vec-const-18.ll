; Test vector replicates that use VECTOR GENERATE MASK, v2f64 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a word-granularity replicate with the lowest value that cannot use
; VREPIF.
define <2 x double> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgmf %v24, 16, 16
; CHECK: br %r14
  ret <2 x double> <double 0x0000800000008000, double 0x0000800000008000>
}

; Test a word-granularity replicate that has the lower 17 bits set.
define <2 x double> @f2() {
; CHECK-LABEL: f2:
; CHECK: vgmf %v24, 15, 31
; CHECK: br %r14
  ret <2 x double> <double 0x0001ffff0001ffff, double 0x0001ffff0001ffff>
}

; Test a word-granularity replicate that has the upper 15 bits set.
define <2 x double> @f3() {
; CHECK-LABEL: f3:
; CHECK: vgmf %v24, 0, 14
; CHECK: br %r14
  ret <2 x double> <double 0xfffe0000fffe0000, double 0xfffe0000fffe0000>
}

; Test a word-granularity replicate that has middle bits set.
define <2 x double> @f4() {
; CHECK-LABEL: f4:
; CHECK: vgmf %v24, 2, 11
; CHECK: br %r14
  ret <2 x double> <double 0x3ff000003ff00000, double 0x3ff000003ff00000>
}

; Test a word-granularity replicate with a wrap-around mask.
define <2 x double> @f5() {
; CHECK-LABEL: f5:
; CHECK: vgmf %v24, 17, 15
; CHECK: br %r14
  ret <2 x double> <double 0xffff7fffffff7fff, double 0xffff7fffffff7fff>
}

; Test a doubleword-granularity replicate with the lowest value that cannot
; use VREPIG.
define <2 x double> @f6() {
; CHECK-LABEL: f6:
; CHECK: vgmg %v24, 48, 48
; CHECK: br %r14
  ret <2 x double> <double 0x0000000000008000, double 0x0000000000008000>
}

; Test a doubleword-granularity replicate that has the lower 22 bits set.
define <2 x double> @f7() {
; CHECK-LABEL: f7:
; CHECK: vgmg %v24, 42, 63
; CHECK: br %r14
  ret <2 x double> <double 0x000000000003fffff, double 0x000000000003fffff>
}

; Test a doubleword-granularity replicate that has the upper 45 bits set.
define <2 x double> @f8() {
; CHECK-LABEL: f8:
; CHECK: vgmg %v24, 0, 44
; CHECK: br %r14
  ret <2 x double> <double 0xfffffffffff80000, double 0xfffffffffff80000>
}

; Test a doubleword-granularity replicate that has middle bits set.
define <2 x double> @f9() {
; CHECK-LABEL: f9:
; CHECK: vgmg %v24, 2, 11
; CHECK: br %r14
  ret <2 x double> <double 0x3ff0000000000000, double 0x3ff0000000000000>
}

; Test a doubleword-granularity replicate with a wrap-around mask.
define <2 x double> @f10() {
; CHECK-LABEL: f10:
; CHECK: vgmg %v24, 10, 0
; CHECK: br %r14
  ret <2 x double> <double 0x803fffffffffffff, double 0x803fffffffffffff>
}
