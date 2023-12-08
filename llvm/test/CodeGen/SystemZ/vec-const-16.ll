; Test vector replicates that use VECTOR GENERATE MASK, v2i64 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a word-granularity replicate with the lowest value that cannot use
; VREPIF.
define <2 x i64> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgmf %v24, 16, 16
; CHECK: br %r14
  ret <2 x i64> <i64 140737488388096, i64 140737488388096>
}

; Test a word-granularity replicate that has the lower 17 bits set.
define <2 x i64> @f2() {
; CHECK-LABEL: f2:
; CHECK: vgmf %v24, 15, 31
; CHECK: br %r14
  ret <2 x i64> <i64 562945658585087, i64 562945658585087>
}

; Test a word-granularity replicate that has the upper 15 bits set.
define <2 x i64> @f3() {
; CHECK-LABEL: f3:
; CHECK: vgmf %v24, 0, 14
; CHECK: br %r14
  ret <2 x i64> <i64 -562945658585088, i64 -562945658585088>
}

; Test a word-granularity replicate that has middle bits set.
define <2 x i64> @f4() {
; CHECK-LABEL: f4:
; CHECK: vgmf %v24, 12, 17
; CHECK: br %r14
  ret <2 x i64> <i64 4433230884225024, i64 4433230884225024>
}

; Test a word-granularity replicate with a wrap-around mask.
define <2 x i64> @f5() {
; CHECK-LABEL: f5:
; CHECK: vgmf %v24, 17, 15
; CHECK: br %r14
  ret <2 x i64> <i64 -140737488388097, i64 -140737488388097>
}

; Test a doubleword-granularity replicate with the lowest value that cannot
; use VREPIG.
define <2 x i64> @f6() {
; CHECK-LABEL: f6:
; CHECK: vgmg %v24, 48, 48
; CHECK: br %r14
  ret <2 x i64> <i64 32768, i64 32768>
}

; Test a doubleword-granularity replicate that has the lower 22 bits set.
define <2 x i64> @f7() {
; CHECK-LABEL: f7:
; CHECK: vgmg %v24, 42, 63
; CHECK: br %r14
  ret <2 x i64> <i64 4194303, i64 4194303>
}

; Test a doubleword-granularity replicate that has the upper 45 bits set.
define <2 x i64> @f8() {
; CHECK-LABEL: f8:
; CHECK: vgmg %v24, 0, 44
; CHECK: br %r14
  ret <2 x i64> <i64 -524288, i64 -524288>
}

; Test a doubleword-granularity replicate that has middle bits set.
define <2 x i64> @f9() {
; CHECK-LABEL: f9:
; CHECK: vgmg %v24, 31, 42
; CHECK: br %r14
  ret <2 x i64> <i64 8587837440, i64 8587837440>
}

; Test a doubleword-granularity replicate with a wrap-around mask.
define <2 x i64> @f10() {
; CHECK-LABEL: f10:
; CHECK: vgmg %v24, 18, 0
; CHECK: br %r14
  ret <2 x i64> <i64 -9223301668110598145, i64 -9223301668110598145>
}
