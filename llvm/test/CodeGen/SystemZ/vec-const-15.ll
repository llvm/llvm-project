; Test vector replicates that use VECTOR GENERATE MASK, v4i32 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a word-granularity replicate with the lowest value that cannot use
; VREPIF.
define <4 x i32> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgmf %v24, 16, 16
; CHECK: br %r14
  ret <4 x i32> <i32 32768, i32 32768, i32 32768, i32 32768>
}

; Test a word-granularity replicate that has the lower 17 bits set.
define <4 x i32> @f2() {
; CHECK-LABEL: f2:
; CHECK: vgmf %v24, 15, 31
; CHECK: br %r14
  ret <4 x i32> <i32 131071, i32 131071, i32 131071, i32 131071>
}

; Test a word-granularity replicate that has the upper 15 bits set.
define <4 x i32> @f3() {
; CHECK-LABEL: f3:
; CHECK: vgmf %v24, 0, 14
; CHECK: br %r14
  ret <4 x i32> <i32 -131072, i32 -131072, i32 -131072, i32 -131072>
}

; Test a word-granularity replicate that has middle bits set.
define <4 x i32> @f4() {
; CHECK-LABEL: f4:
; CHECK: vgmf %v24, 12, 17
; CHECK: br %r14
  ret <4 x i32> <i32 1032192, i32 1032192, i32 1032192, i32 1032192>
}

; Test a word-granularity replicate with a wrap-around mask.
define <4 x i32> @f5() {
; CHECK-LABEL: f5:
; CHECK: vgmf %v24, 17, 15
; CHECK: br %r14
  ret <4 x i32> <i32 -32769, i32 -32769, i32 -32769, i32 -32769>
}

; Test a doubleword-granularity replicate with the lowest value that cannot
; use VREPIG.
define <4 x i32> @f6() {
; CHECK-LABEL: f6:
; CHECK: vgmg %v24, 48, 48
; CHECK: br %r14
  ret <4 x i32> <i32 0, i32 32768, i32 0, i32 32768>
}

; Test a doubleword-granularity replicate that has the lower 22 bits set.
define <4 x i32> @f7() {
; CHECK-LABEL: f7:
; CHECK: vgmg %v24, 42, 63
; CHECK: br %r14
  ret <4 x i32> <i32 0, i32 4194303, i32 0, i32 4194303>
}

; Test a doubleword-granularity replicate that has the upper 45 bits set.
define <4 x i32> @f8() {
; CHECK-LABEL: f8:
; CHECK: vgmg %v24, 0, 44
; CHECK: br %r14
  ret <4 x i32> <i32 -1, i32 -524288, i32 -1, i32 -524288>
}

; Test a doubleword-granularity replicate that has middle bits set.
define <4 x i32> @f9() {
; CHECK-LABEL: f9:
; CHECK: vgmg %v24, 31, 42
; CHECK: br %r14
  ret <4 x i32> <i32 1, i32 -2097152, i32 1, i32 -2097152>
}

; Test a doubleword-granularity replicate with a wrap-around mask.
define <4 x i32> @f10() {
; CHECK-LABEL: f10:
; CHECK: vgmg %v24, 18, 0
; CHECK: br %r14
  ret <4 x i32> <i32 -2147467265, i32 -1, i32 -2147467265, i32 -1>
}
