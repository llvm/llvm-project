; Test vector replicates, v4i32 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a byte-granularity replicate with the lowest useful value.
define <4 x i32> @f1() {
; CHECK-LABEL: f1:
; CHECK: vrepib %v24, 1
; CHECK: br %r14
  ret <4 x i32> <i32 16843009, i32 16843009, i32 16843009, i32 16843009>
}

; Test a byte-granularity replicate with an arbitrary value.
define <4 x i32> @f2() {
; CHECK-LABEL: f2:
; CHECK: vrepib %v24, -55
; CHECK: br %r14
  ret <4 x i32> <i32 3385444809, i32 3385444809, i32 3385444809, i32 3385444809>
}

; Test a byte-granularity replicate with the highest useful value.
define <4 x i32> @f3() {
; CHECK-LABEL: f3:
; CHECK: vrepib %v24, -2
; CHECK: br %r14
  ret <4 x i32> <i32 4278124286, i32 4278124286, i32 4278124286, i32 4278124286>
}

; Test a halfword-granularity replicate with the lowest useful value.
define <4 x i32> @f4() {
; CHECK-LABEL: f4:
; CHECK: vrepih %v24, 1
; CHECK: br %r14
  ret <4 x i32> <i32 65537, i32 65537, i32 65537, i32 65537>
}

; Test a halfword-granularity replicate with an arbitrary value.
define <4 x i32> @f5() {
; CHECK-LABEL: f5:
; CHECK: vrepih %v24, 25650
; CHECK: br %r14
  ret <4 x i32> <i32 1681024050, i32 1681024050, i32 1681024050, i32 1681024050>
}

; Test a halfword-granularity replicate with the highest useful value.
define <4 x i32> @f6() {
; CHECK-LABEL: f6:
; CHECK: vrepih %v24, -2
; CHECK: br %r14
  ret <4 x i32> <i32 -65538, i32 -65538, i32 -65538, i32 -65538>
}

; Test a word-granularity replicate with the lowest useful positive value.
define <4 x i32> @f7() {
; CHECK-LABEL: f7:
; CHECK: vrepif %v24, 1
; CHECK: br %r14
  ret <4 x i32> <i32 1, i32 1, i32 1, i32 1>
}

; Test a word-granularity replicate with the highest in-range value.
define <4 x i32> @f8() {
; CHECK-LABEL: f8:
; CHECK: vrepif %v24, 32767
; CHECK: br %r14
  ret <4 x i32> <i32 32767, i32 32767, i32 32767, i32 32767>
}

; Test a word-granularity replicate with the next highest value.
; This cannot use VREPIF.
define <4 x i32> @f9() {
; CHECK-LABEL: f9:
; CHECK-NOT: vrepif
; CHECK: br %r14
  ret <4 x i32> <i32 32768, i32 32768, i32 32768, i32 32768>
}

; Test a word-granularity replicate with the lowest in-range value.
define <4 x i32> @f10() {
; CHECK-LABEL: f10:
; CHECK: vrepif %v24, -32768
; CHECK: br %r14
  ret <4 x i32> <i32 -32768, i32 -32768, i32 -32768, i32 -32768>
}

; Test a word-granularity replicate with the next lowest value.
; This cannot use VREPIF.
define <4 x i32> @f11() {
; CHECK-LABEL: f11:
; CHECK-NOT: vrepif
; CHECK: br %r14
  ret <4 x i32> <i32 -32769, i32 -32769, i32 -32769, i32 -32769>
}

; Test a word-granularity replicate with the highest useful negative value.
define <4 x i32> @f12() {
; CHECK-LABEL: f12:
; CHECK: vrepif %v24, -2
; CHECK: br %r14
  ret <4 x i32> <i32 -2, i32 -2, i32 -2, i32 -2>
}

; Test a doubleword-granularity replicate with the lowest useful positive
; value.
define <4 x i32> @f13() {
; CHECK-LABEL: f13:
; CHECK: vrepig %v24, 1
; CHECK: br %r14
  ret <4 x i32> <i32 0, i32 1, i32 0, i32 1>
}

; Test a doubleword-granularity replicate with the highest in-range value.
define <4 x i32> @f14() {
; CHECK-LABEL: f14:
; CHECK: vrepig %v24, 32767
; CHECK: br %r14
  ret <4 x i32> <i32 0, i32 32767, i32 0, i32 32767>
}

; Test a doubleword-granularity replicate with the next highest value.
; This cannot use VREPIG.
define <4 x i32> @f15() {
; CHECK-LABEL: f15:
; CHECK-NOT: vrepig
; CHECK: br %r14
  ret <4 x i32> <i32 0, i32 32768, i32 0, i32 32768>
}

; Test a doubleword-granularity replicate with the lowest in-range value.
define <4 x i32> @f16() {
; CHECK-LABEL: f16:
; CHECK: vrepig %v24, -32768
; CHECK: br %r14
  ret <4 x i32> <i32 -1, i32 -32768, i32 -1, i32 -32768>
}

; Test a doubleword-granularity replicate with the next lowest value.
; This cannot use VREPIG.
define <4 x i32> @f17() {
; CHECK-LABEL: f17:
; CHECK-NOT: vrepig
; CHECK: br %r14
  ret <4 x i32> <i32 -1, i32 -32769, i32 -1, i32 -32769>
}

; Test a doubleword-granularity replicate with the highest useful negative
; value.
define <4 x i32> @f18() {
; CHECK-LABEL: f18:
; CHECK: vrepig %v24, -2
; CHECK: br %r14
  ret <4 x i32> <i32 -1, i32 -2, i32 -1, i32 -2>
}

; Repeat f14 with undefs optimistically treated as 0, 32767.
define <4 x i32> @f19() {
; CHECK-LABEL: f19:
; CHECK: vrepig %v24, 32767
; CHECK: br %r14
  ret <4 x i32> <i32 undef, i32 undef, i32 0, i32 32767>
}

; Repeat f18 with undefs optimistically treated as -2, -1.
define <4 x i32> @f20() {
; CHECK-LABEL: f20:
; CHECK: vrepig %v24, -2
; CHECK: br %r14
  ret <4 x i32> <i32 -1, i32 undef, i32 undef, i32 -2>
}
