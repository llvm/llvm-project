; Test vector replicates, v8i16 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a byte-granularity replicate with the lowest useful value.
define <8 x i16> @f1() {
; CHECK-LABEL: f1:
; CHECK: vrepib %v24, 1
; CHECK: br %r14
  ret <8 x i16> <i16 257, i16 257, i16 257, i16 257,
                 i16 257, i16 257, i16 257, i16 257>
}

; Test a byte-granularity replicate with an arbitrary value.
define <8 x i16> @f2() {
; CHECK-LABEL: f2:
; CHECK: vrepib %v24, -55
; CHECK: br %r14
  ret <8 x i16> <i16 51657, i16 51657, i16 51657, i16 51657,
                 i16 51657, i16 51657, i16 51657, i16 51657>
}

; Test a byte-granularity replicate with the highest useful value.
define <8 x i16> @f3() {
; CHECK-LABEL: f3:
; CHECK: vrepib %v24, -2
; CHECK: br %r14
  ret <8 x i16> <i16 -258, i16 -258, i16 -258, i16 -258,
                 i16 -258, i16 -258, i16 -258, i16 -258>
}

; Test a halfword-granularity replicate with the lowest useful value.
define <8 x i16> @f4() {
; CHECK-LABEL: f4:
; CHECK: vrepih %v24, 1
; CHECK: br %r14
  ret <8 x i16> <i16 1, i16 1, i16 1, i16 1,
                 i16 1, i16 1, i16 1, i16 1>
}

; Test a halfword-granularity replicate with an arbitrary value.
define <8 x i16> @f5() {
; CHECK-LABEL: f5:
; CHECK: vrepih %v24, 25650
; CHECK: br %r14
  ret <8 x i16> <i16 25650, i16 25650, i16 25650, i16 25650,
                 i16 25650, i16 25650, i16 25650, i16 25650>
}

; Test a halfword-granularity replicate with the highest useful value.
define <8 x i16> @f6() {
; CHECK-LABEL: f6:
; CHECK: vrepih %v24, -2
; CHECK: br %r14
  ret <8 x i16> <i16 65534, i16 65534, i16 65534, i16 65534,
                 i16 65534, i16 65534, i16 65534, i16 65534>
}

; Test a word-granularity replicate with the lowest useful positive value.
define <8 x i16> @f7() {
; CHECK-LABEL: f7:
; CHECK: vrepif %v24, 1
; CHECK: br %r14
  ret <8 x i16> <i16 0, i16 1, i16 0, i16 1,
                 i16 0, i16 1, i16 0, i16 1>
}

; Test a word-granularity replicate with the highest in-range value.
define <8 x i16> @f8() {
; CHECK-LABEL: f8:
; CHECK: vrepif %v24, 32767
; CHECK: br %r14
  ret <8 x i16> <i16 0, i16 32767, i16 0, i16 32767,
                 i16 0, i16 32767, i16 0, i16 32767>
}

; Test a word-granularity replicate with the next highest value.
; This cannot use VREPIF.
define <8 x i16> @f9() {
; CHECK-LABEL: f9:
; CHECK-NOT: vrepif
; CHECK: br %r14
  ret <8 x i16> <i16 0, i16 32768, i16 0, i16 32768,
                 i16 0, i16 32768, i16 0, i16 32768>
}

; Test a word-granularity replicate with the lowest in-range value.
define <8 x i16> @f10() {
; CHECK-LABEL: f10:
; CHECK: vrepif %v24, -32768
; CHECK: br %r14
  ret <8 x i16> <i16 -1, i16 -32768, i16 -1, i16 -32768,
                 i16 -1, i16 -32768, i16 -1, i16 -32768>
}

; Test a word-granularity replicate with the next lowest value.
; This cannot use VREPIF.
define <8 x i16> @f11() {
; CHECK-LABEL: f11:
; CHECK-NOT: vrepif
; CHECK: br %r14
  ret <8 x i16> <i16 -1, i16 -32769, i16 -1, i16 -32769,
                 i16 -1, i16 -32769, i16 -1, i16 -32769>
}

; Test a word-granularity replicate with the highest useful negative value.
define <8 x i16> @f12() {
; CHECK-LABEL: f12:
; CHECK: vrepif %v24, -2
; CHECK: br %r14
  ret <8 x i16> <i16 -1, i16 -2, i16 -1, i16 -2,
                 i16 -1, i16 -2, i16 -1, i16 -2>
}

; Test a doubleword-granularity replicate with the lowest useful positive
; value.
define <8 x i16> @f13() {
; CHECK-LABEL: f13:
; CHECK: vrepig %v24, 1
; CHECK: br %r14
  ret <8 x i16> <i16 0, i16 0, i16 0, i16 1,
                 i16 0, i16 0, i16 0, i16 1>
}

; Test a doubleword-granularity replicate with the highest in-range value.
define <8 x i16> @f14() {
; CHECK-LABEL: f14:
; CHECK: vrepig %v24, 32767
; CHECK: br %r14
  ret <8 x i16> <i16 0, i16 0, i16 0, i16 32767,
                 i16 0, i16 0, i16 0, i16 32767>
}

; Test a doubleword-granularity replicate with the next highest value.
; This cannot use VREPIG.
define <8 x i16> @f15() {
; CHECK-LABEL: f15:
; CHECK-NOT: vrepig
; CHECK: br %r14
  ret <8 x i16> <i16 0, i16 0, i16 0, i16 32768,
                 i16 0, i16 0, i16 0, i16 32768>
}

; Test a doubleword-granularity replicate with the lowest in-range value.
define <8 x i16> @f16() {
; CHECK-LABEL: f16:
; CHECK: vrepig %v24, -32768
; CHECK: br %r14
  ret <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -32768,
                 i16 -1, i16 -1, i16 -1, i16 -32768>
}

; Test a doubleword-granularity replicate with the next lowest value.
; This cannot use VREPIG.
define <8 x i16> @f17() {
; CHECK-LABEL: f17:
; CHECK-NOT: vrepig
; CHECK: br %r14
  ret <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -32769,
                 i16 -1, i16 -1, i16 -1, i16 -32769>
}

; Test a doubleword-granularity replicate with the highest useful negative
; value.
define <8 x i16> @f18() {
; CHECK-LABEL: f18:
; CHECK: vrepig %v24, -2
; CHECK: br %r14
  ret <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -2,
                 i16 -1, i16 -1, i16 -1, i16 -2>
}

; Repeat f14 with undefs optimistically treated as 0.
define <8 x i16> @f19() {
; CHECK-LABEL: f19:
; CHECK: vrepig %v24, 32767
; CHECK: br %r14
  ret <8 x i16> <i16 0, i16 undef, i16 0, i16 32767,
                 i16 undef, i16 0, i16 undef, i16 32767>
}

; Repeat f18 with undefs optimistically treated as -1.
define <8 x i16> @f20() {
; CHECK-LABEL: f20:
; CHECK: vrepig %v24, -2
; CHECK: br %r14
  ret <8 x i16> <i16 -1, i16 -1, i16 undef, i16 -2,
                 i16 undef, i16 undef, i16 -1, i16 -2>
}
