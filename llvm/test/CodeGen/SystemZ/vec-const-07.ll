; Test vector replicates, v16i8 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a byte-granularity replicate with the lowest useful value.
define <16 x i8> @f1() {
; CHECK-LABEL: f1:
; CHECK: vrepib %v24, 1
; CHECK: br %r14
  ret <16 x i8> <i8 1, i8 1, i8 1, i8 1,
                 i8 1, i8 1, i8 1, i8 1,
                 i8 1, i8 1, i8 1, i8 1,
                 i8 1, i8 1, i8 1, i8 1>
}

; Test a byte-granularity replicate with an arbitrary value.
define <16 x i8> @f2() {
; CHECK-LABEL: f2:
; CHECK: vrepib %v24, -55
; CHECK: br %r14
  ret <16 x i8> <i8 201, i8 201, i8 201, i8 201,
                 i8 201, i8 201, i8 201, i8 201,
                 i8 201, i8 201, i8 201, i8 201,
                 i8 201, i8 201, i8 201, i8 201>
}

; Test a byte-granularity replicate with the highest useful value.
define <16 x i8> @f3() {
; CHECK-LABEL: f3:
; CHECK: vrepib %v24, -2
; CHECK: br %r14
  ret <16 x i8> <i8 254, i8 254, i8 254, i8 254,
                 i8 254, i8 254, i8 254, i8 254,
                 i8 254, i8 254, i8 254, i8 254,
                 i8 254, i8 254, i8 254, i8 254>
}

; Test a halfword-granularity replicate with the lowest useful value.
define <16 x i8> @f4() {
; CHECK-LABEL: f4:
; CHECK: vrepih %v24, 1
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 1, i8 0, i8 1,
                 i8 0, i8 1, i8 0, i8 1,
                 i8 0, i8 1, i8 0, i8 1,
                 i8 0, i8 1, i8 0, i8 1>
}

; Test a halfword-granularity replicate with an arbitrary value.
define <16 x i8> @f5() {
; CHECK-LABEL: f5:
; CHECK: vrepih %v24, 25650
; CHECK: br %r14
  ret <16 x i8> <i8 100, i8 50, i8 100, i8 50,
                 i8 100, i8 50, i8 100, i8 50,
                 i8 100, i8 50, i8 100, i8 50,
                 i8 100, i8 50, i8 100, i8 50>
}

; Test a halfword-granularity replicate with the highest useful value.
define <16 x i8> @f6() {
; CHECK-LABEL: f6:
; CHECK: vrepih %v24, -2
; CHECK: br %r14
  ret <16 x i8> <i8 255, i8 254, i8 255, i8 254,
                 i8 255, i8 254, i8 255, i8 254,
                 i8 255, i8 254, i8 255, i8 254,
                 i8 255, i8 254, i8 255, i8 254>
}

; Test a word-granularity replicate with the lowest useful positive value.
define <16 x i8> @f7() {
; CHECK-LABEL: f7:
; CHECK: vrepif %v24, 1
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 0, i8 0, i8 1,
                 i8 0, i8 0, i8 0, i8 1,
                 i8 0, i8 0, i8 0, i8 1,
                 i8 0, i8 0, i8 0, i8 1>
}

; Test a word-granularity replicate with the highest in-range value.
define <16 x i8> @f8() {
; CHECK-LABEL: f8:
; CHECK: vrepif %v24, 32767
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 0, i8 127, i8 255,
                 i8 0, i8 0, i8 127, i8 255,
                 i8 0, i8 0, i8 127, i8 255,
                 i8 0, i8 0, i8 127, i8 255>
}

; Test a word-granularity replicate with the next highest value.
; This cannot use VREPIF.
define <16 x i8> @f9() {
; CHECK-LABEL: f9:
; CHECK-NOT: vrepif
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 0, i8 128, i8 0,
                 i8 0, i8 0, i8 128, i8 0,
                 i8 0, i8 0, i8 128, i8 0,
                 i8 0, i8 0, i8 128, i8 0>
}

; Test a word-granularity replicate with the lowest in-range value.
define <16 x i8> @f10() {
; CHECK-LABEL: f10:
; CHECK: vrepif %v24, -32768
; CHECK: br %r14
  ret <16 x i8> <i8 255, i8 255, i8 128, i8 0,
                 i8 255, i8 255, i8 128, i8 0,
                 i8 255, i8 255, i8 128, i8 0,
                 i8 255, i8 255, i8 128, i8 0>
}

; Test a word-granularity replicate with the next lowest value.
; This cannot use VREPIF.
define <16 x i8> @f11() {
; CHECK-LABEL: f11:
; CHECK-NOT: vrepif
; CHECK: br %r14
  ret <16 x i8> <i8 255, i8 255, i8 127, i8 255,
                 i8 255, i8 255, i8 127, i8 255,
                 i8 255, i8 255, i8 127, i8 255,
                 i8 255, i8 255, i8 127, i8 255>
}

; Test a word-granularity replicate with the highest useful negative value.
define <16 x i8> @f12() {
; CHECK-LABEL: f12:
; CHECK: vrepif %v24, -2
; CHECK: br %r14
  ret <16 x i8> <i8 255, i8 255, i8 255, i8 254,
                 i8 255, i8 255, i8 255, i8 254,
                 i8 255, i8 255, i8 255, i8 254,
                 i8 255, i8 255, i8 255, i8 254>
}

; Test a doubleword-granularity replicate with the lowest useful positive
; value.
define <16 x i8> @f13() {
; CHECK-LABEL: f13:
; CHECK: vrepig %v24, 1
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 0, i8 0, i8 0,
                 i8 0, i8 0, i8 0, i8 1,
                 i8 0, i8 0, i8 0, i8 0,
                 i8 0, i8 0, i8 0, i8 1>
}

; Test a doubleword-granularity replicate with the highest in-range value.
define <16 x i8> @f14() {
; CHECK-LABEL: f14:
; CHECK: vrepig %v24, 32767
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 0, i8 0, i8 0,
                 i8 0, i8 0, i8 127, i8 255,
                 i8 0, i8 0, i8 0, i8 0,
                 i8 0, i8 0, i8 127, i8 255>
}

; Test a doubleword-granularity replicate with the next highest value.
; This cannot use VREPIG.
define <16 x i8> @f15() {
; CHECK-LABEL: f15:
; CHECK-NOT: vrepig
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 0, i8 0, i8 0,
                 i8 0, i8 0, i8 128, i8 0,
                 i8 0, i8 0, i8 0, i8 0,
                 i8 0, i8 0, i8 128, i8 0>
}

; Test a doubleword-granularity replicate with the lowest in-range value.
define <16 x i8> @f16() {
; CHECK-LABEL: f16:
; CHECK: vrepig %v24, -32768
; CHECK: br %r14
  ret <16 x i8> <i8 255, i8 255, i8 255, i8 255,
                 i8 255, i8 255, i8 128, i8 0,
                 i8 255, i8 255, i8 255, i8 255,
                 i8 255, i8 255, i8 128, i8 0>
}

; Test a doubleword-granularity replicate with the next lowest value.
; This cannot use VREPIG.
define <16 x i8> @f17() {
; CHECK-LABEL: f17:
; CHECK-NOT: vrepig
; CHECK: br %r14
  ret <16 x i8> <i8 255, i8 255, i8 255, i8 255,
                 i8 255, i8 255, i8 127, i8 255,
                 i8 255, i8 255, i8 255, i8 255,
                 i8 255, i8 255, i8 127, i8 255>
}

; Test a doubleword-granularity replicate with the highest useful negative
; value.
define <16 x i8> @f18() {
; CHECK-LABEL: f18:
; CHECK: vrepig %v24, -2
; CHECK: br %r14
  ret <16 x i8> <i8 255, i8 255, i8 255, i8 255,
                 i8 255, i8 255, i8 255, i8 254,
                 i8 255, i8 255, i8 255, i8 255,
                 i8 255, i8 255, i8 255, i8 254>
}

; Repeat f14 with undefs optimistically treated as 0.
define <16 x i8> @f19() {
; CHECK-LABEL: f19:
; CHECK: vrepig %v24, 32767
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 undef, i8 0, i8 0,
                 i8 0, i8 0, i8 127, i8 255,
                 i8 undef, i8 0, i8 undef, i8 0,
                 i8 0, i8 0, i8 127, i8 255>
}

; Repeat f18 with undefs optimistically treated as -1.
define <16 x i8> @f20() {
; CHECK-LABEL: f20:
; CHECK: vrepig %v24, -2
; CHECK: br %r14
  ret <16 x i8> <i8 undef, i8 255, i8 255, i8 255,
                 i8 255, i8 255, i8 undef, i8 254,
                 i8 255, i8 255, i8 255, i8 undef,
                 i8 255, i8 undef, i8 255, i8 254>
}
