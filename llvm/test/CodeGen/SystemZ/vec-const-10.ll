; Test vector replicates, v2i64 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a byte-granularity replicate with the lowest useful value.
define <2 x i64> @f1() {
; CHECK-LABEL: f1:
; CHECK: vrepib %v24, 1
; CHECK: br %r14
  ret <2 x i64> <i64 72340172838076673, i64 72340172838076673>
}

; Test a byte-granularity replicate with an arbitrary value.
define <2 x i64> @f2() {
; CHECK-LABEL: f2:
; CHECK: vrepib %v24, -55
; CHECK: br %r14
  ret <2 x i64> <i64 -3906369333256140343, i64 -3906369333256140343>
}

; Test a byte-granularity replicate with the highest useful value.
define <2 x i64> @f3() {
; CHECK-LABEL: f3:
; CHECK: vrepib %v24, -2
; CHECK: br %r14
  ret <2 x i64> <i64 -72340172838076674, i64 -72340172838076674>
}

; Test a halfword-granularity replicate with the lowest useful value.
define <2 x i64> @f4() {
; CHECK-LABEL: f4:
; CHECK: vrepih %v24, 1
; CHECK: br %r14
  ret <2 x i64> <i64 281479271743489, i64 281479271743489>
}

; Test a halfword-granularity replicate with an arbitrary value.
define <2 x i64> @f5() {
; CHECK-LABEL: f5:
; CHECK: vrepih %v24, 25650
; CHECK: br %r14
  ret <2 x i64> <i64 7219943320220492850, i64 7219943320220492850>
}

; Test a halfword-granularity replicate with the highest useful value.
define <2 x i64> @f6() {
; CHECK-LABEL: f6:
; CHECK: vrepih %v24, -2
; CHECK: br %r14
  ret <2 x i64> <i64 -281479271743490, i64 -281479271743490>
}

; Test a word-granularity replicate with the lowest useful positive value.
define <2 x i64> @f7() {
; CHECK-LABEL: f7:
; CHECK: vrepif %v24, 1
; CHECK: br %r14
  ret <2 x i64> <i64 4294967297, i64 4294967297>
}

; Test a word-granularity replicate with the highest in-range value.
define <2 x i64> @f8() {
; CHECK-LABEL: f8:
; CHECK: vrepif %v24, 32767
; CHECK: br %r14
  ret <2 x i64> <i64 140733193420799, i64 140733193420799>
}

; Test a word-granularity replicate with the next highest value.
; This cannot use VREPIF.
define <2 x i64> @f9() {
; CHECK-LABEL: f9:
; CHECK-NOT: vrepif
; CHECK: br %r14
  ret <2 x i64> <i64 140737488388096, i64 140737488388096>
}

; Test a word-granularity replicate with the lowest in-range value.
define <2 x i64> @f10() {
; CHECK-LABEL: f10:
; CHECK: vrepif %v24, -32768
; CHECK: br %r14
  ret <2 x i64> <i64 -140733193420800, i64 -140733193420800>
}

; Test a word-granularity replicate with the next lowest value.
; This cannot use VREPIF.
define <2 x i64> @f11() {
; CHECK-LABEL: f11:
; CHECK-NOT: vrepif
; CHECK: br %r14
  ret <2 x i64> <i64 -140737488388097, i64 -140737488388097>
}

; Test a word-granularity replicate with the highest useful negative value.
define <2 x i64> @f12() {
; CHECK-LABEL: f12:
; CHECK: vrepif %v24, -2
; CHECK: br %r14
  ret <2 x i64> <i64 -4294967298, i64 -4294967298>
}

; Test a doubleword-granularity replicate with the lowest useful positive
; value.
define <2 x i64> @f13() {
; CHECK-LABEL: f13:
; CHECK: vrepig %v24, 1
; CHECK: br %r14
  ret <2 x i64> <i64 1, i64 1>
}

; Test a doubleword-granularity replicate with the highest in-range value.
define <2 x i64> @f14() {
; CHECK-LABEL: f14:
; CHECK: vrepig %v24, 32767
; CHECK: br %r14
  ret <2 x i64> <i64 32767, i64 32767>
}

; Test a doubleword-granularity replicate with the next highest value.
; This cannot use VREPIG.
define <2 x i64> @f15() {
; CHECK-LABEL: f15:
; CHECK-NOT: vrepig
; CHECK: br %r14
  ret <2 x i64> <i64 32768, i64 32768>
}

; Test a doubleword-granularity replicate with the lowest in-range value.
define <2 x i64> @f16() {
; CHECK-LABEL: f16:
; CHECK: vrepig %v24, -32768
; CHECK: br %r14
  ret <2 x i64> <i64 -32768, i64 -32768>
}

; Test a doubleword-granularity replicate with the next lowest value.
; This cannot use VREPIG.
define <2 x i64> @f17() {
; CHECK-LABEL: f17:
; CHECK-NOT: vrepig
; CHECK: br %r14
  ret <2 x i64> <i64 -32769, i64 -32769>
}

; Test a doubleword-granularity replicate with the highest useful negative
; value.
define <2 x i64> @f18() {
; CHECK-LABEL: f18:
; CHECK: vrepig %v24, -2
; CHECK: br %r14
  ret <2 x i64> <i64 -2, i64 -2>
}

; Repeat f14 with undefs optimistically treated as 32767.
define <2 x i64> @f19() {
; CHECK-LABEL: f19:
; CHECK: vrepig %v24, 32767
; CHECK: br %r14
  ret <2 x i64> <i64 undef, i64 32767>
}

; Repeat f18 with undefs optimistically treated as -2.
define <2 x i64> @f20() {
; CHECK-LABEL: f20:
; CHECK: vrepig %v24, -2
; CHECK: br %r14
  ret <2 x i64> <i64 undef, i64 -2>
}
