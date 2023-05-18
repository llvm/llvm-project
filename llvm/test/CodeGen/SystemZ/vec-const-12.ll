; Test vector replicates, v2f64 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a byte-granularity replicate with the lowest useful value.
define <2 x double> @f1() {
; CHECK-LABEL: f1:
; CHECK: vrepib %v24, 1
; CHECK: br %r14
  ret <2 x double> <double 0x0101010101010101, double 0x0101010101010101>
}

; Test a byte-granularity replicate with an arbitrary value.
define <2 x double> @f2() {
; CHECK-LABEL: f2:
; CHECK: vrepib %v24, -55
; CHECK: br %r14
  ret <2 x double> <double 0xc9c9c9c9c9c9c9c9, double 0xc9c9c9c9c9c9c9c9>
}

; Test a byte-granularity replicate with the highest useful value.
define <2 x double> @f3() {
; CHECK-LABEL: f3:
; CHECK: vrepib %v24, -2
; CHECK: br %r14
  ret <2 x double> <double 0xfefefefefefefefe, double 0xfefefefefefefefe>
}

; Test a halfword-granularity replicate with the lowest useful value.
define <2 x double> @f4() {
; CHECK-LABEL: f4:
; CHECK: vrepih %v24, 1
; CHECK: br %r14
  ret <2 x double> <double 0x0001000100010001, double 0x0001000100010001>
}

; Test a halfword-granularity replicate with an arbitrary value.
define <2 x double> @f5() {
; CHECK-LABEL: f5:
; CHECK: vrepih %v24, 25650
; CHECK: br %r14
  ret <2 x double> <double 0x6432643264326432, double 0x6432643264326432>
}

; Test a halfword-granularity replicate with the highest useful value.
define <2 x double> @f6() {
; CHECK-LABEL: f6:
; CHECK: vrepih %v24, -2
; CHECK: br %r14
  ret <2 x double> <double 0xfffefffefffefffe, double 0xfffefffefffefffe>
}

; Test a word-granularity replicate with the lowest useful positive value.
define <2 x double> @f7() {
; CHECK-LABEL: f7:
; CHECK: vrepif %v24, 1
; CHECK: br %r14
  ret <2 x double> <double 0x0000000100000001, double 0x0000000100000001>
}

; Test a word-granularity replicate with the highest in-range value.
define <2 x double> @f8() {
; CHECK-LABEL: f8:
; CHECK: vrepif %v24, 32767
; CHECK: br %r14
  ret <2 x double> <double 0x00007fff00007fff, double 0x00007fff00007fff>
}

; Test a word-granularity replicate with the next highest value.
; This cannot use VREPIF.
define <2 x double> @f9() {
; CHECK-LABEL: f9:
; CHECK-NOT: vrepif
; CHECK: br %r14
  ret <2 x double> <double 0x0000800000008000, double 0x0000800000008000>
}

; Test a word-granularity replicate with the lowest in-range value.
define <2 x double> @f10() {
; CHECK-LABEL: f10:
; CHECK: vrepif %v24, -32768
; CHECK: br %r14
  ret <2 x double> <double 0xffff8000ffff8000, double 0xffff8000ffff8000>
}

; Test a word-granularity replicate with the next lowest value.
; This cannot use VREPIF.
define <2 x double> @f11() {
; CHECK-LABEL: f11:
; CHECK-NOT: vrepif
; CHECK: br %r14
  ret <2 x double> <double 0xffff7fffffff7fff, double 0xffff7fffffff7fff>
}

; Test a word-granularity replicate with the highest useful negative value.
define <2 x double> @f12() {
; CHECK-LABEL: f12:
; CHECK: vrepif %v24, -2
; CHECK: br %r14
  ret <2 x double> <double 0xfffffffefffffffe, double 0xfffffffefffffffe>
}

; Test a doubleword-granularity replicate with the lowest useful positive
; value.
define <2 x double> @f13() {
; CHECK-LABEL: f13:
; CHECK: vrepig %v24, 1
; CHECK: br %r14
  ret <2 x double> <double 0x0000000000000001, double 0x0000000000000001>
}

; Test a doubleword-granularity replicate with the highest in-range value.
define <2 x double> @f14() {
; CHECK-LABEL: f14:
; CHECK: vrepig %v24, 32767
; CHECK: br %r14
  ret <2 x double> <double 0x0000000000007fff, double 0x0000000000007fff>
}

; Test a doubleword-granularity replicate with the next highest value.
; This cannot use VREPIG.
define <2 x double> @f15() {
; CHECK-LABEL: f15:
; CHECK-NOT: vrepig
; CHECK: br %r14
  ret <2 x double> <double 0x0000000000008000, double 0x0000000000008000>
}

; Test a doubleword-granularity replicate with the lowest in-range value.
define <2 x double> @f16() {
; CHECK-LABEL: f16:
; CHECK: vrepig %v24, -32768
; CHECK: br %r14
  ret <2 x double> <double 0xffffffffffff8000, double 0xffffffffffff8000>
}

; Test a doubleword-granularity replicate with the next lowest value.
; This cannot use VREPIG.
define <2 x double> @f17() {
; CHECK-LABEL: f17:
; CHECK-NOT: vrepig
; CHECK: br %r14
  ret <2 x double> <double 0xffffffffffff7fff, double 0xffffffffffff7fff>
}

; Test a doubleword-granularity replicate with the highest useful negative
; value.
define <2 x double> @f18() {
; CHECK-LABEL: f18:
; CHECK: vrepig %v24, -2
; CHECK: br %r14
  ret <2 x double> <double 0xfffffffffffffffe, double 0xfffffffffffffffe>
}

; Repeat f14 with undefs optimistically treated as 32767.
define <2 x double> @f19() {
; CHECK-LABEL: f19:
; CHECK: vrepig %v24, 32767
; CHECK: br %r14
  ret <2 x double> <double undef, double 0x0000000000007fff>
}

; Repeat f18 with undefs optimistically treated as -2.
define <2 x double> @f20() {
; CHECK-LABEL: f20:
; CHECK: vrepig %v24, -2
; CHECK: br %r14
  ret <2 x double> <double undef, double 0xfffffffffffffffe>
}
