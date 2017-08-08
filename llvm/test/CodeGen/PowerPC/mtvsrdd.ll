; RUN: llc -mcpu=pwr9 -ppc-vsr-nums-as-vr -mtriple=powerpc64le-unknown-unknown \
; RUN:   < %s | FileCheck %s

; This test case checks r0 is used as constant 0 in instruction mtvsrdd.

define <2 x i64> @const0(i64 %a) {
  %vecinit = insertelement <2 x i64> undef, i64 %a, i32 0
  %vecinit1 = insertelement <2 x i64> %vecinit, i64 0, i32 1
  ret <2 x i64> %vecinit1
; CHECK-LABEL: const0
; CHECK: mtvsrdd v2, 0, r3
}

define <2 x i64> @noconst0(i64* %a, i64* %b) {
  %1 = load i64, i64* %a, align 8
  %2 = load i64, i64* %b, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %2, i32 0
  %vecinit1 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit1
; CHECK-LABEL: noconst0
; CHECK: mtvsrdd v2, {{r[0-9]+}}, {{r[0-9]+}}
}
