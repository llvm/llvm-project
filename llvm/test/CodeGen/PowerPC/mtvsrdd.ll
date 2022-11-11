; RUN: llc -mcpu=pwr9 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s

; This test case checks r0 is used as constant 0 in instruction mtvsrdd.

define <2 x i64> @const0(i64 %a) {
  %vecinit = insertelement <2 x i64> undef, i64 %a, i32 0
  %vecinit1 = insertelement <2 x i64> %vecinit, i64 0, i32 1
  ret <2 x i64> %vecinit1
; CHECK-LABEL: const0
; CHECK: mtvsrdd v2, 0, r3
}

define <2 x i64> @noconst0(ptr %a, ptr %b) {
  %1 = load i64, ptr %a, align 8
  %2 = load i64, ptr %b, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %2, i32 0
  %vecinit1 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit1
; CHECK-LABEL: noconst0
; CHECK: mtvsrdd v2, {{r[0-9]+}}, {{r[0-9]+}}
}
