; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr7 < %s | FileCheck %s

; The generic CTTZ expansion should use a floating-point conversion when
; integer CTPOP/CTLZ expansion is unavailable.

define <4 x i32> @cttz_v4i32(<4 x i32> %x) {
; CHECK-LABEL: cttz_v4i32:
; CHECK: xvcvuxwdp
  %r = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %x, i1 false)
  ret <4 x i32> %r
}

define <4 x i32> @cttz_v4i32_zero_poison(<4 x i32> %x) {
; CHECK-LABEL: cttz_v4i32_zero_poison:
; CHECK: xvcvuxwdp
  %r = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %x, i1 true)
  ret <4 x i32> %r
}

declare <4 x i32> @llvm.cttz.v4i32(<4 x i32>, i1)
