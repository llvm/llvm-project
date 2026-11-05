; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define <4 x i32> @smin(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: @smin
; CHECK: call <4 x i32> @llvm.smin.v4i32(<4 x i32> %a, <4 x i32> %b)
  %r = call <4 x i32> @llvm.aarch64.neon.smin.v4i32(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %r
}

define <4 x i32> @smin_overloaded(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: @smin_overloaded
; CHECK: call <4 x i32> @llvm.smin.v4i32(<4 x i32> %a, <4 x i32> %b)
  %r = call <4 x i32> @llvm.aarch64.neon.smin(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %r
}

define <8 x i16> @smax(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: @smax
; CHECK: call <8 x i16> @llvm.smax.v8i16(<8 x i16> %a, <8 x i16> %b)
  %r = call <8 x i16> @llvm.aarch64.neon.smax.v8i16(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %r
}

define <8 x i16> @smax_overloaded(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: @smax_overloaded
; CHECK: call <8 x i16> @llvm.smax.v8i16(<8 x i16> %a, <8 x i16> %b)
  %r = call <8 x i16> @llvm.aarch64.neon.smax(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %r
}

define <16 x i8> @umin(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: @umin
; CHECK: call <16 x i8> @llvm.umin.v16i8(<16 x i8> %a, <16 x i8> %b)
  %r = call <16 x i8> @llvm.aarch64.neon.umin.v16i8(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %r
}

define <16 x i8> @umin_overloaded(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: @umin_overloaded
; CHECK: call <16 x i8> @llvm.umin.v16i8(<16 x i8> %a, <16 x i8> %b)
  %r = call <16 x i8> @llvm.aarch64.neon.umin(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %r
}

define <2 x i32> @umax(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: @umax
; CHECK: call <2 x i32> @llvm.umax.v2i32(<2 x i32> %a, <2 x i32> %b)
  %r = call <2 x i32> @llvm.aarch64.neon.umax.v2i32(<2 x i32> %a, <2 x i32> %b)
  ret <2 x i32> %r
}

define <2 x i32> @umax_overloaded(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: @umax_overloaded
; CHECK: call <2 x i32> @llvm.umax.v2i32(<2 x i32> %a, <2 x i32> %b)
  %r = call <2 x i32> @llvm.aarch64.neon.umax(<2 x i32> %a, <2 x i32> %b)
  ret <2 x i32> %r
}

define i32 @sminv_not_upgraded(<4 x i32> %a) {
; CHECK-LABEL: @sminv_not_upgraded
; CHECK: call i32 @llvm.aarch64.neon.sminv.i32.v4i32(<4 x i32> %a)
  %r = call i32 @llvm.aarch64.neon.sminv.i32.v4i32(<4 x i32> %a)
  ret i32 %r
}

define <4 x i32> @sminp_not_upgraded(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: @sminp_not_upgraded
; CHECK: call <4 x i32> @llvm.aarch64.neon.sminp.v4i32(<4 x i32> %a, <4 x i32> %b)
  %r = call <4 x i32> @llvm.aarch64.neon.sminp.v4i32(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %r
}

define <4 x i32> @umaxp_not_upgraded(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: @umaxp_not_upgraded
; CHECK: call <4 x i32> @llvm.aarch64.neon.umaxp.v4i32(<4 x i32> %a, <4 x i32> %b)
  %r = call <4 x i32> @llvm.aarch64.neon.umaxp.v4i32(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %r
}
