; RUN: llc < %s -mcpu=sm_100 -mattr=+ptx88 -march=nvptx64 | FileCheck %s
; RUN: %if ptxas-sm_100 && ptxas-isa-8.8 %{ llc < %s -mcpu=sm_100 -mattr=+ptx88 -march=nvptx64 | %ptxas-verify -arch=sm_100 %}

target triple = "nvptx64-nvidia-cuda"

define <2 x float> @add_rn(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: add_rn(
; CHECK: add.rn.f32x2
  %r = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %a, <2 x float> %b, i32 1)
  ret <2 x float> %r
}

define <2 x float> @add_rz(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: add_rz(
; CHECK: add.rz.f32x2
  %r = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %a, <2 x float> %b, i32 0)
  ret <2 x float> %r
}

define <2 x float> @add_rm(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: add_rm(
; CHECK: add.rm.f32x2
  %r = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %a, <2 x float> %b, i32 3)
  ret <2 x float> %r
}

define <2 x float> @add_rp(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: add_rp(
; CHECK: add.rp.f32x2
  %r = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %a, <2 x float> %b, i32 2)
  ret <2 x float> %r
}

define <2 x float> @add_rn_ftz(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: add_rn_ftz(
; CHECK: add.rn.ftz.f32x2
  %r = call <2 x float> @llvm.nvvm.fadd.ftz.v2f32(<2 x float> %a, <2 x float> %b, i32 1)
  ret <2 x float> %r
}

define <2 x float> @add_rz_ftz(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: add_rz_ftz(
; CHECK: add.rz.ftz.f32x2
  %r = call <2 x float> @llvm.nvvm.fadd.ftz.v2f32(<2 x float> %a, <2 x float> %b, i32 0)
  ret <2 x float> %r
}

define <2 x float> @add_rm_ftz(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: add_rm_ftz(
; CHECK: add.rm.ftz.f32x2
  %r = call <2 x float> @llvm.nvvm.fadd.ftz.v2f32(<2 x float> %a, <2 x float> %b, i32 3)
  ret <2 x float> %r
}

define <2 x float> @add_rp_ftz(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: add_rp_ftz(
; CHECK: add.rp.ftz.f32x2
  %r = call <2 x float> @llvm.nvvm.fadd.ftz.v2f32(<2 x float> %a, <2 x float> %b, i32 2)
  ret <2 x float> %r
}
