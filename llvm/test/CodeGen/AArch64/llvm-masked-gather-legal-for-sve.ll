; RUN: opt -mtriple=aarch64-linux-gnu -mattr=+sve -passes=scalarize-masked-mem-intrin -S < %s | FileCheck %s

; Testing that masked gathers operating on scalable vectors that are
; packed in SVE registers are not scalarized.

; CHECK-LABEL: @masked_gather_nxv4i32(
; CHECK: call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32
define <vscale x 4 x i32> @masked_gather_nxv4i32(<vscale x 4 x ptr> %ld, <vscale x 4 x i1> %masks, <vscale x 4 x i32> %passthru) {
  %res = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32(<vscale x 4 x ptr> %ld, i32 0, <vscale x 4 x i1> %masks, <vscale x 4 x i32> %passthru)
  ret <vscale x 4 x i32> %res
}

; Testing that masked gathers operating on scalable vectors of FP data
; that is packed in SVE registers are not scalarized.

; CHECK-LABEL: @masked_gather_nxv2f64(
; CHECK: call <vscale x 2 x double> @llvm.masked.gather.nxv2f64
define <vscale x 2 x double> @masked_gather_nxv2f64(<vscale x 2 x ptr> %ld, <vscale x 2 x i1> %masks, <vscale x 2 x double> %passthru) {
  %res = call <vscale x 2 x double> @llvm.masked.gather.nxv2f64(<vscale x 2 x ptr> %ld, i32 0, <vscale x 2 x i1> %masks, <vscale x 2 x double> %passthru)
  ret <vscale x 2 x double> %res
}

; Testing that masked gathers operating on scalable vectors of FP data
; that is unpacked in SVE registers are not scalarized.

; CHECK-LABEL: @masked_gather_nxv2f16(
; CHECK: call <vscale x 2 x half> @llvm.masked.gather.nxv2f16
define <vscale x 2 x half> @masked_gather_nxv2f16(<vscale x 2 x ptr> %ld, <vscale x 2 x i1> %masks, <vscale x 2 x half> %passthru) {
  %res = call <vscale x 2 x half> @llvm.masked.gather.nxv2f16(<vscale x 2 x ptr> %ld, i32 0, <vscale x 2 x i1> %masks, <vscale x 2 x half> %passthru)
  ret <vscale x 2 x half> %res
}

; Testing that masked gathers operating on 64-bit fixed vectors are
; scalarized because NEON doesn't have support for masked gather
; instructions.

; CHECK-LABEL: @masked_gather_v2f32(
; CHECK-NOT: @llvm.masked.gather.v2f32(
define <2 x float> @masked_gather_v2f32(<2 x ptr> %ld, <2 x i1> %masks, <2 x float> %passthru) {
  %res = call <2 x float> @llvm.masked.gather.v2f32(<2 x ptr> %ld, i32 0, <2 x i1> %masks, <2 x float> %passthru)
  ret <2 x float> %res
}

; Testing that masked gathers operating on 128-bit fixed vectors are
; scalarized because NEON doesn't have support for masked gather
; instructions and because we are not targeting fixed width SVE.

; CHECK-LABEL: @masked_gather_v4i32(
; CHECK-NOT: @llvm.masked.gather.v4i32(
define <4 x i32> @masked_gather_v4i32(<4 x ptr> %ld, <4 x i1> %masks, <4 x i32> %passthru) {
  %res = call <4 x i32> @llvm.masked.gather.v4i32(<4 x ptr> %ld, i32 0, <4 x i1> %masks, <4 x i32> %passthru)
  ret <4 x i32> %res
}

declare <vscale x 4 x i32> @llvm.masked.gather.nxv4i32(<vscale x 4 x ptr> %ptrs, i32 %align, <vscale x 4 x i1> %masks, <vscale x 4 x i32> %passthru)
declare <vscale x 2 x double> @llvm.masked.gather.nxv2f64(<vscale x 2 x ptr> %ptrs, i32 %align, <vscale x 2 x i1> %masks, <vscale x 2 x double> %passthru)
declare <vscale x 2 x half> @llvm.masked.gather.nxv2f16(<vscale x 2 x ptr> %ptrs, i32 %align, <vscale x 2 x i1> %masks, <vscale x 2 x half> %passthru)
declare <2 x float> @llvm.masked.gather.v2f32(<2 x ptr> %ptrs, i32 %align, <2 x i1> %masks, <2 x float> %passthru)
declare <4 x i32> @llvm.masked.gather.v4i32(<4 x ptr> %ptrs, i32 %align, <4 x i1> %masks, <4 x i32> %passthru)
