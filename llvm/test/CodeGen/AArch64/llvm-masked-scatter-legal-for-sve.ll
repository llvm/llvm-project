; RUN: opt -mtriple=aarch64-linux-gnu -mattr=+sve -scalarize-masked-mem-intrin -S < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; Testing that masked scatters operating on scalable vectors that are
; packed in SVE registers are not scalarized.

; CHECK-LABEL: @masked_scatter_nxv4i32(
; CHECK: call void @llvm.masked.scatter.nxv4i32
define void @masked_scatter_nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32*> %ptrs, <vscale x 4 x i1> %masks) {
  call void @llvm.masked.scatter.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32*> %ptrs, i32 0, <vscale x 4 x i1> %masks)
  ret void
}

; Testing that masked scatters operating on scalable vectors of FP
; data that is packed in SVE registers are not scalarized.

; CHECK-LABEL: @masked_scatter_nxv2f64(
; CHECK: call void @llvm.masked.scatter.nxv2f64
define void @masked_scatter_nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x double*> %ptrs, <vscale x 2 x i1> %masks) {
  call void @llvm.masked.scatter.nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x double*> %ptrs, i32 0, <vscale x 2 x i1> %masks)
  ret void
}

; Testing that masked scatters operating on scalable vectors of FP
; data that is unpacked in SVE registers are not scalarized.

; CHECK-LABEL: @masked_scatter_nxv2f16(
; CHECK: call void @llvm.masked.scatter.nxv2f16
define void @masked_scatter_nxv2f16(<vscale x 2 x half> %data, <vscale x 2 x half*> %ptrs, <vscale x 2 x i1> %masks) {
  call void @llvm.masked.scatter.nxv2f16(<vscale x 2 x half> %data, <vscale x 2 x half*> %ptrs, i32 0, <vscale x 2 x i1> %masks)
  ret void
}

; Testing that masked scatters operating on 64-bit fixed vectors are
; scalarized because NEON doesn't have support for masked scatter
; instructions.

; CHECK-LABEL: @masked_scatter_v2f32(
; CHECK-NOT: @llvm.masked.scatter.v2f32(
define void @masked_scatter_v2f32(<2 x float> %data, <2 x float*> %ptrs, <2 x i1> %masks) {
  call void @llvm.masked.scatter.v2f32(<2 x float> %data, <2 x float*> %ptrs, i32 0, <2 x i1> %masks)
  ret void
}

; Testing that masked scatters operating on 128-bit fixed vectors are
; scalarized because NEON doesn't have support for masked scatter
; instructions and because we are not targeting fixed width SVE.

; CHECK-LABEL: @masked_scatter_v4i32(
; CHECK-NOT: @llvm.masked.scatter.v4i32(
define void @masked_scatter_v4i32(<4 x i32> %data, <4 x i32*> %ptrs, <4 x i1> %masks) {
  call void @llvm.masked.scatter.v4i32(<4 x i32> %data, <4 x i32*> %ptrs, i32 0, <4 x i1> %masks)
  ret void
}

declare void @llvm.masked.scatter.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32*> %ptrs, i32 %align, <vscale x 4 x i1> %masks)
declare void @llvm.masked.scatter.nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x double*> %ptrs, i32 %align, <vscale x 2 x i1> %masks)
declare void @llvm.masked.scatter.nxv2f16(<vscale x 2 x half> %data, <vscale x 2 x half*> %ptrs, i32 %align, <vscale x 2 x i1> %masks)
declare void @llvm.masked.scatter.v2f32(<2 x float> %data, <2 x float*> %ptrs, i32 %align, <2 x i1> %masks)
declare void @llvm.masked.scatter.v4i32(<4 x i32> %data, <4 x i32*> %ptrs, i32 %align, <4 x i1> %masks)
