; RUN: llc -mtriple=riscv32 -mattr=+experimental-v,+f,+experimental-zfh -verify-machineinstrs \
; RUN:   --riscv-no-aliases < %s | FileCheck %s
declare <vscale x 2 x float> @llvm.riscv.vfwredosum.nxv2f32.nxv32f16(
  <vscale x 2 x float>,
  <vscale x 32 x half>,
  <vscale x 2 x float>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredosum_vs_nxv2f32_nxv32f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 32 x half> %1, <vscale x 2 x float> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwredosum_vs_nxv2f32_nxv32f16_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,ta,mu
; CHECK:       vfwredosum.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredosum.nxv2f32.nxv32f16(
    <vscale x 2 x float> %0,
    <vscale x 32 x half> %1,
    <vscale x 2 x float> %2,
    i32 %3)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwredosum.mask.nxv2f32.nxv32f16.nxv32i1(
  <vscale x 2 x float>,
  <vscale x 32 x half>,
  <vscale x 2 x float>,
  <vscale x 32 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwredosum_mask_vs_nxv2f32_nxv32f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 32 x half> %1, <vscale x 2 x float> %2, <vscale x 32 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwredosum_mask_vs_nxv2f32_nxv32f16_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,ta,mu
; CHECK:       vfwredosum.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredosum.mask.nxv2f32.nxv32f16.nxv32i1(
    <vscale x 2 x float> %0,
    <vscale x 32 x half> %1,
    <vscale x 2 x float> %2,
    <vscale x 32 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}
