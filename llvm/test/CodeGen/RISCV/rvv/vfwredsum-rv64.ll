; RUN: llc -mtriple=riscv64 -mattr=+experimental-v,+d,+experimental-zfh -verify-machineinstrs \
; RUN:   --riscv-no-aliases < %s | FileCheck %s
declare <vscale x 2 x float> @llvm.riscv.vfwredsum.nxv2f32.nxv32f16(
  <vscale x 2 x float>,
  <vscale x 32 x half>,
  <vscale x 2 x float>,
  i64);

define <vscale x 2 x float> @intrinsic_vfwredsum_vs_nxv2f32_nxv32f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 32 x half> %1, <vscale x 2 x float> %2, i64 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwredsum_vs_nxv2f32_nxv32f16_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,ta,mu
; CHECK:       vfwredsum.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredsum.nxv2f32.nxv32f16(
    <vscale x 2 x float> %0,
    <vscale x 32 x half> %1,
    <vscale x 2 x float> %2,
    i64 %3)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwredsum.mask.nxv2f32.nxv32f16.nxv32i1(
  <vscale x 2 x float>,
  <vscale x 32 x half>,
  <vscale x 2 x float>,
  <vscale x 32 x i1>,
  i64);

define <vscale x 2 x float> @intrinsic_vfwredsum_mask_vs_nxv2f32_nxv32f16_nxv2f32(<vscale x 2 x float> %0, <vscale x 32 x half> %1, <vscale x 2 x float> %2, <vscale x 32 x i1> %3, i64 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwredsum_mask_vs_nxv2f32_nxv32f16_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,ta,mu
; CHECK:       vfwredsum.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x float> @llvm.riscv.vfwredsum.mask.nxv2f32.nxv32f16.nxv32i1(
    <vscale x 2 x float> %0,
    <vscale x 32 x half> %1,
    <vscale x 2 x float> %2,
    <vscale x 32 x i1> %3,
    i64 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 1 x double> @llvm.riscv.vfwredsum.nxv1f64.nxv16f32(
  <vscale x 1 x double>,
  <vscale x 16 x float>,
  <vscale x 1 x double>,
  i64);

define <vscale x 1 x double> @intrinsic_vfwredsum_vs_nxv1f64_nxv16f32_nxv1f64(<vscale x 1 x double> %0, <vscale x 16 x float> %1, <vscale x 1 x double> %2, i64 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwredsum_vs_nxv1f64_nxv16f32_nxv1f64
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m8,ta,mu
; CHECK:       vfwredsum.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 1 x double> @llvm.riscv.vfwredsum.nxv1f64.nxv16f32(
    <vscale x 1 x double> %0,
    <vscale x 16 x float> %1,
    <vscale x 1 x double> %2,
    i64 %3)

  ret <vscale x 1 x double> %a
}

declare <vscale x 1 x double> @llvm.riscv.vfwredsum.mask.nxv1f64.nxv16f32.nxv16i1(
  <vscale x 1 x double>,
  <vscale x 16 x float>,
  <vscale x 1 x double>,
  <vscale x 16 x i1>,
  i64);

define <vscale x 1 x double> @intrinsic_vfwredsum_mask_vs_nxv1f64_nxv16f32_nxv1f64(<vscale x 1 x double> %0, <vscale x 16 x float> %1, <vscale x 1 x double> %2, <vscale x 16 x i1> %3, i64 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwredsum_mask_vs_nxv1f64_nxv16f32_nxv1f64
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m8,ta,mu
; CHECK:       vfwredsum.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 1 x double> @llvm.riscv.vfwredsum.mask.nxv1f64.nxv16f32.nxv16i1(
    <vscale x 1 x double> %0,
    <vscale x 16 x float> %1,
    <vscale x 1 x double> %2,
    <vscale x 16 x i1> %3,
    i64 %4)

  ret <vscale x 1 x double> %a
}
