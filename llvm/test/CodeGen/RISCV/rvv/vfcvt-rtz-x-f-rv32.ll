; RUN: llc -mtriple=riscv32 -mattr=+experimental-v,+f,+experimental-zfh -verify-machineinstrs \
; RUN:   --riscv-no-aliases < %s | FileCheck %s
declare <vscale x 1 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.nxv1i16.nxv1f16(
  <vscale x 1 x half>,
  i32);

define <vscale x 1 x i16> @intrinsic_vfcvt_rtz.x.f.v_nxv1i16_nxv1f16(<vscale x 1 x half> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_rtz.x.f.v_nxv1i16_nxv1f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,ta,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 1 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.nxv1i16.nxv1f16(
    <vscale x 1 x half> %0,
    i32 %1)

  ret <vscale x 1 x i16> %a
}

declare <vscale x 1 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv1i16.nxv1f16(
  <vscale x 1 x i16>,
  <vscale x 1 x half>,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x i16> @intrinsic_vfcvt_mask_rtz.x.f.v_nxv1i16_nxv1f16(<vscale x 1 x i16> %0, <vscale x 1 x half> %1, <vscale x 1 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_mask_rtz.x.f.v_nxv1i16_nxv1f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,tu,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 1 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv1i16.nxv1f16(
    <vscale x 1 x i16> %0,
    <vscale x 1 x half> %1,
    <vscale x 1 x i1> %2,
    i32 %3)

  ret <vscale x 1 x i16> %a
}

declare <vscale x 2 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.nxv2i16.nxv2f16(
  <vscale x 2 x half>,
  i32);

define <vscale x 2 x i16> @intrinsic_vfcvt_rtz.x.f.v_nxv2i16_nxv2f16(<vscale x 2 x half> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_rtz.x.f.v_nxv2i16_nxv2f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,ta,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.nxv2i16.nxv2f16(
    <vscale x 2 x half> %0,
    i32 %1)

  ret <vscale x 2 x i16> %a
}

declare <vscale x 2 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv2i16.nxv2f16(
  <vscale x 2 x i16>,
  <vscale x 2 x half>,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x i16> @intrinsic_vfcvt_mask_rtz.x.f.v_nxv2i16_nxv2f16(<vscale x 2 x i16> %0, <vscale x 2 x half> %1, <vscale x 2 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_mask_rtz.x.f.v_nxv2i16_nxv2f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,tu,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv2i16.nxv2f16(
    <vscale x 2 x i16> %0,
    <vscale x 2 x half> %1,
    <vscale x 2 x i1> %2,
    i32 %3)

  ret <vscale x 2 x i16> %a
}

declare <vscale x 4 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.nxv4i16.nxv4f16(
  <vscale x 4 x half>,
  i32);

define <vscale x 4 x i16> @intrinsic_vfcvt_rtz.x.f.v_nxv4i16_nxv4f16(<vscale x 4 x half> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_rtz.x.f.v_nxv4i16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,ta,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.nxv4i16.nxv4f16(
    <vscale x 4 x half> %0,
    i32 %1)

  ret <vscale x 4 x i16> %a
}

declare <vscale x 4 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv4i16.nxv4f16(
  <vscale x 4 x i16>,
  <vscale x 4 x half>,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x i16> @intrinsic_vfcvt_mask_rtz.x.f.v_nxv4i16_nxv4f16(<vscale x 4 x i16> %0, <vscale x 4 x half> %1, <vscale x 4 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_mask_rtz.x.f.v_nxv4i16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,tu,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv4i16.nxv4f16(
    <vscale x 4 x i16> %0,
    <vscale x 4 x half> %1,
    <vscale x 4 x i1> %2,
    i32 %3)

  ret <vscale x 4 x i16> %a
}

declare <vscale x 8 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.nxv8i16.nxv8f16(
  <vscale x 8 x half>,
  i32);

define <vscale x 8 x i16> @intrinsic_vfcvt_rtz.x.f.v_nxv8i16_nxv8f16(<vscale x 8 x half> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_rtz.x.f.v_nxv8i16_nxv8f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,ta,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 8 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.nxv8i16.nxv8f16(
    <vscale x 8 x half> %0,
    i32 %1)

  ret <vscale x 8 x i16> %a
}

declare <vscale x 8 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv8i16.nxv8f16(
  <vscale x 8 x i16>,
  <vscale x 8 x half>,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x i16> @intrinsic_vfcvt_mask_rtz.x.f.v_nxv8i16_nxv8f16(<vscale x 8 x i16> %0, <vscale x 8 x half> %1, <vscale x 8 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_mask_rtz.x.f.v_nxv8i16_nxv8f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,tu,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 8 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv8i16.nxv8f16(
    <vscale x 8 x i16> %0,
    <vscale x 8 x half> %1,
    <vscale x 8 x i1> %2,
    i32 %3)

  ret <vscale x 8 x i16> %a
}

declare <vscale x 16 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.nxv16i16.nxv16f16(
  <vscale x 16 x half>,
  i32);

define <vscale x 16 x i16> @intrinsic_vfcvt_rtz.x.f.v_nxv16i16_nxv16f16(<vscale x 16 x half> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_rtz.x.f.v_nxv16i16_nxv16f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,ta,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 16 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.nxv16i16.nxv16f16(
    <vscale x 16 x half> %0,
    i32 %1)

  ret <vscale x 16 x i16> %a
}

declare <vscale x 16 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv16i16.nxv16f16(
  <vscale x 16 x i16>,
  <vscale x 16 x half>,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x i16> @intrinsic_vfcvt_mask_rtz.x.f.v_nxv16i16_nxv16f16(<vscale x 16 x i16> %0, <vscale x 16 x half> %1, <vscale x 16 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_mask_rtz.x.f.v_nxv16i16_nxv16f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,tu,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 16 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv16i16.nxv16f16(
    <vscale x 16 x i16> %0,
    <vscale x 16 x half> %1,
    <vscale x 16 x i1> %2,
    i32 %3)

  ret <vscale x 16 x i16> %a
}

declare <vscale x 32 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.nxv32i16.nxv32f16(
  <vscale x 32 x half>,
  i32);

define <vscale x 32 x i16> @intrinsic_vfcvt_rtz.x.f.v_nxv32i16_nxv32f16(<vscale x 32 x half> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_rtz.x.f.v_nxv32i16_nxv32f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,ta,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 32 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.nxv32i16.nxv32f16(
    <vscale x 32 x half> %0,
    i32 %1)

  ret <vscale x 32 x i16> %a
}

declare <vscale x 32 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv32i16.nxv32f16(
  <vscale x 32 x i16>,
  <vscale x 32 x half>,
  <vscale x 32 x i1>,
  i32);

define <vscale x 32 x i16> @intrinsic_vfcvt_mask_rtz.x.f.v_nxv32i16_nxv32f16(<vscale x 32 x i16> %0, <vscale x 32 x half> %1, <vscale x 32 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_mask_rtz.x.f.v_nxv32i16_nxv32f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,tu,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 32 x i16> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv32i16.nxv32f16(
    <vscale x 32 x i16> %0,
    <vscale x 32 x half> %1,
    <vscale x 32 x i1> %2,
    i32 %3)

  ret <vscale x 32 x i16> %a
}

declare <vscale x 1 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.nxv1i32.nxv1f32(
  <vscale x 1 x float>,
  i32);

define <vscale x 1 x i32> @intrinsic_vfcvt_rtz.x.f.v_nxv1i32_nxv1f32(<vscale x 1 x float> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_rtz.x.f.v_nxv1i32_nxv1f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,mf2,ta,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 1 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.nxv1i32.nxv1f32(
    <vscale x 1 x float> %0,
    i32 %1)

  ret <vscale x 1 x i32> %a
}

declare <vscale x 1 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv1i32.nxv1f32(
  <vscale x 1 x i32>,
  <vscale x 1 x float>,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x i32> @intrinsic_vfcvt_mask_rtz.x.f.v_nxv1i32_nxv1f32(<vscale x 1 x i32> %0, <vscale x 1 x float> %1, <vscale x 1 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_mask_rtz.x.f.v_nxv1i32_nxv1f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,mf2,tu,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 1 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv1i32.nxv1f32(
    <vscale x 1 x i32> %0,
    <vscale x 1 x float> %1,
    <vscale x 1 x i1> %2,
    i32 %3)

  ret <vscale x 1 x i32> %a
}

declare <vscale x 2 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.nxv2i32.nxv2f32(
  <vscale x 2 x float>,
  i32);

define <vscale x 2 x i32> @intrinsic_vfcvt_rtz.x.f.v_nxv2i32_nxv2f32(<vscale x 2 x float> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_rtz.x.f.v_nxv2i32_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m1,ta,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.nxv2i32.nxv2f32(
    <vscale x 2 x float> %0,
    i32 %1)

  ret <vscale x 2 x i32> %a
}

declare <vscale x 2 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv2i32.nxv2f32(
  <vscale x 2 x i32>,
  <vscale x 2 x float>,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x i32> @intrinsic_vfcvt_mask_rtz.x.f.v_nxv2i32_nxv2f32(<vscale x 2 x i32> %0, <vscale x 2 x float> %1, <vscale x 2 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_mask_rtz.x.f.v_nxv2i32_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m1,tu,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv2i32.nxv2f32(
    <vscale x 2 x i32> %0,
    <vscale x 2 x float> %1,
    <vscale x 2 x i1> %2,
    i32 %3)

  ret <vscale x 2 x i32> %a
}

declare <vscale x 4 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.nxv4i32.nxv4f32(
  <vscale x 4 x float>,
  i32);

define <vscale x 4 x i32> @intrinsic_vfcvt_rtz.x.f.v_nxv4i32_nxv4f32(<vscale x 4 x float> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_rtz.x.f.v_nxv4i32_nxv4f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m2,ta,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.nxv4i32.nxv4f32(
    <vscale x 4 x float> %0,
    i32 %1)

  ret <vscale x 4 x i32> %a
}

declare <vscale x 4 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv4i32.nxv4f32(
  <vscale x 4 x i32>,
  <vscale x 4 x float>,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x i32> @intrinsic_vfcvt_mask_rtz.x.f.v_nxv4i32_nxv4f32(<vscale x 4 x i32> %0, <vscale x 4 x float> %1, <vscale x 4 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_mask_rtz.x.f.v_nxv4i32_nxv4f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m2,tu,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv4i32.nxv4f32(
    <vscale x 4 x i32> %0,
    <vscale x 4 x float> %1,
    <vscale x 4 x i1> %2,
    i32 %3)

  ret <vscale x 4 x i32> %a
}

declare <vscale x 8 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.nxv8i32.nxv8f32(
  <vscale x 8 x float>,
  i32);

define <vscale x 8 x i32> @intrinsic_vfcvt_rtz.x.f.v_nxv8i32_nxv8f32(<vscale x 8 x float> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_rtz.x.f.v_nxv8i32_nxv8f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m4,ta,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 8 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.nxv8i32.nxv8f32(
    <vscale x 8 x float> %0,
    i32 %1)

  ret <vscale x 8 x i32> %a
}

declare <vscale x 8 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv8i32.nxv8f32(
  <vscale x 8 x i32>,
  <vscale x 8 x float>,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x i32> @intrinsic_vfcvt_mask_rtz.x.f.v_nxv8i32_nxv8f32(<vscale x 8 x i32> %0, <vscale x 8 x float> %1, <vscale x 8 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_mask_rtz.x.f.v_nxv8i32_nxv8f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m4,tu,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 8 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv8i32.nxv8f32(
    <vscale x 8 x i32> %0,
    <vscale x 8 x float> %1,
    <vscale x 8 x i1> %2,
    i32 %3)

  ret <vscale x 8 x i32> %a
}

declare <vscale x 16 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.nxv16i32.nxv16f32(
  <vscale x 16 x float>,
  i32);

define <vscale x 16 x i32> @intrinsic_vfcvt_rtz.x.f.v_nxv16i32_nxv16f32(<vscale x 16 x float> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_rtz.x.f.v_nxv16i32_nxv16f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m8,ta,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 16 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.nxv16i32.nxv16f32(
    <vscale x 16 x float> %0,
    i32 %1)

  ret <vscale x 16 x i32> %a
}

declare <vscale x 16 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv16i32.nxv16f32(
  <vscale x 16 x i32>,
  <vscale x 16 x float>,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x i32> @intrinsic_vfcvt_mask_rtz.x.f.v_nxv16i32_nxv16f32(<vscale x 16 x i32> %0, <vscale x 16 x float> %1, <vscale x 16 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfcvt_mask_rtz.x.f.v_nxv16i32_nxv16f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m8,tu,mu
; CHECK:       vfcvt.rtz.x.f.v {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 16 x i32> @llvm.riscv.vfcvt.rtz.x.f.v.mask.nxv16i32.nxv16f32(
    <vscale x 16 x i32> %0,
    <vscale x 16 x float> %1,
    <vscale x 16 x i1> %2,
    i32 %3)

  ret <vscale x 16 x i32> %a
}
