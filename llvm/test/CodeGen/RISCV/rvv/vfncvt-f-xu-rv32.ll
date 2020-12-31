; RUN: llc -mtriple=riscv32 -mattr=+experimental-v,+f,+experimental-zfh -verify-machineinstrs \
; RUN:   --riscv-no-aliases < %s | FileCheck %s
declare <vscale x 1 x half> @llvm.riscv.vfncvt.f.xu.w.nxv1f16.nxv1i32(
  <vscale x 1 x i32>,
  i32);

define <vscale x 1 x half> @intrinsic_vfncvt_f.xu.w_nxv1f16_nxv1i32(<vscale x 1 x i32> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfncvt_f.xu.w_nxv1f16_nxv1i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,ta,mu
; CHECK:       vfncvt.f.xu.w {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 1 x half> @llvm.riscv.vfncvt.f.xu.w.nxv1f16.nxv1i32(
    <vscale x 1 x i32> %0,
    i32 %1)

  ret <vscale x 1 x half> %a
}

declare <vscale x 1 x half> @llvm.riscv.vfncvt.f.xu.w.mask.nxv1f16.nxv1i32(
  <vscale x 1 x half>,
  <vscale x 1 x i32>,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x half> @intrinsic_vfncvt_mask_f.xu.w_nxv1f16_nxv1i32(<vscale x 1 x half> %0, <vscale x 1 x i32> %1, <vscale x 1 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfncvt_mask_f.xu.w_nxv1f16_nxv1i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,tu,mu
; CHECK:       vfncvt.f.xu.w {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 1 x half> @llvm.riscv.vfncvt.f.xu.w.mask.nxv1f16.nxv1i32(
    <vscale x 1 x half> %0,
    <vscale x 1 x i32> %1,
    <vscale x 1 x i1> %2,
    i32 %3)

  ret <vscale x 1 x half> %a
}

declare <vscale x 2 x half> @llvm.riscv.vfncvt.f.xu.w.nxv2f16.nxv2i32(
  <vscale x 2 x i32>,
  i32);

define <vscale x 2 x half> @intrinsic_vfncvt_f.xu.w_nxv2f16_nxv2i32(<vscale x 2 x i32> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfncvt_f.xu.w_nxv2f16_nxv2i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,ta,mu
; CHECK:       vfncvt.f.xu.w {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x half> @llvm.riscv.vfncvt.f.xu.w.nxv2f16.nxv2i32(
    <vscale x 2 x i32> %0,
    i32 %1)

  ret <vscale x 2 x half> %a
}

declare <vscale x 2 x half> @llvm.riscv.vfncvt.f.xu.w.mask.nxv2f16.nxv2i32(
  <vscale x 2 x half>,
  <vscale x 2 x i32>,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x half> @intrinsic_vfncvt_mask_f.xu.w_nxv2f16_nxv2i32(<vscale x 2 x half> %0, <vscale x 2 x i32> %1, <vscale x 2 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfncvt_mask_f.xu.w_nxv2f16_nxv2i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,tu,mu
; CHECK:       vfncvt.f.xu.w {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x half> @llvm.riscv.vfncvt.f.xu.w.mask.nxv2f16.nxv2i32(
    <vscale x 2 x half> %0,
    <vscale x 2 x i32> %1,
    <vscale x 2 x i1> %2,
    i32 %3)

  ret <vscale x 2 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfncvt.f.xu.w.nxv4f16.nxv4i32(
  <vscale x 4 x i32>,
  i32);

define <vscale x 4 x half> @intrinsic_vfncvt_f.xu.w_nxv4f16_nxv4i32(<vscale x 4 x i32> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfncvt_f.xu.w_nxv4f16_nxv4i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,ta,mu
; CHECK:       vfncvt.f.xu.w {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x half> @llvm.riscv.vfncvt.f.xu.w.nxv4f16.nxv4i32(
    <vscale x 4 x i32> %0,
    i32 %1)

  ret <vscale x 4 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfncvt.f.xu.w.mask.nxv4f16.nxv4i32(
  <vscale x 4 x half>,
  <vscale x 4 x i32>,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x half> @intrinsic_vfncvt_mask_f.xu.w_nxv4f16_nxv4i32(<vscale x 4 x half> %0, <vscale x 4 x i32> %1, <vscale x 4 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfncvt_mask_f.xu.w_nxv4f16_nxv4i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,tu,mu
; CHECK:       vfncvt.f.xu.w {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x half> @llvm.riscv.vfncvt.f.xu.w.mask.nxv4f16.nxv4i32(
    <vscale x 4 x half> %0,
    <vscale x 4 x i32> %1,
    <vscale x 4 x i1> %2,
    i32 %3)

  ret <vscale x 4 x half> %a
}

declare <vscale x 8 x half> @llvm.riscv.vfncvt.f.xu.w.nxv8f16.nxv8i32(
  <vscale x 8 x i32>,
  i32);

define <vscale x 8 x half> @intrinsic_vfncvt_f.xu.w_nxv8f16_nxv8i32(<vscale x 8 x i32> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfncvt_f.xu.w_nxv8f16_nxv8i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,ta,mu
; CHECK:       vfncvt.f.xu.w {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 8 x half> @llvm.riscv.vfncvt.f.xu.w.nxv8f16.nxv8i32(
    <vscale x 8 x i32> %0,
    i32 %1)

  ret <vscale x 8 x half> %a
}

declare <vscale x 8 x half> @llvm.riscv.vfncvt.f.xu.w.mask.nxv8f16.nxv8i32(
  <vscale x 8 x half>,
  <vscale x 8 x i32>,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x half> @intrinsic_vfncvt_mask_f.xu.w_nxv8f16_nxv8i32(<vscale x 8 x half> %0, <vscale x 8 x i32> %1, <vscale x 8 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfncvt_mask_f.xu.w_nxv8f16_nxv8i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,tu,mu
; CHECK:       vfncvt.f.xu.w {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 8 x half> @llvm.riscv.vfncvt.f.xu.w.mask.nxv8f16.nxv8i32(
    <vscale x 8 x half> %0,
    <vscale x 8 x i32> %1,
    <vscale x 8 x i1> %2,
    i32 %3)

  ret <vscale x 8 x half> %a
}

declare <vscale x 16 x half> @llvm.riscv.vfncvt.f.xu.w.nxv16f16.nxv16i32(
  <vscale x 16 x i32>,
  i32);

define <vscale x 16 x half> @intrinsic_vfncvt_f.xu.w_nxv16f16_nxv16i32(<vscale x 16 x i32> %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfncvt_f.xu.w_nxv16f16_nxv16i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,ta,mu
; CHECK:       vfncvt.f.xu.w {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 16 x half> @llvm.riscv.vfncvt.f.xu.w.nxv16f16.nxv16i32(
    <vscale x 16 x i32> %0,
    i32 %1)

  ret <vscale x 16 x half> %a
}

declare <vscale x 16 x half> @llvm.riscv.vfncvt.f.xu.w.mask.nxv16f16.nxv16i32(
  <vscale x 16 x half>,
  <vscale x 16 x i32>,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x half> @intrinsic_vfncvt_mask_f.xu.w_nxv16f16_nxv16i32(<vscale x 16 x half> %0, <vscale x 16 x i32> %1, <vscale x 16 x i1> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfncvt_mask_f.xu.w_nxv16f16_nxv16i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,tu,mu
; CHECK:       vfncvt.f.xu.w {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 16 x half> @llvm.riscv.vfncvt.f.xu.w.mask.nxv16f16.nxv16i32(
    <vscale x 16 x half> %0,
    <vscale x 16 x i32> %1,
    <vscale x 16 x i1> %2,
    i32 %3)

  ret <vscale x 16 x half> %a
}
