; RUN: llc -mtriple=riscv32 -mattr=+experimental-v,+f,+experimental-zfh -verify-machineinstrs \
; RUN:   --riscv-no-aliases < %s | FileCheck %s
declare <vscale x 1 x float> @llvm.riscv.vfwsub.nxv1f16(
  <vscale x 1 x half>,
  <vscale x 1 x half>,
  i32);

define <vscale x 1 x float> @intrinsic_vfwsub_vv_nxv1f16_nxv1f16(<vscale x 1 x half> %0, <vscale x 1 x half> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_vv_nxv1f16_nxv1f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,ta,mu
; CHECK:       vfwsub.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 1 x float> @llvm.riscv.vfwsub.nxv1f16(
    <vscale x 1 x half> %0,
    <vscale x 1 x half> %1,
    i32 %2)

  ret <vscale x 1 x float> %a
}

declare <vscale x 1 x float> @llvm.riscv.vfwsub.mask.nxv1f16(
  <vscale x 1 x float>,
  <vscale x 1 x half>,
  <vscale x 1 x half>,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x float> @intrinsic_vfwsub_mask_vv_nxv1f16_nxv1f16(<vscale x 1 x float> %0, <vscale x 1 x half> %1, <vscale x 1 x half> %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_mask_vv_nxv1f16_nxv1f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,ta,mu
; CHECK:       vfwsub.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 1 x float> @llvm.riscv.vfwsub.mask.nxv1f16(
    <vscale x 1 x float> %0,
    <vscale x 1 x half> %1,
    <vscale x 1 x half> %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 1 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwsub.nxv2f16(
  <vscale x 2 x half>,
  <vscale x 2 x half>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwsub_vv_nxv2f16_nxv2f16(<vscale x 2 x half> %0, <vscale x 2 x half> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_vv_nxv2f16_nxv2f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,ta,mu
; CHECK:       vfwsub.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x float> @llvm.riscv.vfwsub.nxv2f16(
    <vscale x 2 x half> %0,
    <vscale x 2 x half> %1,
    i32 %2)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwsub.mask.nxv2f16(
  <vscale x 2 x float>,
  <vscale x 2 x half>,
  <vscale x 2 x half>,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwsub_mask_vv_nxv2f16_nxv2f16(<vscale x 2 x float> %0, <vscale x 2 x half> %1, <vscale x 2 x half> %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_mask_vv_nxv2f16_nxv2f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,ta,mu
; CHECK:       vfwsub.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x float> @llvm.riscv.vfwsub.mask.nxv2f16(
    <vscale x 2 x float> %0,
    <vscale x 2 x half> %1,
    <vscale x 2 x half> %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 4 x float> @llvm.riscv.vfwsub.nxv4f16(
  <vscale x 4 x half>,
  <vscale x 4 x half>,
  i32);

define <vscale x 4 x float> @intrinsic_vfwsub_vv_nxv4f16_nxv4f16(<vscale x 4 x half> %0, <vscale x 4 x half> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_vv_nxv4f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,ta,mu
; CHECK:       vfwsub.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x float> @llvm.riscv.vfwsub.nxv4f16(
    <vscale x 4 x half> %0,
    <vscale x 4 x half> %1,
    i32 %2)

  ret <vscale x 4 x float> %a
}

declare <vscale x 4 x float> @llvm.riscv.vfwsub.mask.nxv4f16(
  <vscale x 4 x float>,
  <vscale x 4 x half>,
  <vscale x 4 x half>,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x float> @intrinsic_vfwsub_mask_vv_nxv4f16_nxv4f16(<vscale x 4 x float> %0, <vscale x 4 x half> %1, <vscale x 4 x half> %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_mask_vv_nxv4f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,ta,mu
; CHECK:       vfwsub.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x float> @llvm.riscv.vfwsub.mask.nxv4f16(
    <vscale x 4 x float> %0,
    <vscale x 4 x half> %1,
    <vscale x 4 x half> %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 4 x float> %a
}

declare <vscale x 8 x float> @llvm.riscv.vfwsub.nxv8f16(
  <vscale x 8 x half>,
  <vscale x 8 x half>,
  i32);

define <vscale x 8 x float> @intrinsic_vfwsub_vv_nxv8f16_nxv8f16(<vscale x 8 x half> %0, <vscale x 8 x half> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_vv_nxv8f16_nxv8f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,ta,mu
; CHECK:       vfwsub.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 8 x float> @llvm.riscv.vfwsub.nxv8f16(
    <vscale x 8 x half> %0,
    <vscale x 8 x half> %1,
    i32 %2)

  ret <vscale x 8 x float> %a
}

declare <vscale x 8 x float> @llvm.riscv.vfwsub.mask.nxv8f16(
  <vscale x 8 x float>,
  <vscale x 8 x half>,
  <vscale x 8 x half>,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x float> @intrinsic_vfwsub_mask_vv_nxv8f16_nxv8f16(<vscale x 8 x float> %0, <vscale x 8 x half> %1, <vscale x 8 x half> %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_mask_vv_nxv8f16_nxv8f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,ta,mu
; CHECK:       vfwsub.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 8 x float> @llvm.riscv.vfwsub.mask.nxv8f16(
    <vscale x 8 x float> %0,
    <vscale x 8 x half> %1,
    <vscale x 8 x half> %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 8 x float> %a
}

declare <vscale x 16 x float> @llvm.riscv.vfwsub.nxv16f16(
  <vscale x 16 x half>,
  <vscale x 16 x half>,
  i32);

define <vscale x 16 x float> @intrinsic_vfwsub_vv_nxv16f16_nxv16f16(<vscale x 16 x half> %0, <vscale x 16 x half> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_vv_nxv16f16_nxv16f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,ta,mu
; CHECK:       vfwsub.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 16 x float> @llvm.riscv.vfwsub.nxv16f16(
    <vscale x 16 x half> %0,
    <vscale x 16 x half> %1,
    i32 %2)

  ret <vscale x 16 x float> %a
}

declare <vscale x 16 x float> @llvm.riscv.vfwsub.mask.nxv16f16(
  <vscale x 16 x float>,
  <vscale x 16 x half>,
  <vscale x 16 x half>,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x float> @intrinsic_vfwsub_mask_vv_nxv16f16_nxv16f16(<vscale x 16 x float> %0, <vscale x 16 x half> %1, <vscale x 16 x half> %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_mask_vv_nxv16f16_nxv16f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,ta,mu
; CHECK:       vfwsub.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 16 x float> @llvm.riscv.vfwsub.mask.nxv16f16(
    <vscale x 16 x float> %0,
    <vscale x 16 x half> %1,
    <vscale x 16 x half> %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 16 x float> %a
}

declare <vscale x 1 x float> @llvm.riscv.vfwsub.nxv1f16.f16(
  <vscale x 1 x half>,
  half,
  i32);

define <vscale x 1 x float> @intrinsic_vfwsub_vf_nxv1f16_f16(<vscale x 1 x half> %0, half %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_vf_nxv1f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,ta,mu
; CHECK:       vfwsub.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 1 x float> @llvm.riscv.vfwsub.nxv1f16.f16(
    <vscale x 1 x half> %0,
    half %1,
    i32 %2)

  ret <vscale x 1 x float> %a
}

declare <vscale x 1 x float> @llvm.riscv.vfwsub.mask.nxv1f16.f16(
  <vscale x 1 x float>,
  <vscale x 1 x half>,
  half,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x float> @intrinsic_vfwsub_mask_vf_nxv1f16_f16(<vscale x 1 x float> %0, <vscale x 1 x half> %1, half %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_mask_vf_nxv1f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,ta,mu
; CHECK:       vfwsub.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 1 x float> @llvm.riscv.vfwsub.mask.nxv1f16.f16(
    <vscale x 1 x float> %0,
    <vscale x 1 x half> %1,
    half %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 1 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwsub.nxv2f16.f16(
  <vscale x 2 x half>,
  half,
  i32);

define <vscale x 2 x float> @intrinsic_vfwsub_vf_nxv2f16_f16(<vscale x 2 x half> %0, half %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_vf_nxv2f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,ta,mu
; CHECK:       vfwsub.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 2 x float> @llvm.riscv.vfwsub.nxv2f16.f16(
    <vscale x 2 x half> %0,
    half %1,
    i32 %2)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfwsub.mask.nxv2f16.f16(
  <vscale x 2 x float>,
  <vscale x 2 x half>,
  half,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfwsub_mask_vf_nxv2f16_f16(<vscale x 2 x float> %0, <vscale x 2 x half> %1, half %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_mask_vf_nxv2f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,ta,mu
; CHECK:       vfwsub.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 2 x float> @llvm.riscv.vfwsub.mask.nxv2f16.f16(
    <vscale x 2 x float> %0,
    <vscale x 2 x half> %1,
    half %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 4 x float> @llvm.riscv.vfwsub.nxv4f16.f16(
  <vscale x 4 x half>,
  half,
  i32);

define <vscale x 4 x float> @intrinsic_vfwsub_vf_nxv4f16_f16(<vscale x 4 x half> %0, half %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_vf_nxv4f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,ta,mu
; CHECK:       vfwsub.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 4 x float> @llvm.riscv.vfwsub.nxv4f16.f16(
    <vscale x 4 x half> %0,
    half %1,
    i32 %2)

  ret <vscale x 4 x float> %a
}

declare <vscale x 4 x float> @llvm.riscv.vfwsub.mask.nxv4f16.f16(
  <vscale x 4 x float>,
  <vscale x 4 x half>,
  half,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x float> @intrinsic_vfwsub_mask_vf_nxv4f16_f16(<vscale x 4 x float> %0, <vscale x 4 x half> %1, half %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_mask_vf_nxv4f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,ta,mu
; CHECK:       vfwsub.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 4 x float> @llvm.riscv.vfwsub.mask.nxv4f16.f16(
    <vscale x 4 x float> %0,
    <vscale x 4 x half> %1,
    half %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 4 x float> %a
}

declare <vscale x 8 x float> @llvm.riscv.vfwsub.nxv8f16.f16(
  <vscale x 8 x half>,
  half,
  i32);

define <vscale x 8 x float> @intrinsic_vfwsub_vf_nxv8f16_f16(<vscale x 8 x half> %0, half %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_vf_nxv8f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,ta,mu
; CHECK:       vfwsub.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 8 x float> @llvm.riscv.vfwsub.nxv8f16.f16(
    <vscale x 8 x half> %0,
    half %1,
    i32 %2)

  ret <vscale x 8 x float> %a
}

declare <vscale x 8 x float> @llvm.riscv.vfwsub.mask.nxv8f16.f16(
  <vscale x 8 x float>,
  <vscale x 8 x half>,
  half,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x float> @intrinsic_vfwsub_mask_vf_nxv8f16_f16(<vscale x 8 x float> %0, <vscale x 8 x half> %1, half %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_mask_vf_nxv8f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,ta,mu
; CHECK:       vfwsub.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 8 x float> @llvm.riscv.vfwsub.mask.nxv8f16.f16(
    <vscale x 8 x float> %0,
    <vscale x 8 x half> %1,
    half %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 8 x float> %a
}

declare <vscale x 16 x float> @llvm.riscv.vfwsub.nxv16f16.f16(
  <vscale x 16 x half>,
  half,
  i32);

define <vscale x 16 x float> @intrinsic_vfwsub_vf_nxv16f16_f16(<vscale x 16 x half> %0, half %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_vf_nxv16f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,ta,mu
; CHECK:       vfwsub.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 16 x float> @llvm.riscv.vfwsub.nxv16f16.f16(
    <vscale x 16 x half> %0,
    half %1,
    i32 %2)

  ret <vscale x 16 x float> %a
}

declare <vscale x 16 x float> @llvm.riscv.vfwsub.mask.nxv16f16.f16(
  <vscale x 16 x float>,
  <vscale x 16 x half>,
  half,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x float> @intrinsic_vfwsub_mask_vf_nxv16f16_f16(<vscale x 16 x float> %0, <vscale x 16 x half> %1, half %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfwsub_mask_vf_nxv16f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,ta,mu
; CHECK:       vfwsub.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 16 x float> @llvm.riscv.vfwsub.mask.nxv16f16.f16(
    <vscale x 16 x float> %0,
    <vscale x 16 x half> %1,
    half %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 16 x float> %a
}
