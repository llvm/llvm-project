; RUN: llc -mtriple=riscv32 -mattr=+experimental-v,+f,+experimental-zfh -verify-machineinstrs \
; RUN:   --riscv-no-aliases < %s | FileCheck %s
declare <vscale x 1 x half> @llvm.riscv.vfrdiv.nxv1f16.f16(
  <vscale x 1 x half>,
  half,
  i32);

define <vscale x 1 x half> @intrinsic_vfrdiv_vf_nxv1f16_f16(<vscale x 1 x half> %0, half %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_vf_nxv1f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,ta,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 1 x half> @llvm.riscv.vfrdiv.nxv1f16.f16(
    <vscale x 1 x half> %0,
    half %1,
    i32 %2)

  ret <vscale x 1 x half> %a
}

declare <vscale x 1 x half> @llvm.riscv.vfrdiv.mask.nxv1f16.f16(
  <vscale x 1 x half>,
  <vscale x 1 x half>,
  half,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x half> @intrinsic_vfrdiv_mask_vf_nxv1f16_f16(<vscale x 1 x half> %0, <vscale x 1 x half> %1, half %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_mask_vf_nxv1f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,tu,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 1 x half> @llvm.riscv.vfrdiv.mask.nxv1f16.f16(
    <vscale x 1 x half> %0,
    <vscale x 1 x half> %1,
    half %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 1 x half> %a
}

declare <vscale x 2 x half> @llvm.riscv.vfrdiv.nxv2f16.f16(
  <vscale x 2 x half>,
  half,
  i32);

define <vscale x 2 x half> @intrinsic_vfrdiv_vf_nxv2f16_f16(<vscale x 2 x half> %0, half %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_vf_nxv2f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,ta,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 2 x half> @llvm.riscv.vfrdiv.nxv2f16.f16(
    <vscale x 2 x half> %0,
    half %1,
    i32 %2)

  ret <vscale x 2 x half> %a
}

declare <vscale x 2 x half> @llvm.riscv.vfrdiv.mask.nxv2f16.f16(
  <vscale x 2 x half>,
  <vscale x 2 x half>,
  half,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x half> @intrinsic_vfrdiv_mask_vf_nxv2f16_f16(<vscale x 2 x half> %0, <vscale x 2 x half> %1, half %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_mask_vf_nxv2f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,tu,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 2 x half> @llvm.riscv.vfrdiv.mask.nxv2f16.f16(
    <vscale x 2 x half> %0,
    <vscale x 2 x half> %1,
    half %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 2 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfrdiv.nxv4f16.f16(
  <vscale x 4 x half>,
  half,
  i32);

define <vscale x 4 x half> @intrinsic_vfrdiv_vf_nxv4f16_f16(<vscale x 4 x half> %0, half %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_vf_nxv4f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,ta,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 4 x half> @llvm.riscv.vfrdiv.nxv4f16.f16(
    <vscale x 4 x half> %0,
    half %1,
    i32 %2)

  ret <vscale x 4 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfrdiv.mask.nxv4f16.f16(
  <vscale x 4 x half>,
  <vscale x 4 x half>,
  half,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x half> @intrinsic_vfrdiv_mask_vf_nxv4f16_f16(<vscale x 4 x half> %0, <vscale x 4 x half> %1, half %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_mask_vf_nxv4f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,tu,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 4 x half> @llvm.riscv.vfrdiv.mask.nxv4f16.f16(
    <vscale x 4 x half> %0,
    <vscale x 4 x half> %1,
    half %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 4 x half> %a
}

declare <vscale x 8 x half> @llvm.riscv.vfrdiv.nxv8f16.f16(
  <vscale x 8 x half>,
  half,
  i32);

define <vscale x 8 x half> @intrinsic_vfrdiv_vf_nxv8f16_f16(<vscale x 8 x half> %0, half %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_vf_nxv8f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,ta,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 8 x half> @llvm.riscv.vfrdiv.nxv8f16.f16(
    <vscale x 8 x half> %0,
    half %1,
    i32 %2)

  ret <vscale x 8 x half> %a
}

declare <vscale x 8 x half> @llvm.riscv.vfrdiv.mask.nxv8f16.f16(
  <vscale x 8 x half>,
  <vscale x 8 x half>,
  half,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x half> @intrinsic_vfrdiv_mask_vf_nxv8f16_f16(<vscale x 8 x half> %0, <vscale x 8 x half> %1, half %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_mask_vf_nxv8f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,tu,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 8 x half> @llvm.riscv.vfrdiv.mask.nxv8f16.f16(
    <vscale x 8 x half> %0,
    <vscale x 8 x half> %1,
    half %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 8 x half> %a
}

declare <vscale x 16 x half> @llvm.riscv.vfrdiv.nxv16f16.f16(
  <vscale x 16 x half>,
  half,
  i32);

define <vscale x 16 x half> @intrinsic_vfrdiv_vf_nxv16f16_f16(<vscale x 16 x half> %0, half %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_vf_nxv16f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,ta,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 16 x half> @llvm.riscv.vfrdiv.nxv16f16.f16(
    <vscale x 16 x half> %0,
    half %1,
    i32 %2)

  ret <vscale x 16 x half> %a
}

declare <vscale x 16 x half> @llvm.riscv.vfrdiv.mask.nxv16f16.f16(
  <vscale x 16 x half>,
  <vscale x 16 x half>,
  half,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x half> @intrinsic_vfrdiv_mask_vf_nxv16f16_f16(<vscale x 16 x half> %0, <vscale x 16 x half> %1, half %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_mask_vf_nxv16f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,tu,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 16 x half> @llvm.riscv.vfrdiv.mask.nxv16f16.f16(
    <vscale x 16 x half> %0,
    <vscale x 16 x half> %1,
    half %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 16 x half> %a
}

declare <vscale x 32 x half> @llvm.riscv.vfrdiv.nxv32f16.f16(
  <vscale x 32 x half>,
  half,
  i32);

define <vscale x 32 x half> @intrinsic_vfrdiv_vf_nxv32f16_f16(<vscale x 32 x half> %0, half %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_vf_nxv32f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,ta,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 32 x half> @llvm.riscv.vfrdiv.nxv32f16.f16(
    <vscale x 32 x half> %0,
    half %1,
    i32 %2)

  ret <vscale x 32 x half> %a
}

declare <vscale x 32 x half> @llvm.riscv.vfrdiv.mask.nxv32f16.f16(
  <vscale x 32 x half>,
  <vscale x 32 x half>,
  half,
  <vscale x 32 x i1>,
  i32);

define <vscale x 32 x half> @intrinsic_vfrdiv_mask_vf_nxv32f16_f16(<vscale x 32 x half> %0, <vscale x 32 x half> %1, half %2, <vscale x 32 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_mask_vf_nxv32f16_f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,tu,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 32 x half> @llvm.riscv.vfrdiv.mask.nxv32f16.f16(
    <vscale x 32 x half> %0,
    <vscale x 32 x half> %1,
    half %2,
    <vscale x 32 x i1> %3,
    i32 %4)

  ret <vscale x 32 x half> %a
}

declare <vscale x 1 x float> @llvm.riscv.vfrdiv.nxv1f32.f32(
  <vscale x 1 x float>,
  float,
  i32);

define <vscale x 1 x float> @intrinsic_vfrdiv_vf_nxv1f32_f32(<vscale x 1 x float> %0, float %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_vf_nxv1f32_f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,mf2,ta,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 1 x float> @llvm.riscv.vfrdiv.nxv1f32.f32(
    <vscale x 1 x float> %0,
    float %1,
    i32 %2)

  ret <vscale x 1 x float> %a
}

declare <vscale x 1 x float> @llvm.riscv.vfrdiv.mask.nxv1f32.f32(
  <vscale x 1 x float>,
  <vscale x 1 x float>,
  float,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x float> @intrinsic_vfrdiv_mask_vf_nxv1f32_f32(<vscale x 1 x float> %0, <vscale x 1 x float> %1, float %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_mask_vf_nxv1f32_f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,mf2,tu,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 1 x float> @llvm.riscv.vfrdiv.mask.nxv1f32.f32(
    <vscale x 1 x float> %0,
    <vscale x 1 x float> %1,
    float %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 1 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfrdiv.nxv2f32.f32(
  <vscale x 2 x float>,
  float,
  i32);

define <vscale x 2 x float> @intrinsic_vfrdiv_vf_nxv2f32_f32(<vscale x 2 x float> %0, float %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_vf_nxv2f32_f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m1,ta,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 2 x float> @llvm.riscv.vfrdiv.nxv2f32.f32(
    <vscale x 2 x float> %0,
    float %1,
    i32 %2)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfrdiv.mask.nxv2f32.f32(
  <vscale x 2 x float>,
  <vscale x 2 x float>,
  float,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfrdiv_mask_vf_nxv2f32_f32(<vscale x 2 x float> %0, <vscale x 2 x float> %1, float %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_mask_vf_nxv2f32_f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m1,tu,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 2 x float> @llvm.riscv.vfrdiv.mask.nxv2f32.f32(
    <vscale x 2 x float> %0,
    <vscale x 2 x float> %1,
    float %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 4 x float> @llvm.riscv.vfrdiv.nxv4f32.f32(
  <vscale x 4 x float>,
  float,
  i32);

define <vscale x 4 x float> @intrinsic_vfrdiv_vf_nxv4f32_f32(<vscale x 4 x float> %0, float %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_vf_nxv4f32_f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m2,ta,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 4 x float> @llvm.riscv.vfrdiv.nxv4f32.f32(
    <vscale x 4 x float> %0,
    float %1,
    i32 %2)

  ret <vscale x 4 x float> %a
}

declare <vscale x 4 x float> @llvm.riscv.vfrdiv.mask.nxv4f32.f32(
  <vscale x 4 x float>,
  <vscale x 4 x float>,
  float,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x float> @intrinsic_vfrdiv_mask_vf_nxv4f32_f32(<vscale x 4 x float> %0, <vscale x 4 x float> %1, float %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_mask_vf_nxv4f32_f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m2,tu,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 4 x float> @llvm.riscv.vfrdiv.mask.nxv4f32.f32(
    <vscale x 4 x float> %0,
    <vscale x 4 x float> %1,
    float %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 4 x float> %a
}

declare <vscale x 8 x float> @llvm.riscv.vfrdiv.nxv8f32.f32(
  <vscale x 8 x float>,
  float,
  i32);

define <vscale x 8 x float> @intrinsic_vfrdiv_vf_nxv8f32_f32(<vscale x 8 x float> %0, float %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_vf_nxv8f32_f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m4,ta,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 8 x float> @llvm.riscv.vfrdiv.nxv8f32.f32(
    <vscale x 8 x float> %0,
    float %1,
    i32 %2)

  ret <vscale x 8 x float> %a
}

declare <vscale x 8 x float> @llvm.riscv.vfrdiv.mask.nxv8f32.f32(
  <vscale x 8 x float>,
  <vscale x 8 x float>,
  float,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x float> @intrinsic_vfrdiv_mask_vf_nxv8f32_f32(<vscale x 8 x float> %0, <vscale x 8 x float> %1, float %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_mask_vf_nxv8f32_f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m4,tu,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 8 x float> @llvm.riscv.vfrdiv.mask.nxv8f32.f32(
    <vscale x 8 x float> %0,
    <vscale x 8 x float> %1,
    float %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 8 x float> %a
}

declare <vscale x 16 x float> @llvm.riscv.vfrdiv.nxv16f32.f32(
  <vscale x 16 x float>,
  float,
  i32);

define <vscale x 16 x float> @intrinsic_vfrdiv_vf_nxv16f32_f32(<vscale x 16 x float> %0, float %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_vf_nxv16f32_f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m8,ta,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}
  %a = call <vscale x 16 x float> @llvm.riscv.vfrdiv.nxv16f32.f32(
    <vscale x 16 x float> %0,
    float %1,
    i32 %2)

  ret <vscale x 16 x float> %a
}

declare <vscale x 16 x float> @llvm.riscv.vfrdiv.mask.nxv16f32.f32(
  <vscale x 16 x float>,
  <vscale x 16 x float>,
  float,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x float> @intrinsic_vfrdiv_mask_vf_nxv16f32_f32(<vscale x 16 x float> %0, <vscale x 16 x float> %1, float %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfrdiv_mask_vf_nxv16f32_f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m8,tu,mu
; CHECK:       vfrdiv.vf {{v[0-9]+}}, {{v[0-9]+}}, {{ft[0-9]+}}, v0.t
  %a = call <vscale x 16 x float> @llvm.riscv.vfrdiv.mask.nxv16f32.f32(
    <vscale x 16 x float> %0,
    <vscale x 16 x float> %1,
    float %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 16 x float> %a
}
