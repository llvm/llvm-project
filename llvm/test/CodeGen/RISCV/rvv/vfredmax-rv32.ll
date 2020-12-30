; RUN: llc -mtriple=riscv32 -mattr=+experimental-v,+f,+experimental-zfh -verify-machineinstrs \
; RUN:   --riscv-no-aliases < %s | FileCheck %s
declare <vscale x 4 x half> @llvm.riscv.vfredmax.nxv4f16.nxv1f16(
  <vscale x 4 x half>,
  <vscale x 1 x half>,
  <vscale x 4 x half>,
  i32);

define <vscale x 4 x half> @intrinsic_vfredmax_vs_nxv4f16_nxv1f16_nxv4f16(<vscale x 4 x half> %0, <vscale x 1 x half> %1, <vscale x 4 x half> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_vs_nxv4f16_nxv1f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x half> @llvm.riscv.vfredmax.nxv4f16.nxv1f16(
    <vscale x 4 x half> %0,
    <vscale x 1 x half> %1,
    <vscale x 4 x half> %2,
    i32 %3)

  ret <vscale x 4 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfredmax.mask.nxv4f16.nxv1f16(
  <vscale x 4 x half>,
  <vscale x 1 x half>,
  <vscale x 4 x half>,
  <vscale x 1 x i1>,
  i32);

define <vscale x 4 x half> @intrinsic_vfredmax_mask_vs_nxv4f16_nxv1f16_nxv4f16(<vscale x 4 x half> %0, <vscale x 1 x half> %1, <vscale x 4 x half> %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_mask_vs_nxv4f16_nxv1f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x half> @llvm.riscv.vfredmax.mask.nxv4f16.nxv1f16(
    <vscale x 4 x half> %0,
    <vscale x 1 x half> %1,
    <vscale x 4 x half> %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 4 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfredmax.nxv4f16.nxv2f16(
  <vscale x 4 x half>,
  <vscale x 2 x half>,
  <vscale x 4 x half>,
  i32);

define <vscale x 4 x half> @intrinsic_vfredmax_vs_nxv4f16_nxv2f16_nxv4f16(<vscale x 4 x half> %0, <vscale x 2 x half> %1, <vscale x 4 x half> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_vs_nxv4f16_nxv2f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x half> @llvm.riscv.vfredmax.nxv4f16.nxv2f16(
    <vscale x 4 x half> %0,
    <vscale x 2 x half> %1,
    <vscale x 4 x half> %2,
    i32 %3)

  ret <vscale x 4 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfredmax.mask.nxv4f16.nxv2f16(
  <vscale x 4 x half>,
  <vscale x 2 x half>,
  <vscale x 4 x half>,
  <vscale x 2 x i1>,
  i32);

define <vscale x 4 x half> @intrinsic_vfredmax_mask_vs_nxv4f16_nxv2f16_nxv4f16(<vscale x 4 x half> %0, <vscale x 2 x half> %1, <vscale x 4 x half> %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_mask_vs_nxv4f16_nxv2f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x half> @llvm.riscv.vfredmax.mask.nxv4f16.nxv2f16(
    <vscale x 4 x half> %0,
    <vscale x 2 x half> %1,
    <vscale x 4 x half> %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 4 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfredmax.nxv4f16.nxv4f16(
  <vscale x 4 x half>,
  <vscale x 4 x half>,
  <vscale x 4 x half>,
  i32);

define <vscale x 4 x half> @intrinsic_vfredmax_vs_nxv4f16_nxv4f16_nxv4f16(<vscale x 4 x half> %0, <vscale x 4 x half> %1, <vscale x 4 x half> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_vs_nxv4f16_nxv4f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x half> @llvm.riscv.vfredmax.nxv4f16.nxv4f16(
    <vscale x 4 x half> %0,
    <vscale x 4 x half> %1,
    <vscale x 4 x half> %2,
    i32 %3)

  ret <vscale x 4 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfredmax.mask.nxv4f16.nxv4f16(
  <vscale x 4 x half>,
  <vscale x 4 x half>,
  <vscale x 4 x half>,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x half> @intrinsic_vfredmax_mask_vs_nxv4f16_nxv4f16_nxv4f16(<vscale x 4 x half> %0, <vscale x 4 x half> %1, <vscale x 4 x half> %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_mask_vs_nxv4f16_nxv4f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x half> @llvm.riscv.vfredmax.mask.nxv4f16.nxv4f16(
    <vscale x 4 x half> %0,
    <vscale x 4 x half> %1,
    <vscale x 4 x half> %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 4 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfredmax.nxv4f16.nxv8f16(
  <vscale x 4 x half>,
  <vscale x 8 x half>,
  <vscale x 4 x half>,
  i32);

define <vscale x 4 x half> @intrinsic_vfredmax_vs_nxv4f16_nxv8f16_nxv4f16(<vscale x 4 x half> %0, <vscale x 8 x half> %1, <vscale x 4 x half> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_vs_nxv4f16_nxv8f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x half> @llvm.riscv.vfredmax.nxv4f16.nxv8f16(
    <vscale x 4 x half> %0,
    <vscale x 8 x half> %1,
    <vscale x 4 x half> %2,
    i32 %3)

  ret <vscale x 4 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfredmax.mask.nxv4f16.nxv8f16(
  <vscale x 4 x half>,
  <vscale x 8 x half>,
  <vscale x 4 x half>,
  <vscale x 8 x i1>,
  i32);

define <vscale x 4 x half> @intrinsic_vfredmax_mask_vs_nxv4f16_nxv8f16_nxv4f16(<vscale x 4 x half> %0, <vscale x 8 x half> %1, <vscale x 4 x half> %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_mask_vs_nxv4f16_nxv8f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x half> @llvm.riscv.vfredmax.mask.nxv4f16.nxv8f16(
    <vscale x 4 x half> %0,
    <vscale x 8 x half> %1,
    <vscale x 4 x half> %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 4 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfredmax.nxv4f16.nxv16f16(
  <vscale x 4 x half>,
  <vscale x 16 x half>,
  <vscale x 4 x half>,
  i32);

define <vscale x 4 x half> @intrinsic_vfredmax_vs_nxv4f16_nxv16f16_nxv4f16(<vscale x 4 x half> %0, <vscale x 16 x half> %1, <vscale x 4 x half> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_vs_nxv4f16_nxv16f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x half> @llvm.riscv.vfredmax.nxv4f16.nxv16f16(
    <vscale x 4 x half> %0,
    <vscale x 16 x half> %1,
    <vscale x 4 x half> %2,
    i32 %3)

  ret <vscale x 4 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfredmax.mask.nxv4f16.nxv16f16(
  <vscale x 4 x half>,
  <vscale x 16 x half>,
  <vscale x 4 x half>,
  <vscale x 16 x i1>,
  i32);

define <vscale x 4 x half> @intrinsic_vfredmax_mask_vs_nxv4f16_nxv16f16_nxv4f16(<vscale x 4 x half> %0, <vscale x 16 x half> %1, <vscale x 4 x half> %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_mask_vs_nxv4f16_nxv16f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x half> @llvm.riscv.vfredmax.mask.nxv4f16.nxv16f16(
    <vscale x 4 x half> %0,
    <vscale x 16 x half> %1,
    <vscale x 4 x half> %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 4 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfredmax.nxv4f16.nxv32f16(
  <vscale x 4 x half>,
  <vscale x 32 x half>,
  <vscale x 4 x half>,
  i32);

define <vscale x 4 x half> @intrinsic_vfredmax_vs_nxv4f16_nxv32f16_nxv4f16(<vscale x 4 x half> %0, <vscale x 32 x half> %1, <vscale x 4 x half> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_vs_nxv4f16_nxv32f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x half> @llvm.riscv.vfredmax.nxv4f16.nxv32f16(
    <vscale x 4 x half> %0,
    <vscale x 32 x half> %1,
    <vscale x 4 x half> %2,
    i32 %3)

  ret <vscale x 4 x half> %a
}

declare <vscale x 4 x half> @llvm.riscv.vfredmax.mask.nxv4f16.nxv32f16(
  <vscale x 4 x half>,
  <vscale x 32 x half>,
  <vscale x 4 x half>,
  <vscale x 32 x i1>,
  i32);

define <vscale x 4 x half> @intrinsic_vfredmax_mask_vs_nxv4f16_nxv32f16_nxv4f16(<vscale x 4 x half> %0, <vscale x 32 x half> %1, <vscale x 4 x half> %2, <vscale x 32 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_mask_vs_nxv4f16_nxv32f16_nxv4f16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x half> @llvm.riscv.vfredmax.mask.nxv4f16.nxv32f16(
    <vscale x 4 x half> %0,
    <vscale x 32 x half> %1,
    <vscale x 4 x half> %2,
    <vscale x 32 x i1> %3,
    i32 %4)

  ret <vscale x 4 x half> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfredmax.nxv2f32.nxv1f32(
  <vscale x 2 x float>,
  <vscale x 1 x float>,
  <vscale x 2 x float>,
  i32);

define <vscale x 2 x float> @intrinsic_vfredmax_vs_nxv2f32_nxv1f32_nxv2f32(<vscale x 2 x float> %0, <vscale x 1 x float> %1, <vscale x 2 x float> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_vs_nxv2f32_nxv1f32_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,mf2,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x float> @llvm.riscv.vfredmax.nxv2f32.nxv1f32(
    <vscale x 2 x float> %0,
    <vscale x 1 x float> %1,
    <vscale x 2 x float> %2,
    i32 %3)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfredmax.mask.nxv2f32.nxv1f32(
  <vscale x 2 x float>,
  <vscale x 1 x float>,
  <vscale x 2 x float>,
  <vscale x 1 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfredmax_mask_vs_nxv2f32_nxv1f32_nxv2f32(<vscale x 2 x float> %0, <vscale x 1 x float> %1, <vscale x 2 x float> %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_mask_vs_nxv2f32_nxv1f32_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,mf2,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x float> @llvm.riscv.vfredmax.mask.nxv2f32.nxv1f32(
    <vscale x 2 x float> %0,
    <vscale x 1 x float> %1,
    <vscale x 2 x float> %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfredmax.nxv2f32.nxv2f32(
  <vscale x 2 x float>,
  <vscale x 2 x float>,
  <vscale x 2 x float>,
  i32);

define <vscale x 2 x float> @intrinsic_vfredmax_vs_nxv2f32_nxv2f32_nxv2f32(<vscale x 2 x float> %0, <vscale x 2 x float> %1, <vscale x 2 x float> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_vs_nxv2f32_nxv2f32_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m1,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x float> @llvm.riscv.vfredmax.nxv2f32.nxv2f32(
    <vscale x 2 x float> %0,
    <vscale x 2 x float> %1,
    <vscale x 2 x float> %2,
    i32 %3)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfredmax.mask.nxv2f32.nxv2f32(
  <vscale x 2 x float>,
  <vscale x 2 x float>,
  <vscale x 2 x float>,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfredmax_mask_vs_nxv2f32_nxv2f32_nxv2f32(<vscale x 2 x float> %0, <vscale x 2 x float> %1, <vscale x 2 x float> %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_mask_vs_nxv2f32_nxv2f32_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m1,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x float> @llvm.riscv.vfredmax.mask.nxv2f32.nxv2f32(
    <vscale x 2 x float> %0,
    <vscale x 2 x float> %1,
    <vscale x 2 x float> %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfredmax.nxv2f32.nxv4f32(
  <vscale x 2 x float>,
  <vscale x 4 x float>,
  <vscale x 2 x float>,
  i32);

define <vscale x 2 x float> @intrinsic_vfredmax_vs_nxv2f32_nxv4f32_nxv2f32(<vscale x 2 x float> %0, <vscale x 4 x float> %1, <vscale x 2 x float> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_vs_nxv2f32_nxv4f32_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m2,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x float> @llvm.riscv.vfredmax.nxv2f32.nxv4f32(
    <vscale x 2 x float> %0,
    <vscale x 4 x float> %1,
    <vscale x 2 x float> %2,
    i32 %3)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfredmax.mask.nxv2f32.nxv4f32(
  <vscale x 2 x float>,
  <vscale x 4 x float>,
  <vscale x 2 x float>,
  <vscale x 4 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfredmax_mask_vs_nxv2f32_nxv4f32_nxv2f32(<vscale x 2 x float> %0, <vscale x 4 x float> %1, <vscale x 2 x float> %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_mask_vs_nxv2f32_nxv4f32_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m2,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x float> @llvm.riscv.vfredmax.mask.nxv2f32.nxv4f32(
    <vscale x 2 x float> %0,
    <vscale x 4 x float> %1,
    <vscale x 2 x float> %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfredmax.nxv2f32.nxv8f32(
  <vscale x 2 x float>,
  <vscale x 8 x float>,
  <vscale x 2 x float>,
  i32);

define <vscale x 2 x float> @intrinsic_vfredmax_vs_nxv2f32_nxv8f32_nxv2f32(<vscale x 2 x float> %0, <vscale x 8 x float> %1, <vscale x 2 x float> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_vs_nxv2f32_nxv8f32_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m4,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x float> @llvm.riscv.vfredmax.nxv2f32.nxv8f32(
    <vscale x 2 x float> %0,
    <vscale x 8 x float> %1,
    <vscale x 2 x float> %2,
    i32 %3)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfredmax.mask.nxv2f32.nxv8f32(
  <vscale x 2 x float>,
  <vscale x 8 x float>,
  <vscale x 2 x float>,
  <vscale x 8 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfredmax_mask_vs_nxv2f32_nxv8f32_nxv2f32(<vscale x 2 x float> %0, <vscale x 8 x float> %1, <vscale x 2 x float> %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_mask_vs_nxv2f32_nxv8f32_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m4,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x float> @llvm.riscv.vfredmax.mask.nxv2f32.nxv8f32(
    <vscale x 2 x float> %0,
    <vscale x 8 x float> %1,
    <vscale x 2 x float> %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfredmax.nxv2f32.nxv16f32(
  <vscale x 2 x float>,
  <vscale x 16 x float>,
  <vscale x 2 x float>,
  i32);

define <vscale x 2 x float> @intrinsic_vfredmax_vs_nxv2f32_nxv16f32_nxv2f32(<vscale x 2 x float> %0, <vscale x 16 x float> %1, <vscale x 2 x float> %2, i32 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_vs_nxv2f32_nxv16f32_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m8,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x float> @llvm.riscv.vfredmax.nxv2f32.nxv16f32(
    <vscale x 2 x float> %0,
    <vscale x 16 x float> %1,
    <vscale x 2 x float> %2,
    i32 %3)

  ret <vscale x 2 x float> %a
}

declare <vscale x 2 x float> @llvm.riscv.vfredmax.mask.nxv2f32.nxv16f32(
  <vscale x 2 x float>,
  <vscale x 16 x float>,
  <vscale x 2 x float>,
  <vscale x 16 x i1>,
  i32);

define <vscale x 2 x float> @intrinsic_vfredmax_mask_vs_nxv2f32_nxv16f32_nxv2f32(<vscale x 2 x float> %0, <vscale x 16 x float> %1, <vscale x 2 x float> %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vfredmax_mask_vs_nxv2f32_nxv16f32_nxv2f32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m8,ta,mu
; CHECK:       vfredmax.vs {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x float> @llvm.riscv.vfredmax.mask.nxv2f32.nxv16f32(
    <vscale x 2 x float> %0,
    <vscale x 16 x float> %1,
    <vscale x 2 x float> %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 2 x float> %a
}
