; RUN: llc -mtriple=riscv64 -mattr=+experimental-v -verify-machineinstrs \
; RUN:   --riscv-no-aliases < %s | FileCheck %s
declare <vscale x 1 x i1> @llvm.riscv.vmsif.nxv1i1(
  <vscale x 1 x i1>,
  i64);

define <vscale x 1 x i1> @intrinsic_vmsif_m_nxv1i1(<vscale x 1 x i1> %0, i64 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_m_nxv1i1
; CHECK:       vsetvli {{.*}}, a0, e8,mf8,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 1 x i1> @llvm.riscv.vmsif.nxv1i1(
    <vscale x 1 x i1> %0,
    i64 %1)
  ret <vscale x 1 x i1> %a
}

declare <vscale x 1 x i1> @llvm.riscv.vmsif.mask.nxv1i1(
  <vscale x 1 x i1>,
  <vscale x 1 x i1>,
  <vscale x 1 x i1>,
  i64);

define <vscale x 1 x i1> @intrinsic_vmsif_mask_m_nxv1i1_nxv1i1(<vscale x 1 x i1> %0, <vscale x 1 x i1> %1, <vscale x 1 x i1> %2, i64 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_mask_m_nxv1i1
; CHECK:       vsetvli {{.*}}, a0, e8,mf8,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 1 x i1> @llvm.riscv.vmsif.mask.nxv1i1(
    <vscale x 1 x i1> %0,
    <vscale x 1 x i1> %1,
    <vscale x 1 x i1> %2,
    i64 %3)
  ret <vscale x 1 x i1> %a
}

declare <vscale x 2 x i1> @llvm.riscv.vmsif.nxv2i1(
  <vscale x 2 x i1>,
  i64);

define <vscale x 2 x i1> @intrinsic_vmsif_m_nxv2i1(<vscale x 2 x i1> %0, i64 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_m_nxv2i1
; CHECK:       vsetvli {{.*}}, a0, e8,mf4,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x i1> @llvm.riscv.vmsif.nxv2i1(
    <vscale x 2 x i1> %0,
    i64 %1)
  ret <vscale x 2 x i1> %a
}

declare <vscale x 2 x i1> @llvm.riscv.vmsif.mask.nxv2i1(
  <vscale x 2 x i1>,
  <vscale x 2 x i1>,
  <vscale x 2 x i1>,
  i64);

define <vscale x 2 x i1> @intrinsic_vmsif_mask_m_nxv2i1_nxv2i1(<vscale x 2 x i1> %0, <vscale x 2 x i1> %1, <vscale x 2 x i1> %2, i64 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_mask_m_nxv2i1
; CHECK:       vsetvli {{.*}}, a0, e8,mf4,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x i1> @llvm.riscv.vmsif.mask.nxv2i1(
    <vscale x 2 x i1> %0,
    <vscale x 2 x i1> %1,
    <vscale x 2 x i1> %2,
    i64 %3)
  ret <vscale x 2 x i1> %a
}

declare <vscale x 4 x i1> @llvm.riscv.vmsif.nxv4i1(
  <vscale x 4 x i1>,
  i64);

define <vscale x 4 x i1> @intrinsic_vmsif_m_nxv4i1(<vscale x 4 x i1> %0, i64 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_m_nxv4i1
; CHECK:       vsetvli {{.*}}, a0, e8,mf2,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x i1> @llvm.riscv.vmsif.nxv4i1(
    <vscale x 4 x i1> %0,
    i64 %1)
  ret <vscale x 4 x i1> %a
}

declare <vscale x 4 x i1> @llvm.riscv.vmsif.mask.nxv4i1(
  <vscale x 4 x i1>,
  <vscale x 4 x i1>,
  <vscale x 4 x i1>,
  i64);

define <vscale x 4 x i1> @intrinsic_vmsif_mask_m_nxv4i1_nxv4i1(<vscale x 4 x i1> %0, <vscale x 4 x i1> %1, <vscale x 4 x i1> %2, i64 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_mask_m_nxv4i1
; CHECK:       vsetvli {{.*}}, a0, e8,mf2,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x i1> @llvm.riscv.vmsif.mask.nxv4i1(
    <vscale x 4 x i1> %0,
    <vscale x 4 x i1> %1,
    <vscale x 4 x i1> %2,
    i64 %3)
  ret <vscale x 4 x i1> %a
}

declare <vscale x 8 x i1> @llvm.riscv.vmsif.nxv8i1(
  <vscale x 8 x i1>,
  i64);

define <vscale x 8 x i1> @intrinsic_vmsif_m_nxv8i1(<vscale x 8 x i1> %0, i64 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_m_nxv8i1
; CHECK:       vsetvli {{.*}}, a0, e8,m1,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 8 x i1> @llvm.riscv.vmsif.nxv8i1(
    <vscale x 8 x i1> %0,
    i64 %1)
  ret <vscale x 8 x i1> %a
}

declare <vscale x 8 x i1> @llvm.riscv.vmsif.mask.nxv8i1(
  <vscale x 8 x i1>,
  <vscale x 8 x i1>,
  <vscale x 8 x i1>,
  i64);

define <vscale x 8 x i1> @intrinsic_vmsif_mask_m_nxv8i1_nxv8i1(<vscale x 8 x i1> %0, <vscale x 8 x i1> %1, <vscale x 8 x i1> %2, i64 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_mask_m_nxv8i1
; CHECK:       vsetvli {{.*}}, a0, e8,m1,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 8 x i1> @llvm.riscv.vmsif.mask.nxv8i1(
    <vscale x 8 x i1> %0,
    <vscale x 8 x i1> %1,
    <vscale x 8 x i1> %2,
    i64 %3)
  ret <vscale x 8 x i1> %a
}

declare <vscale x 16 x i1> @llvm.riscv.vmsif.nxv16i1(
  <vscale x 16 x i1>,
  i64);

define <vscale x 16 x i1> @intrinsic_vmsif_m_nxv16i1(<vscale x 16 x i1> %0, i64 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_m_nxv16i1
; CHECK:       vsetvli {{.*}}, a0, e8,m2,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 16 x i1> @llvm.riscv.vmsif.nxv16i1(
    <vscale x 16 x i1> %0,
    i64 %1)
  ret <vscale x 16 x i1> %a
}

declare <vscale x 16 x i1> @llvm.riscv.vmsif.mask.nxv16i1(
  <vscale x 16 x i1>,
  <vscale x 16 x i1>,
  <vscale x 16 x i1>,
  i64);

define <vscale x 16 x i1> @intrinsic_vmsif_mask_m_nxv16i1_nxv16i1(<vscale x 16 x i1> %0, <vscale x 16 x i1> %1, <vscale x 16 x i1> %2, i64 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_mask_m_nxv16i1
; CHECK:       vsetvli {{.*}}, a0, e8,m2,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 16 x i1> @llvm.riscv.vmsif.mask.nxv16i1(
    <vscale x 16 x i1> %0,
    <vscale x 16 x i1> %1,
    <vscale x 16 x i1> %2,
    i64 %3)
  ret <vscale x 16 x i1> %a
}

declare <vscale x 32 x i1> @llvm.riscv.vmsif.nxv32i1(
  <vscale x 32 x i1>,
  i64);

define <vscale x 32 x i1> @intrinsic_vmsif_m_nxv32i1(<vscale x 32 x i1> %0, i64 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_m_nxv32i1
; CHECK:       vsetvli {{.*}}, a0, e8,m4,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 32 x i1> @llvm.riscv.vmsif.nxv32i1(
    <vscale x 32 x i1> %0,
    i64 %1)
  ret <vscale x 32 x i1> %a
}

declare <vscale x 32 x i1> @llvm.riscv.vmsif.mask.nxv32i1(
  <vscale x 32 x i1>,
  <vscale x 32 x i1>,
  <vscale x 32 x i1>,
  i64);

define <vscale x 32 x i1> @intrinsic_vmsif_mask_m_nxv32i1_nxv32i1(<vscale x 32 x i1> %0, <vscale x 32 x i1> %1, <vscale x 32 x i1> %2, i64 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_mask_m_nxv32i1
; CHECK:       vsetvli {{.*}}, a0, e8,m4,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 32 x i1> @llvm.riscv.vmsif.mask.nxv32i1(
    <vscale x 32 x i1> %0,
    <vscale x 32 x i1> %1,
    <vscale x 32 x i1> %2,
    i64 %3)
  ret <vscale x 32 x i1> %a
}

declare <vscale x 64 x i1> @llvm.riscv.vmsif.nxv64i1(
  <vscale x 64 x i1>,
  i64);

define <vscale x 64 x i1> @intrinsic_vmsif_m_nxv64i1(<vscale x 64 x i1> %0, i64 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_m_nxv64i1
; CHECK:       vsetvli {{.*}}, a0, e8,m8,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 64 x i1> @llvm.riscv.vmsif.nxv64i1(
    <vscale x 64 x i1> %0,
    i64 %1)
  ret <vscale x 64 x i1> %a
}

declare <vscale x 64 x i1> @llvm.riscv.vmsif.mask.nxv64i1(
  <vscale x 64 x i1>,
  <vscale x 64 x i1>,
  <vscale x 64 x i1>,
  i64);

define <vscale x 64 x i1> @intrinsic_vmsif_mask_m_nxv64i1_nxv64i1(<vscale x 64 x i1> %0, <vscale x 64 x i1> %1, <vscale x 64 x i1> %2, i64 %3) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmsif_mask_m_nxv64i1
; CHECK:       vsetvli {{.*}}, a0, e8,m8,ta,mu
; CHECK:       vmsif.m {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 64 x i1> @llvm.riscv.vmsif.mask.nxv64i1(
    <vscale x 64 x i1> %0,
    <vscale x 64 x i1> %1,
    <vscale x 64 x i1> %2,
    i64 %3)
  ret <vscale x 64 x i1> %a
}
