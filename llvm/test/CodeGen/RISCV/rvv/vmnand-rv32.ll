; RUN: llc -mtriple=riscv32 -mattr=+experimental-v,+f -verify-machineinstrs \
; RUN:   --riscv-no-aliases < %s | FileCheck %s
declare <vscale x 1 x i1> @llvm.riscv.vmnand.nxv1i1(
  <vscale x 1 x i1>,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x i1> @intrinsic_vmnand_mm_nxv1i1(<vscale x 1 x i1> %0, <vscale x 1 x i1> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmnand_mm_nxv1i1
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf8,ta,mu
; CHECK:       vmnand.mm {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 1 x i1> @llvm.riscv.vmnand.nxv1i1(
    <vscale x 1 x i1> %0,
    <vscale x 1 x i1> %1,
    i32 %2)

  ret <vscale x 1 x i1> %a
}

declare <vscale x 2 x i1> @llvm.riscv.vmnand.nxv2i1(
  <vscale x 2 x i1>,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x i1> @intrinsic_vmnand_mm_nxv2i1(<vscale x 2 x i1> %0, <vscale x 2 x i1> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmnand_mm_nxv2i1
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf4,ta,mu
; CHECK:       vmnand.mm {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x i1> @llvm.riscv.vmnand.nxv2i1(
    <vscale x 2 x i1> %0,
    <vscale x 2 x i1> %1,
    i32 %2)

  ret <vscale x 2 x i1> %a
}

declare <vscale x 4 x i1> @llvm.riscv.vmnand.nxv4i1(
  <vscale x 4 x i1>,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x i1> @intrinsic_vmnand_mm_nxv4i1(<vscale x 4 x i1> %0, <vscale x 4 x i1> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmnand_mm_nxv4i1
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf2,ta,mu
; CHECK:       vmnand.mm {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x i1> @llvm.riscv.vmnand.nxv4i1(
    <vscale x 4 x i1> %0,
    <vscale x 4 x i1> %1,
    i32 %2)

  ret <vscale x 4 x i1> %a
}

declare <vscale x 8 x i1> @llvm.riscv.vmnand.nxv8i1(
  <vscale x 8 x i1>,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x i1> @intrinsic_vmnand_mm_nxv8i1(<vscale x 8 x i1> %0, <vscale x 8 x i1> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmnand_mm_nxv8i1
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m1,ta,mu
; CHECK:       vmnand.mm {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 8 x i1> @llvm.riscv.vmnand.nxv8i1(
    <vscale x 8 x i1> %0,
    <vscale x 8 x i1> %1,
    i32 %2)

  ret <vscale x 8 x i1> %a
}

declare <vscale x 16 x i1> @llvm.riscv.vmnand.nxv16i1(
  <vscale x 16 x i1>,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x i1> @intrinsic_vmnand_mm_nxv16i1(<vscale x 16 x i1> %0, <vscale x 16 x i1> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmnand_mm_nxv16i1
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m2,ta,mu
; CHECK:       vmnand.mm {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 16 x i1> @llvm.riscv.vmnand.nxv16i1(
    <vscale x 16 x i1> %0,
    <vscale x 16 x i1> %1,
    i32 %2)

  ret <vscale x 16 x i1> %a
}

declare <vscale x 32 x i1> @llvm.riscv.vmnand.nxv32i1(
  <vscale x 32 x i1>,
  <vscale x 32 x i1>,
  i32);

define <vscale x 32 x i1> @intrinsic_vmnand_mm_nxv32i1(<vscale x 32 x i1> %0, <vscale x 32 x i1> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmnand_mm_nxv32i1
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m4,ta,mu
; CHECK:       vmnand.mm {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 32 x i1> @llvm.riscv.vmnand.nxv32i1(
    <vscale x 32 x i1> %0,
    <vscale x 32 x i1> %1,
    i32 %2)

  ret <vscale x 32 x i1> %a
}

declare <vscale x 64 x i1> @llvm.riscv.vmnand.nxv64i1(
  <vscale x 64 x i1>,
  <vscale x 64 x i1>,
  i32);

define <vscale x 64 x i1> @intrinsic_vmnand_mm_nxv64i1(<vscale x 64 x i1> %0, <vscale x 64 x i1> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmnand_mm_nxv64i1
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m8,ta,mu
; CHECK:       vmnand.mm {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 64 x i1> @llvm.riscv.vmnand.nxv64i1(
    <vscale x 64 x i1> %0,
    <vscale x 64 x i1> %1,
    i32 %2)

  ret <vscale x 64 x i1> %a
}
