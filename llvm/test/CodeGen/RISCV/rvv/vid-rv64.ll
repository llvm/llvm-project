; RUN: llc -mtriple=riscv64 -mattr=+experimental-v -verify-machineinstrs \
; RUN:   --riscv-no-aliases < %s | FileCheck %s
declare <vscale x 1 x i8> @llvm.riscv.vid.nxv1i8(
  i64);

define <vscale x 1 x i8> @intrinsic_vid_v_nxv1i8(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv1i8
; CHECK:       vsetvli {{.*}}, a0, e8,mf8,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 1 x i8> @llvm.riscv.vid.nxv1i8(
    i64 %0)

  ret <vscale x 1 x i8> %a
}

declare <vscale x 1 x i8> @llvm.riscv.vid.mask.nxv1i8(
  <vscale x 1 x i8>,
  <vscale x 1 x i1>,
  i64);

define <vscale x 1 x i8> @intrinsic_vid_mask_v_nxv1i8(<vscale x 1 x i8> %0, <vscale x 1 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv1i8
; CHECK:       vsetvli {{.*}}, a0, e8,mf8,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 1 x i8> @llvm.riscv.vid.mask.nxv1i8(
    <vscale x 1 x i8> %0,
    <vscale x 1 x i1> %1,
    i64 %2)

  ret <vscale x 1 x i8> %a
}

declare <vscale x 2 x i8> @llvm.riscv.vid.nxv2i8(
  i64);

define <vscale x 2 x i8> @intrinsic_vid_v_nxv2i8(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv2i8
; CHECK:       vsetvli {{.*}}, a0, e8,mf4,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 2 x i8> @llvm.riscv.vid.nxv2i8(
    i64 %0)

  ret <vscale x 2 x i8> %a
}

declare <vscale x 2 x i8> @llvm.riscv.vid.mask.nxv2i8(
  <vscale x 2 x i8>,
  <vscale x 2 x i1>,
  i64);

define <vscale x 2 x i8> @intrinsic_vid_mask_v_nxv2i8(<vscale x 2 x i8> %0, <vscale x 2 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv2i8
; CHECK:       vsetvli {{.*}}, a0, e8,mf4,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x i8> @llvm.riscv.vid.mask.nxv2i8(
    <vscale x 2 x i8> %0,
    <vscale x 2 x i1> %1,
    i64 %2)

  ret <vscale x 2 x i8> %a
}

declare <vscale x 4 x i8> @llvm.riscv.vid.nxv4i8(
  i64);

define <vscale x 4 x i8> @intrinsic_vid_v_nxv4i8(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv4i8
; CHECK:       vsetvli {{.*}}, a0, e8,mf2,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 4 x i8> @llvm.riscv.vid.nxv4i8(
    i64 %0)

  ret <vscale x 4 x i8> %a
}

declare <vscale x 4 x i8> @llvm.riscv.vid.mask.nxv4i8(
  <vscale x 4 x i8>,
  <vscale x 4 x i1>,
  i64);

define <vscale x 4 x i8> @intrinsic_vid_mask_v_nxv4i8(<vscale x 4 x i8> %0, <vscale x 4 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv4i8
; CHECK:       vsetvli {{.*}}, a0, e8,mf2,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x i8> @llvm.riscv.vid.mask.nxv4i8(
    <vscale x 4 x i8> %0,
    <vscale x 4 x i1> %1,
    i64 %2)

  ret <vscale x 4 x i8> %a
}

declare <vscale x 8 x i8> @llvm.riscv.vid.nxv8i8(
  i64);

define <vscale x 8 x i8> @intrinsic_vid_v_nxv8i8(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv8i8
; CHECK:       vsetvli {{.*}}, a0, e8,m1,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 8 x i8> @llvm.riscv.vid.nxv8i8(
    i64 %0)

  ret <vscale x 8 x i8> %a
}

declare <vscale x 8 x i8> @llvm.riscv.vid.mask.nxv8i8(
  <vscale x 8 x i8>,
  <vscale x 8 x i1>,
  i64);

define <vscale x 8 x i8> @intrinsic_vid_mask_v_nxv8i8(<vscale x 8 x i8> %0, <vscale x 8 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv8i8
; CHECK:       vsetvli {{.*}}, a0, e8,m1,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 8 x i8> @llvm.riscv.vid.mask.nxv8i8(
    <vscale x 8 x i8> %0,
    <vscale x 8 x i1> %1,
    i64 %2)

  ret <vscale x 8 x i8> %a
}

declare <vscale x 16 x i8> @llvm.riscv.vid.nxv16i8(
  i64);

define <vscale x 16 x i8> @intrinsic_vid_v_nxv16i8(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv16i8
; CHECK:       vsetvli {{.*}}, a0, e8,m2,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 16 x i8> @llvm.riscv.vid.nxv16i8(
    i64 %0)

  ret <vscale x 16 x i8> %a
}

declare <vscale x 16 x i8> @llvm.riscv.vid.mask.nxv16i8(
  <vscale x 16 x i8>,
  <vscale x 16 x i1>,
  i64);

define <vscale x 16 x i8> @intrinsic_vid_mask_v_nxv16i8(<vscale x 16 x i8> %0, <vscale x 16 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv16i8
; CHECK:       vsetvli {{.*}}, a0, e8,m2,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 16 x i8> @llvm.riscv.vid.mask.nxv16i8(
    <vscale x 16 x i8> %0,
    <vscale x 16 x i1> %1,
    i64 %2)

  ret <vscale x 16 x i8> %a
}

declare <vscale x 32 x i8> @llvm.riscv.vid.nxv32i8(
  i64);

define <vscale x 32 x i8> @intrinsic_vid_v_nxv32i8(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv32i8
; CHECK:       vsetvli {{.*}}, a0, e8,m4,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 32 x i8> @llvm.riscv.vid.nxv32i8(
    i64 %0)

  ret <vscale x 32 x i8> %a
}

declare <vscale x 32 x i8> @llvm.riscv.vid.mask.nxv32i8(
  <vscale x 32 x i8>,
  <vscale x 32 x i1>,
  i64);

define <vscale x 32 x i8> @intrinsic_vid_mask_v_nxv32i8(<vscale x 32 x i8> %0, <vscale x 32 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv32i8
; CHECK:       vsetvli {{.*}}, a0, e8,m4,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 32 x i8> @llvm.riscv.vid.mask.nxv32i8(
    <vscale x 32 x i8> %0,
    <vscale x 32 x i1> %1,
    i64 %2)

  ret <vscale x 32 x i8> %a
}

declare <vscale x 1 x i16> @llvm.riscv.vid.nxv1i16(
  i64);

define <vscale x 1 x i16> @intrinsic_vid_v_nxv1i16(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv1i16
; CHECK:       vsetvli {{.*}}, a0, e16,mf4,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 1 x i16> @llvm.riscv.vid.nxv1i16(
    i64 %0)

  ret <vscale x 1 x i16> %a
}

declare <vscale x 1 x i16> @llvm.riscv.vid.mask.nxv1i16(
  <vscale x 1 x i16>,
  <vscale x 1 x i1>,
  i64);

define <vscale x 1 x i16> @intrinsic_vid_mask_v_nxv1i16(<vscale x 1 x i16> %0, <vscale x 1 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv1i16
; CHECK:       vsetvli {{.*}}, a0, e16,mf4,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 1 x i16> @llvm.riscv.vid.mask.nxv1i16(
    <vscale x 1 x i16> %0,
    <vscale x 1 x i1> %1,
    i64 %2)

  ret <vscale x 1 x i16> %a
}

declare <vscale x 2 x i16> @llvm.riscv.vid.nxv2i16(
  i64);

define <vscale x 2 x i16> @intrinsic_vid_v_nxv2i16(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv2i16
; CHECK:       vsetvli {{.*}}, a0, e16,mf2,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 2 x i16> @llvm.riscv.vid.nxv2i16(
    i64 %0)

  ret <vscale x 2 x i16> %a
}

declare <vscale x 2 x i16> @llvm.riscv.vid.mask.nxv2i16(
  <vscale x 2 x i16>,
  <vscale x 2 x i1>,
  i64);

define <vscale x 2 x i16> @intrinsic_vid_mask_v_nxv2i16(<vscale x 2 x i16> %0, <vscale x 2 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv2i16
; CHECK:       vsetvli {{.*}}, a0, e16,mf2,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x i16> @llvm.riscv.vid.mask.nxv2i16(
    <vscale x 2 x i16> %0,
    <vscale x 2 x i1> %1,
    i64 %2)

  ret <vscale x 2 x i16> %a
}

declare <vscale x 4 x i16> @llvm.riscv.vid.nxv4i16(
  i64);

define <vscale x 4 x i16> @intrinsic_vid_v_nxv4i16(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv4i16
; CHECK:       vsetvli {{.*}}, a0, e16,m1,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 4 x i16> @llvm.riscv.vid.nxv4i16(
    i64 %0)

  ret <vscale x 4 x i16> %a
}

declare <vscale x 4 x i16> @llvm.riscv.vid.mask.nxv4i16(
  <vscale x 4 x i16>,
  <vscale x 4 x i1>,
  i64);

define <vscale x 4 x i16> @intrinsic_vid_mask_v_nxv4i16(<vscale x 4 x i16> %0, <vscale x 4 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv4i16
; CHECK:       vsetvli {{.*}}, a0, e16,m1,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x i16> @llvm.riscv.vid.mask.nxv4i16(
    <vscale x 4 x i16> %0,
    <vscale x 4 x i1> %1,
    i64 %2)

  ret <vscale x 4 x i16> %a
}

declare <vscale x 8 x i16> @llvm.riscv.vid.nxv8i16(
  i64);

define <vscale x 8 x i16> @intrinsic_vid_v_nxv8i16(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv8i16
; CHECK:       vsetvli {{.*}}, a0, e16,m2,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 8 x i16> @llvm.riscv.vid.nxv8i16(
    i64 %0)

  ret <vscale x 8 x i16> %a
}

declare <vscale x 8 x i16> @llvm.riscv.vid.mask.nxv8i16(
  <vscale x 8 x i16>,
  <vscale x 8 x i1>,
  i64);

define <vscale x 8 x i16> @intrinsic_vid_mask_v_nxv8i16(<vscale x 8 x i16> %0, <vscale x 8 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv8i16
; CHECK:       vsetvli {{.*}}, a0, e16,m2,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 8 x i16> @llvm.riscv.vid.mask.nxv8i16(
    <vscale x 8 x i16> %0,
    <vscale x 8 x i1> %1,
    i64 %2)

  ret <vscale x 8 x i16> %a
}

declare <vscale x 16 x i16> @llvm.riscv.vid.nxv16i16(
  i64);

define <vscale x 16 x i16> @intrinsic_vid_v_nxv16i16(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv16i16
; CHECK:       vsetvli {{.*}}, a0, e16,m4,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 16 x i16> @llvm.riscv.vid.nxv16i16(
    i64 %0)

  ret <vscale x 16 x i16> %a
}

declare <vscale x 16 x i16> @llvm.riscv.vid.mask.nxv16i16(
  <vscale x 16 x i16>,
  <vscale x 16 x i1>,
  i64);

define <vscale x 16 x i16> @intrinsic_vid_mask_v_nxv16i16(<vscale x 16 x i16> %0, <vscale x 16 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv16i16
; CHECK:       vsetvli {{.*}}, a0, e16,m4,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 16 x i16> @llvm.riscv.vid.mask.nxv16i16(
    <vscale x 16 x i16> %0,
    <vscale x 16 x i1> %1,
    i64 %2)

  ret <vscale x 16 x i16> %a
}

declare <vscale x 32 x i16> @llvm.riscv.vid.nxv32i16(
  i64);

define <vscale x 32 x i16> @intrinsic_vid_v_nxv32i16(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv32i16
; CHECK:       vsetvli {{.*}}, a0, e16,m8,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 32 x i16> @llvm.riscv.vid.nxv32i16(
    i64 %0)

  ret <vscale x 32 x i16> %a
}

declare <vscale x 32 x i16> @llvm.riscv.vid.mask.nxv32i16(
  <vscale x 32 x i16>,
  <vscale x 32 x i1>,
  i64);

define <vscale x 32 x i16> @intrinsic_vid_mask_v_nxv32i16(<vscale x 32 x i16> %0, <vscale x 32 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv32i16
; CHECK:       vsetvli {{.*}}, a0, e16,m8,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 32 x i16> @llvm.riscv.vid.mask.nxv32i16(
    <vscale x 32 x i16> %0,
    <vscale x 32 x i1> %1,
    i64 %2)

  ret <vscale x 32 x i16> %a
}

declare <vscale x 1 x i32> @llvm.riscv.vid.nxv1i32(
  i64);

define <vscale x 1 x i32> @intrinsic_vid_v_nxv1i32(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv1i32
; CHECK:       vsetvli {{.*}}, a0, e32,mf2,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 1 x i32> @llvm.riscv.vid.nxv1i32(
    i64 %0)

  ret <vscale x 1 x i32> %a
}

declare <vscale x 1 x i32> @llvm.riscv.vid.mask.nxv1i32(
  <vscale x 1 x i32>,
  <vscale x 1 x i1>,
  i64);

define <vscale x 1 x i32> @intrinsic_vid_mask_v_nxv1i32(<vscale x 1 x i32> %0, <vscale x 1 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv1i32
; CHECK:       vsetvli {{.*}}, a0, e32,mf2,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 1 x i32> @llvm.riscv.vid.mask.nxv1i32(
    <vscale x 1 x i32> %0,
    <vscale x 1 x i1> %1,
    i64 %2)

  ret <vscale x 1 x i32> %a
}

declare <vscale x 2 x i32> @llvm.riscv.vid.nxv2i32(
  i64);

define <vscale x 2 x i32> @intrinsic_vid_v_nxv2i32(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv2i32
; CHECK:       vsetvli {{.*}}, a0, e32,m1,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 2 x i32> @llvm.riscv.vid.nxv2i32(
    i64 %0)

  ret <vscale x 2 x i32> %a
}

declare <vscale x 2 x i32> @llvm.riscv.vid.mask.nxv2i32(
  <vscale x 2 x i32>,
  <vscale x 2 x i1>,
  i64);

define <vscale x 2 x i32> @intrinsic_vid_mask_v_nxv2i32(<vscale x 2 x i32> %0, <vscale x 2 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv2i32
; CHECK:       vsetvli {{.*}}, a0, e32,m1,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x i32> @llvm.riscv.vid.mask.nxv2i32(
    <vscale x 2 x i32> %0,
    <vscale x 2 x i1> %1,
    i64 %2)

  ret <vscale x 2 x i32> %a
}

declare <vscale x 4 x i32> @llvm.riscv.vid.nxv4i32(
  i64);

define <vscale x 4 x i32> @intrinsic_vid_v_nxv4i32(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv4i32
; CHECK:       vsetvli {{.*}}, a0, e32,m2,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 4 x i32> @llvm.riscv.vid.nxv4i32(
    i64 %0)

  ret <vscale x 4 x i32> %a
}

declare <vscale x 4 x i32> @llvm.riscv.vid.mask.nxv4i32(
  <vscale x 4 x i32>,
  <vscale x 4 x i1>,
  i64);

define <vscale x 4 x i32> @intrinsic_vid_mask_v_nxv4i32(<vscale x 4 x i32> %0, <vscale x 4 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv4i32
; CHECK:       vsetvli {{.*}}, a0, e32,m2,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x i32> @llvm.riscv.vid.mask.nxv4i32(
    <vscale x 4 x i32> %0,
    <vscale x 4 x i1> %1,
    i64 %2)

  ret <vscale x 4 x i32> %a
}

declare <vscale x 8 x i32> @llvm.riscv.vid.nxv8i32(
  i64);

define <vscale x 8 x i32> @intrinsic_vid_v_nxv8i32(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv8i32
; CHECK:       vsetvli {{.*}}, a0, e32,m4,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 8 x i32> @llvm.riscv.vid.nxv8i32(
    i64 %0)

  ret <vscale x 8 x i32> %a
}

declare <vscale x 8 x i32> @llvm.riscv.vid.mask.nxv8i32(
  <vscale x 8 x i32>,
  <vscale x 8 x i1>,
  i64);

define <vscale x 8 x i32> @intrinsic_vid_mask_v_nxv8i32(<vscale x 8 x i32> %0, <vscale x 8 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv8i32
; CHECK:       vsetvli {{.*}}, a0, e32,m4,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 8 x i32> @llvm.riscv.vid.mask.nxv8i32(
    <vscale x 8 x i32> %0,
    <vscale x 8 x i1> %1,
    i64 %2)

  ret <vscale x 8 x i32> %a
}

declare <vscale x 16 x i32> @llvm.riscv.vid.nxv16i32(
  i64);

define <vscale x 16 x i32> @intrinsic_vid_v_nxv16i32(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv16i32
; CHECK:       vsetvli {{.*}}, a0, e32,m8,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 16 x i32> @llvm.riscv.vid.nxv16i32(
    i64 %0)

  ret <vscale x 16 x i32> %a
}

declare <vscale x 16 x i32> @llvm.riscv.vid.mask.nxv16i32(
  <vscale x 16 x i32>,
  <vscale x 16 x i1>,
  i64);

define <vscale x 16 x i32> @intrinsic_vid_mask_v_nxv16i32(<vscale x 16 x i32> %0, <vscale x 16 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv16i32
; CHECK:       vsetvli {{.*}}, a0, e32,m8,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 16 x i32> @llvm.riscv.vid.mask.nxv16i32(
    <vscale x 16 x i32> %0,
    <vscale x 16 x i1> %1,
    i64 %2)

  ret <vscale x 16 x i32> %a
}

declare <vscale x 1 x i64> @llvm.riscv.vid.nxv1i64(
  i64);

define <vscale x 1 x i64> @intrinsic_vid_v_nxv1i64(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv1i64
; CHECK:       vsetvli {{.*}}, a0, e64,m1,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 1 x i64> @llvm.riscv.vid.nxv1i64(
    i64 %0)

  ret <vscale x 1 x i64> %a
}

declare <vscale x 1 x i64> @llvm.riscv.vid.mask.nxv1i64(
  <vscale x 1 x i64>,
  <vscale x 1 x i1>,
  i64);

define <vscale x 1 x i64> @intrinsic_vid_mask_v_nxv1i64(<vscale x 1 x i64> %0, <vscale x 1 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv1i64
; CHECK:       vsetvli {{.*}}, a0, e64,m1,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 1 x i64> @llvm.riscv.vid.mask.nxv1i64(
    <vscale x 1 x i64> %0,
    <vscale x 1 x i1> %1,
    i64 %2)

  ret <vscale x 1 x i64> %a
}

declare <vscale x 2 x i64> @llvm.riscv.vid.nxv2i64(
  i64);

define <vscale x 2 x i64> @intrinsic_vid_v_nxv2i64(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv2i64
; CHECK:       vsetvli {{.*}}, a0, e64,m2,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 2 x i64> @llvm.riscv.vid.nxv2i64(
    i64 %0)

  ret <vscale x 2 x i64> %a
}

declare <vscale x 2 x i64> @llvm.riscv.vid.mask.nxv2i64(
  <vscale x 2 x i64>,
  <vscale x 2 x i1>,
  i64);

define <vscale x 2 x i64> @intrinsic_vid_mask_v_nxv2i64(<vscale x 2 x i64> %0, <vscale x 2 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv2i64
; CHECK:       vsetvli {{.*}}, a0, e64,m2,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x i64> @llvm.riscv.vid.mask.nxv2i64(
    <vscale x 2 x i64> %0,
    <vscale x 2 x i1> %1,
    i64 %2)

  ret <vscale x 2 x i64> %a
}

declare <vscale x 4 x i64> @llvm.riscv.vid.nxv4i64(
  i64);

define <vscale x 4 x i64> @intrinsic_vid_v_nxv4i64(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv4i64
; CHECK:       vsetvli {{.*}}, a0, e64,m4,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 4 x i64> @llvm.riscv.vid.nxv4i64(
    i64 %0)

  ret <vscale x 4 x i64> %a
}

declare <vscale x 4 x i64> @llvm.riscv.vid.mask.nxv4i64(
  <vscale x 4 x i64>,
  <vscale x 4 x i1>,
  i64);

define <vscale x 4 x i64> @intrinsic_vid_mask_v_nxv4i64(<vscale x 4 x i64> %0, <vscale x 4 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv4i64
; CHECK:       vsetvli {{.*}}, a0, e64,m4,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x i64> @llvm.riscv.vid.mask.nxv4i64(
    <vscale x 4 x i64> %0,
    <vscale x 4 x i1> %1,
    i64 %2)

  ret <vscale x 4 x i64> %a
}

declare <vscale x 8 x i64> @llvm.riscv.vid.nxv8i64(
  i64);

define <vscale x 8 x i64> @intrinsic_vid_v_nxv8i64(i64 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_v_nxv8i64
; CHECK:       vsetvli {{.*}}, a0, e64,m8,ta,mu
; CHECK:       vid.v {{v[0-9]+}}
  %a = call <vscale x 8 x i64> @llvm.riscv.vid.nxv8i64(
    i64 %0)

  ret <vscale x 8 x i64> %a
}

declare <vscale x 8 x i64> @llvm.riscv.vid.mask.nxv8i64(
  <vscale x 8 x i64>,
  <vscale x 8 x i1>,
  i64);

define <vscale x 8 x i64> @intrinsic_vid_mask_v_nxv8i64(<vscale x 8 x i64> %0, <vscale x 8 x i1> %1, i64 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vid_mask_v_nxv8i64
; CHECK:       vsetvli {{.*}}, a0, e64,m8,tu,mu
; CHECK:       vid.v {{v[0-9]+}}, v0.t
  %a = call <vscale x 8 x i64> @llvm.riscv.vid.mask.nxv8i64(
    <vscale x 8 x i64> %0,
    <vscale x 8 x i1> %1,
    i64 %2)

  ret <vscale x 8 x i64> %a
}
