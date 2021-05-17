; RUN: opt -S -instcombine < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define <vscale x 16 x i8> @dup_insertelement_0(<vscale x 16 x i8> %v, i8 %s) #0 {
; CHECK-LABEL: @dup_insertelement_0(
; CHECK: %insert = insertelement <vscale x 16 x i8> %v, i8 %s, i64 0
; CHECK-NEXT: ret <vscale x 16 x i8> %insert
  %pg = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 1)
  %insert = tail call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> %v, <vscale x 16 x i1> %pg, i8 %s)
  ret <vscale x 16 x i8> %insert
}

define <vscale x 16 x i8> @dup_insertelement_1(<vscale x 16 x i8> %v, i8 %s) #0 {
; CHECK-LABEL: @dup_insertelement_1(
; CHECK: %pg = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 2)
; CHECK-NEXT: %insert = tail call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> %v, <vscale x 16 x i1> %pg, i8 %s)
; CHECK-NEXT: ret <vscale x 16 x i8> %insert
  %pg = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 2)
  %insert = tail call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> %v, <vscale x 16 x i1> %pg, i8 %s)
  ret <vscale x 16 x i8> %insert
}

define <vscale x 16 x i8> @dup_insertelement_x(<vscale x 16 x i8> %v, i8 %s, <vscale x 16 x i1> %pg) #0 {
; CHECK-LABEL: @dup_insertelement_x(
; CHECK: %insert = tail call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> %v, <vscale x 16 x i1> %pg, i8 %s)
; CHECK-NEXT: ret <vscale x 16 x i8> %insert
  %insert = tail call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> %v, <vscale x 16 x i1> %pg, i8 %s)
  ret <vscale x 16 x i8> %insert
}

define <vscale x 8 x i16> @dup_insertelement_0_convert(<vscale x 8 x i16> %v, i16 %s) #0 {
; CHECK-LABEL: @dup_insertelement_0_convert(
; CHECK: %insert = insertelement <vscale x 8 x i16> %v, i16 %s, i64 0
; CHECK-NEXT: ret <vscale x 8 x i16> %insert
  %pg = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 1)
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %pg)
  %2 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %1)
  %insert = tail call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> %v, <vscale x 8 x i1> %2, i16 %s)
  ret <vscale x 8 x i16> %insert
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, i8)
declare <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, i16)

declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32)

declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1>)

attributes #0 = { "target-features"="+sve" }
