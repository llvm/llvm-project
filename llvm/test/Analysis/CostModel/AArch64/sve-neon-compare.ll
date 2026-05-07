; RUN: opt < %s -passes='print<cost-model>' -disable-output 2>&1 | FileCheck %s --check-prefix=NEON
; RUN: opt < %s -passes='print<cost-model>' -disable-output -mattr=+sve 2>&1 | FileCheck %s --check-prefix=SVE

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; --- Fixed-width (NEON) types ---

define <16 x i8> @smin_v16i8(<16 x i8> %a, <16 x i8> %b) {
; NEON: cost of {{.*}} call {{.*}} @llvm.smin.v16i8
; SVE:  cost of {{.*}} call {{.*}} @llvm.smin.v16i8
  %r = call <16 x i8> @llvm.smin.v16i8(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %r
}

define <4 x i32> @smin_v4i32(<4 x i32> %a, <4 x i32> %b) {
; NEON: cost of {{.*}} call {{.*}} @llvm.smin.v4i32
; SVE:  cost of {{.*}} call {{.*}} @llvm.smin.v4i32
  %r = call <4 x i32> @llvm.smin.v4i32(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %r
}

; --- Scalable (SVE) types ---

define <vscale x 16 x i8> @smin_nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; NEON: cost of {{.*}} call {{.*}} @llvm.smin.nxv16i8
; SVE:  cost of {{.*}} call {{.*}} @llvm.smin.nxv16i8
  %r = call <vscale x 16 x i8> @llvm.smin.nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %r
}

define <vscale x 4 x i32> @smin_nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; NEON: cost of {{.*}} call {{.*}} @llvm.smin.nxv4i32
; SVE:  cost of {{.*}} call {{.*}} @llvm.smin.nxv4i32
  %r = call <vscale x 4 x i32> @llvm.smin.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %r
}

define <vscale x 2 x i64> @smin_nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; NEON: cost of {{.*}} call {{.*}} @llvm.smin.nxv2i64
; SVE:  cost of {{.*}} call {{.*}} @llvm.smin.nxv2i64
  %r = call <vscale x 2 x i64> @llvm.smin.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %r
}
