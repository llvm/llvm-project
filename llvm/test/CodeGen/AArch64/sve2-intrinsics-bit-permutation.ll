; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2,+sve2-bitperm < %s | FileCheck %s

;
; BDEP
;

define <vscale x 16 x i8> @bdep_nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: bdep_nxv16i8:
; CHECK: bdep z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.bdep.x.nx16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @bdep_nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: bdep_nxv8i16:
; CHECK: bdep z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.bdep.x.nx8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @bdep_nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: bdep_nxv4i32:
; CHECK: bdep z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.bdep.x.nx4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @bdep_nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: bdep_nxv2i64:
; CHECK: bdep z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.bdep.x.nx2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; BEXT
;

define <vscale x 16 x i8> @bext_nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: bext_nxv16i8:
; CHECK: bext z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.bext.x.nx16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @bext_nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: bext_nxv8i16:
; CHECK: bext z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.bext.x.nx8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @bext_nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: bext_nxv4i32:
; CHECK: bext z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.bext.x.nx4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @bext_nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: bext_nxv2i64:
; CHECK: bext z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.bext.x.nx2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; BGRP
;

define <vscale x 16 x i8> @bgrp_nxv16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: bgrp_nxv16i8:
; CHECK: bgrp z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.bgrp.x.nx16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @bgrp_nxv8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: bgrp_nxv8i16:
; CHECK: bgrp z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.bgrp.x.nx8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @bgrp_nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: bgrp_nxv4i32:
; CHECK: bgrp z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.bgrp.x.nx4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @bgrp_nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: bgrp_nxv2i64:
; CHECK: bgrp z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.bgrp.x.nx2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.bdep.x.nx16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b)
declare <vscale x 8 x i16> @llvm.aarch64.sve.bdep.x.nx8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b)
declare <vscale x 4 x i32> @llvm.aarch64.sve.bdep.x.nx4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b)
declare <vscale x 2 x i64> @llvm.aarch64.sve.bdep.x.nx2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b)

declare <vscale x 16 x i8> @llvm.aarch64.sve.bext.x.nx16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b)
declare <vscale x 8 x i16> @llvm.aarch64.sve.bext.x.nx8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b)
declare <vscale x 4 x i32> @llvm.aarch64.sve.bext.x.nx4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b)
declare <vscale x 2 x i64> @llvm.aarch64.sve.bext.x.nx2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b)

declare <vscale x 16 x i8> @llvm.aarch64.sve.bgrp.x.nx16i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b)
declare <vscale x 8 x i16> @llvm.aarch64.sve.bgrp.x.nx8i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b)
declare <vscale x 4 x i32> @llvm.aarch64.sve.bgrp.x.nx4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b)
declare <vscale x 2 x i64> @llvm.aarch64.sve.bgrp.x.nx2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b)
