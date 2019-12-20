; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; SUNPKHI
;

define <vscale x 8 x i16> @sunpkhi_i16(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sunpkhi_i16
; CHECK: sunpkhi z0.h, z0.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.sunpkhi.nxv8i16(<vscale x 16 x i8> %a)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sunpkhi_i32(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sunpkhi_i32
; CHECK: sunpkhi z0.s, z0.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sunpkhi.nxv4i32(<vscale x 8 x i16> %a)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sunpkhi_i64(<vscale x 4 x i32> %a) {
; CHECK-LABEL:  sunpkhi_i64
; CHECK: sunpkhi z0.d, z0.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sunpkhi.nxv2i64(<vscale x 4 x i32> %a)
  ret <vscale x 2 x i64> %res
}

;
; SUNPKLO
;

define <vscale x 8 x i16> @sunpklo_i16(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sunpklo_i16
; CHECK: sunpklo z0.h, z0.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.sunpklo.nxv8i16(<vscale x 16 x i8> %a)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sunpklo_i32(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sunpklo_i32
; CHECK: sunpklo z0.s, z0.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sunpklo.nxv4i32(<vscale x 8 x i16> %a)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sunpklo_i64(<vscale x 4 x i32> %a) {
; CHECK-LABEL:  sunpklo_i64
; CHECK: sunpklo z0.d, z0.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sunpklo.nxv2i64(<vscale x 4 x i32> %a)
  ret <vscale x 2 x i64> %res
}

;
; UUNPKHI
;

define <vscale x 8 x i16> @uunpkhi_i16(<vscale x 16 x i8> %a) {
; CHECK-LABEL: uunpkhi_i16
; CHECK: uunpkhi z0.h, z0.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.uunpkhi.nxv8i16(<vscale x 16 x i8> %a)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @uunpkhi_i32(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uunpkhi_i32
; CHECK: uunpkhi z0.s, z0.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.uunpkhi.nxv4i32(<vscale x 8 x i16> %a)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @uunpkhi_i64(<vscale x 4 x i32> %a) {
; CHECK-LABEL:  uunpkhi_i64
; CHECK: uunpkhi z0.d, z0.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.uunpkhi.nxv2i64(<vscale x 4 x i32> %a)
  ret <vscale x 2 x i64> %res
}

;
; UUNPKLO
;

define <vscale x 8 x i16> @uunpklo_i16(<vscale x 16 x i8> %a) {
; CHECK-LABEL: uunpklo_i16
; CHECK: uunpklo z0.h, z0.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.uunpklo.nxv8i16(<vscale x 16 x i8> %a)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @uunpklo_i32(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uunpklo_i32
; CHECK: uunpklo z0.s, z0.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.uunpklo.nxv4i32(<vscale x 8 x i16> %a)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @uunpklo_i64(<vscale x 4 x i32> %a) {
; CHECK-LABEL:  uunpklo_i64
; CHECK: uunpklo z0.d, z0.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.uunpklo.nxv2i64(<vscale x 4 x i32> %a)
  ret <vscale x 2 x i64> %res
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.sunpkhi.nxv8i16(<vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sunpkhi.nxv4i32(<vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sunpkhi.nxv2i64(<vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sunpklo.nxv8i16(<vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sunpklo.nxv4i32(<vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sunpklo.nxv2i64(<vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uunpkhi.nxv8i16(<vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uunpkhi.nxv4i32(<vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uunpkhi.nxv2i64(<vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uunpklo.nxv8i16(<vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uunpklo.nxv4i32(<vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uunpklo.nxv2i64(<vscale x 4 x i32>)
