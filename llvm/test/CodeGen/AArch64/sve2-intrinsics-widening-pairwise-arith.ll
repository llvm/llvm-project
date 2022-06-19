; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme < %s | FileCheck %s

;
; SADALP
;

define <vscale x 8 x i16> @sadalp_i8(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sadalp_i8:
; CHECK: sadalp z0.h, p0/m, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sadalp.nxv8i16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sadalp_i16(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sadalp_i16:
; CHECK: sadalp z0.s, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sadalp.nxv4i32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sadalp_i32(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sadalp_i32:
; CHECK: sadalp z0.d, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sadalp.nxv2i64(<vscale x 2 x i1> %pg,
                                                                  <vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; UADALP
;

define <vscale x 8 x i16> @uadalp_i8(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uadalp_i8:
; CHECK: uadalp z0.h, p0/m, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uadalp.nxv8i16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uadalp_i16(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uadalp_i16:
; CHECK: uadalp z0.s, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uadalp.nxv4i32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uadalp_i32(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uadalp_i32:
; CHECK: uadalp z0.d, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uadalp.nxv2i64(<vscale x 2 x i1> %pg,
                                                                  <vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.sadalp.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sadalp.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sadalp.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uadalp.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uadalp.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uadalp.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 4 x i32>)
