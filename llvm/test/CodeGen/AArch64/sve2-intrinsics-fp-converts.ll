; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme < %s | FileCheck %s

;
; FCVTLT
;

define <vscale x 4 x float> @fcvtlt_f32_f16(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: fcvtlt_f32_f16:
; CHECK: fcvtlt z0.s, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fcvtlt.f32f16(<vscale x 4 x float> %a,
                                                                   <vscale x 4 x i1> %pg,
                                                                   <vscale x 8 x half> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fcvtlt_f64_f32(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: fcvtlt_f64_f32:
; CHECK: fcvtlt	z0.d, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fcvtlt.f64f32(<vscale x 2 x double> %a,
                                                                    <vscale x 2 x i1> %pg,
                                                                    <vscale x 4 x float> %b)
  ret <vscale x 2 x double> %out
}

;
; FCVTNT
;

define <vscale x 8 x half> @fcvtnt_f16_f32(<vscale x 8 x half> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: fcvtnt_f16_f32:
; CHECK: fcvtnt z0.h, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fcvtnt.f16f32(<vscale x 8 x half> %a,
                                                             <vscale x 4 x i1> %pg,
                                                             <vscale x 4 x float> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fcvtnt_f32_f64(<vscale x 4 x float> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: fcvtnt_f32_f64:
; CHECK: fcvtnt	z0.s, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fcvtnt.f32f64(<vscale x 4 x float> %a,
                                                                   <vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 4 x float> %out
}

;
; FCVTX
;

define <vscale x 4 x float> @fcvtx_f32_f64(<vscale x 4 x float> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: fcvtx_f32_f64:
; CHECK: fcvtx z0.s, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fcvtx.f32f64(<vscale x 4 x float> %a,
                                                                  <vscale x 2 x i1> %pg,
                                                                  <vscale x 2 x double> %b)
  ret <vscale x 4 x float> %out
}

;
; FCVTXNT
;

define <vscale x 4 x float> @fcvtxnt_f32_f64(<vscale x 4 x float> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: fcvtxnt_f32_f64:
; CHECK: fcvtxnt z0.s, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fcvtxnt.f32f64(<vscale x 4 x float> %a,
                                                                    <vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 4 x float> %out
}

declare <vscale x 4 x float> @llvm.aarch64.sve.fcvtlt.f32f16(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 8 x half>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fcvtlt.f64f32(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 4 x float>)
declare <vscale x 8 x half> @llvm.aarch64.sve.fcvtnt.f16f32(<vscale x 8 x half>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fcvtnt.f32f64(<vscale x 4 x float>, <vscale x 2 x i1>, <vscale x 2 x double>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fcvtx.f32f64(<vscale x 4 x float>, <vscale x 2 x i1>, <vscale x 2 x double>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fcvtxnt.f32f64(<vscale x 4 x float>, <vscale x 2 x i1>, <vscale x 2 x double>)
