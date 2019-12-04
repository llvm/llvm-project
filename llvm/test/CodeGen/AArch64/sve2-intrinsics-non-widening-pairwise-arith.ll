; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s

;
; FADDP
;

define <vscale x 8 x half> @faddp_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: faddp_f16:
; CHECK: faddp z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.faddp.nxv8f16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @faddp_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: faddp_f32:
; CHECK: faddp z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.faddp.nxv4f32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @faddp_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: faddp_f64:
; CHECK: faddp z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.faddp.nxv2f64(<vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMAXP
;

define <vscale x 8 x half> @fmaxp_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fmaxp_f16:
; CHECK: fmaxp z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmaxp.nxv8f16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmaxp_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fmaxp_f32:
; CHECK: fmaxp z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmaxp.nxv4f32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmaxp_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fmaxp_f64:
; CHECK: fmaxp z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmaxp.nxv2f64(<vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMAXNMP
;

define <vscale x 8 x half> @fmaxnmp_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fmaxnmp_f16:
; CHECK: fmaxnmp z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmaxnmp.nxv8f16(<vscale x 8 x i1> %pg,
                                                                    <vscale x 8 x half> %a,
                                                                    <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmaxnmp_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fmaxnmp_f32:
; CHECK: fmaxnmp z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmaxnmp.nxv4f32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x float> %a,
                                                                     <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmaxnmp_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fmaxnmp_f64:
; CHECK: fmaxnmp z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmaxnmp.nxv2f64(<vscale x 2 x i1> %pg,
                                                                      <vscale x 2 x double> %a,
                                                                      <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMINP
;

define <vscale x 8 x half> @fminp_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fminp_f16:
; CHECK: fminp z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fminp.nxv8f16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fminp_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fminp_f32:
; CHECK: fminp z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fminp.nxv4f32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fminp_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fminp_f64:
; CHECK: fminp z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fminp.nxv2f64(<vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMINNMP
;

define <vscale x 8 x half> @fminnmp_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fminnmp_f16:
; CHECK: fminnmp z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fminnmp.nxv8f16(<vscale x 8 x i1> %pg,
                                                                    <vscale x 8 x half> %a,
                                                                    <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fminnmp_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fminnmp_f32:
; CHECK: fminnmp z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fminnmp.nxv4f32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x float> %a,
                                                                     <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fminnmp_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fminnmp_f64:
; CHECK: fminnmp z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fminnmp.nxv2f64(<vscale x 2 x i1> %pg,
                                                                      <vscale x 2 x double> %a,
                                                                      <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

declare <vscale x 8 x half> @llvm.aarch64.sve.faddp.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.faddp.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.faddp.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmaxp.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmaxp.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmaxp.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmaxnmp.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmaxnmp.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmaxnmp.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fminp.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fminp.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fminp.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fminnmp.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fminnmp.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fminnmp.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)
