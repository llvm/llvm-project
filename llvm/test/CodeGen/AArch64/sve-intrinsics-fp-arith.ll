; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; FABD
;

define <vscale x 8 x half> @fabd_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fabd_h:
; CHECK: fabd z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fabd.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fabd_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fabd_s:
; CHECK: fabd z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fabd.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fabd_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fabd_d:
; CHECK: fabd z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fabd.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FADD
;

define <vscale x 8 x half> @fadd_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fadd_h:
; CHECK: fadd z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fadd_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fadd_s:
; CHECK: fadd z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fadd_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fadd_d:
; CHECK: fadd z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fadd.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FCADD
;

define <vscale x 8 x half> @fcadd_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fcadd_h:
; CHECK: fcadd z0.h, p0/m, z0.h, z1.h, #90
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fcadd.nxv8f16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b,
                                                                  i32 90)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fcadd_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fcadd_s:
; CHECK: fcadd z0.s, p0/m, z0.s, z1.s, #270
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fcadd.nxv4f32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b,
                                                                   i32 270)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fcadd_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fcadd_d:
; CHECK: fcadd z0.d, p0/m, z0.d, z1.d, #90
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fcadd.nxv2f64(<vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b,
                                                                    i32 90)
  ret <vscale x 2 x double> %out
}

;
; FCMLA
;

define <vscale x 8 x half> @fcmla_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fcmla_h:
; CHECK: fcmla z0.h, p0/m, z1.h, z2.h, #90
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fcmla.nxv8f16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b,
                                                                  <vscale x 8 x half> %c,
                                                                  i32 90)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fcmla_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x float> %c) {
; CHECK-LABEL: fcmla_s:
; CHECK: fcmla z0.s, p0/m, z1.s, z2.s, #180
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fcmla.nxv4f32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b,
                                                                   <vscale x 4 x float> %c,
                                                                   i32 180)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fcmla_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x double> %c) {
; CHECK-LABEL: fcmla_d:
; CHECK: fcmla z0.d, p0/m, z1.d, z2.d, #270
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fcmla.nxv2f64(<vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b,
                                                                    <vscale x 2 x double> %c,
                                                                    i32 270)
  ret <vscale x 2 x double> %out
}

;
; FCMLA (Indexed)
;

define <vscale x 8 x half> @fcmla_lane_h(<vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fcmla_lane_h:
; CHECK: fcmla z0.h, z1.h, z2.h[3], #0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fcmla.lane.nxv8f16(<vscale x 8 x half> %a,
                                                                       <vscale x 8 x half> %b,
                                                                       <vscale x 8 x half> %c,
                                                                       i32 3,
                                                                       i32 0)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fcmla_lane_s(<vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x float> %c) {
; CHECK-LABEL: fcmla_lane_s:
; CHECK: fcmla z0.s, z1.s, z2.s[1], #90
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fcmla.lane.nxv4f32(<vscale x 4 x float> %a,
                                                                        <vscale x 4 x float> %b,
                                                                        <vscale x 4 x float> %c,
                                                                        i32 1,
                                                                        i32 90)
  ret <vscale x 4 x float> %out
}

;
; FDIV
;

define <vscale x 8 x half> @fdiv_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fdiv_h:
; CHECK: fdiv z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fdiv.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fdiv_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fdiv_s:
; CHECK: fdiv z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fdiv.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fdiv_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fdiv_d:
; CHECK: fdiv z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fdiv.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FDIVR
;

define <vscale x 8 x half> @fdivr_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fdivr_h:
; CHECK: fdivr z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fdivr.nxv8f16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fdivr_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fdivr_s:
; CHECK: fdivr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fdivr.nxv4f32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fdivr_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fdivr_d:
; CHECK: fdivr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fdivr.nxv2f64(<vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMAD
;

define <vscale x 8 x half> @fmad_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmad_h:
; CHECK: fmad z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmad.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b,
                                                                 <vscale x 8 x half> %c)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmad_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x float> %c) {
; CHECK-LABEL: fmad_s:
; CHECK: fmad z0.s, p0/m, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmad.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b,
                                                                  <vscale x 4 x float> %c)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmad_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x double> %c) {
; CHECK-LABEL: fmad_d:
; CHECK: fmad z0.d, p0/m, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmad.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b,
                                                                   <vscale x 2 x double> %c)
  ret <vscale x 2 x double> %out
}

;
; FMAX
;

define <vscale x 8 x half> @fmax_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fmax_h:
; CHECK: fmax z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmax.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmax_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fmax_s:
; CHECK: fmax z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmax.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmax_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fmax_d:
; CHECK: fmax z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmax.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMAXNM
;

define <vscale x 8 x half> @fmaxnm_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fmaxnm_h:
; CHECK: fmaxnm z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmaxnm.nxv8f16(<vscale x 8 x i1> %pg,
                                                                   <vscale x 8 x half> %a,
                                                                   <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmaxnm_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fmaxnm_s:
; CHECK: fmaxnm z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmaxnm.nxv4f32(<vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x float> %a,
                                                                    <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmaxnm_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fmaxnm_d:
; CHECK: fmaxnm z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmaxnm.nxv2f64(<vscale x 2 x i1> %pg,
                                                                     <vscale x 2 x double> %a,
                                                                     <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMIN
;

define <vscale x 8 x half> @fmin_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fmin_h:
; CHECK: fmin z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmin.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmin_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fmin_s:
; CHECK: fmin z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmin.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmin_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fmin_d:
; CHECK: fmin z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmin.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMINNM
;

define <vscale x 8 x half> @fminnm_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fminnm_h:
; CHECK: fminnm z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fminnm.nxv8f16(<vscale x 8 x i1> %pg,
                                                                   <vscale x 8 x half> %a,
                                                                   <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fminnm_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fminnm_s:
; CHECK: fminnm z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fminnm.nxv4f32(<vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x float> %a,
                                                                    <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fminnm_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fminnm_d:
; CHECK: fminnm z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fminnm.nxv2f64(<vscale x 2 x i1> %pg,
                                                                     <vscale x 2 x double> %a,
                                                                     <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMLA
;

define <vscale x 8 x half> @fmla_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmla_h:
; CHECK: fmla z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmla.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b,
                                                                 <vscale x 8 x half> %c)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmla_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x float> %c) {
; CHECK-LABEL: fmla_s:
; CHECK: fmla z0.s, p0/m, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmla.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b,
                                                                  <vscale x 4 x float> %c)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmla_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x double> %c) {
; CHECK-LABEL: fmla_d:
; CHECK: fmla z0.d, p0/m, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmla.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b,
                                                                   <vscale x 2 x double> %c)
  ret <vscale x 2 x double> %out
}

;
; FMLA (Indexed)
;

define <vscale x 8 x half> @fmla_lane_h(<vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmla_lane_h:
; CHECK: fmla z0.h, z1.h, z2.h[3]
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmla.lane.nxv8f16(<vscale x 8 x half> %a,
                                                                      <vscale x 8 x half> %b,
                                                                      <vscale x 8 x half> %c,
                                                                      i32 3)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmla_lane_s(<vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x float> %c) {
; CHECK-LABEL: fmla_lane_s:
; CHECK: fmla z0.s, z1.s, z2.s[2]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmla.lane.nxv4f32(<vscale x 4 x float> %a,
                                                                       <vscale x 4 x float> %b,
                                                                       <vscale x 4 x float> %c,
                                                                       i32 2)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmla_lane_d(<vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x double> %c) {
; CHECK-LABEL: fmla_lane_d:
; CHECK: fmla z0.d, z1.d, z2.d[1]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmla.lane.nxv2f64(<vscale x 2 x double> %a,
                                                                        <vscale x 2 x double> %b,
                                                                        <vscale x 2 x double> %c,
                                                                        i32 1)
  ret <vscale x 2 x double> %out
}

;
; FMLS
;

define <vscale x 8 x half> @fmls_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmls_h:
; CHECK: fmls z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmls.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b,
                                                                 <vscale x 8 x half> %c)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmls_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x float> %c) {
; CHECK-LABEL: fmls_s:
; CHECK: fmls z0.s, p0/m, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmls.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b,
                                                                  <vscale x 4 x float> %c)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmls_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x double> %c) {
; CHECK-LABEL: fmls_d:
; CHECK: fmls z0.d, p0/m, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmls.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b,
                                                                   <vscale x 2 x double> %c)
  ret <vscale x 2 x double> %out
}

;
; FMLS (Indexed)
;

define <vscale x 8 x half> @fmls_lane_h(<vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmls_lane_h:
; CHECK: fmls z0.h, z1.h, z2.h[3]
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmls.lane.nxv8f16(<vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b,
                                                                 <vscale x 8 x half> %c,
                                                                 i32 3)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmls_lane_s(<vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x float> %c) {
; CHECK-LABEL: fmls_lane_s:
; CHECK: fmls z0.s, z1.s, z2.s[2]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmls.lane.nxv4f32(<vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b,
                                                                  <vscale x 4 x float> %c,
                                                                  i32 2)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmls_lane_d(<vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x double> %c) {
; CHECK-LABEL: fmls_lane_d:
; CHECK: fmls z0.d, z1.d, z2.d[1]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmls.lane.nxv2f64(<vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b,
                                                                   <vscale x 2 x double> %c,
                                                                   i32 1)
  ret <vscale x 2 x double> %out
}

;
; FMSB
;

define <vscale x 8 x half> @fmsb_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmsb_h:
; CHECK: fmsb z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmsb.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b,
                                                                 <vscale x 8 x half> %c)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmsb_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x float> %c) {
; CHECK-LABEL: fmsb_s:
; CHECK: fmsb z0.s, p0/m, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmsb.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b,
                                                                  <vscale x 4 x float> %c)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmsb_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x double> %c) {
; CHECK-LABEL: fmsb_d:
; CHECK: fmsb z0.d, p0/m, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmsb.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b,
                                                                   <vscale x 2 x double> %c)
  ret <vscale x 2 x double> %out
}

;
; FMUL
;

define <vscale x 8 x half> @fmul_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fmul_h:
; CHECK: fmul z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmul_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fmul_s:
; CHECK: fmul z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmul.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmul_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fmul_d:
; CHECK: fmul z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMUL (Indexed)
;

define <vscale x 8 x half> @fmul_lane_h(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fmul_lane_h:
; CHECK: fmul z0.h, z0.h, z1.h[3]
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmul.lane.nxv8f16(<vscale x 8 x half> %a,
                                                                      <vscale x 8 x half> %b,
                                                                      i32 3)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmul_lane_s(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fmul_lane_s:
; CHECK: fmul z0.s, z0.s, z1.s[2]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmul.lane.nxv4f32(<vscale x 4 x float> %a,
                                                                       <vscale x 4 x float> %b,
                                                                       i32 2)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmul_lane_d(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fmul_lane_d:
; CHECK: fmul z0.d, z0.d, z1.d[1]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmul.lane.nxv2f64(<vscale x 2 x double> %a,
                                                                        <vscale x 2 x double> %b,
                                                                        i32 1)
  ret <vscale x 2 x double> %out
}

;
; FMULX
;

define <vscale x 8 x half> @fmulx_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fmulx_h:
; CHECK: fmulx z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmulx.nxv8f16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmulx_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fmulx_s:
; CHECK: fmulx z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmulx.nxv4f32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmulx_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fmulx_d:
; CHECK: fmulx z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmulx.nxv2f64(<vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FSCALE
;

define <vscale x 8 x half> @fscale_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: fscale_h:
; CHECK: fscale z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fscale.nxv8f16(<vscale x 8 x i1> %pg,
                                                                   <vscale x 8 x half> %a,
                                                                   <vscale x 8 x i16> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fscale_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: fscale_s:
; CHECK: fscale z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fscale.nxv4f32(<vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x float> %a,
                                                                    <vscale x 4 x i32> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fscale_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: fscale_d:
; CHECK: fscale z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fscale.nxv2f64(<vscale x 2 x i1> %pg,
                                                                     <vscale x 2 x double> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 2 x double> %out
}

;
; FNMAD
;

define <vscale x 8 x half> @fnmad_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fnmad_h:
; CHECK: fnmad z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fnmad.nxv8f16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b,
                                                                  <vscale x 8 x half> %c)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fnmad_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x float> %c) {
; CHECK-LABEL: fnmad_s:
; CHECK: fnmad z0.s, p0/m, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fnmad.nxv4f32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b,
                                                                   <vscale x 4 x float> %c)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fnmad_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x double> %c) {
; CHECK-LABEL: fnmad_d:
; CHECK: fnmad z0.d, p0/m, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fnmad.nxv2f64(<vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b,
                                                                    <vscale x 2 x double> %c)
  ret <vscale x 2 x double> %out
}

;
; FNMLA
;

define <vscale x 8 x half> @fnmla_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fnmla_h:
; CHECK: fnmla z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fnmla.nxv8f16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b,
                                                                  <vscale x 8 x half> %c)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fnmla_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x float> %c) {
; CHECK-LABEL: fnmla_s:
; CHECK: fnmla z0.s, p0/m, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fnmla.nxv4f32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b,
                                                                   <vscale x 4 x float> %c)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fnmla_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x double> %c) {
; CHECK-LABEL: fnmla_d:
; CHECK: fnmla z0.d, p0/m, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fnmla.nxv2f64(<vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b,
                                                                    <vscale x 2 x double> %c)
  ret <vscale x 2 x double> %out
}

;
; FNMLS
;

define <vscale x 8 x half> @fnmls_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fnmls_h:
; CHECK: fnmls z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fnmls.nxv8f16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b,
                                                                  <vscale x 8 x half> %c)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fnmls_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x float> %c) {
; CHECK-LABEL: fnmls_s:
; CHECK: fnmls z0.s, p0/m, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fnmls.nxv4f32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b,
                                                                   <vscale x 4 x float> %c)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fnmls_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x double> %c) {
; CHECK-LABEL: fnmls_d:
; CHECK: fnmls z0.d, p0/m, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fnmls.nxv2f64(<vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b,
                                                                    <vscale x 2 x double> %c)
  ret <vscale x 2 x double> %out
}

;
; FNMSB
;

define <vscale x 8 x half> @fnmsb_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fnmsb_h:
; CHECK: fnmsb z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fnmsb.nxv8f16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b,
                                                                  <vscale x 8 x half> %c)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fnmsb_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x float> %c) {
; CHECK-LABEL: fnmsb_s:
; CHECK: fnmsb z0.s, p0/m, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fnmsb.nxv4f32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b,
                                                                   <vscale x 4 x float> %c)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fnmsb_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x double> %c) {
; CHECK-LABEL: fnmsb_d:
; CHECK: fnmsb z0.d, p0/m, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fnmsb.nxv2f64(<vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b,
                                                                    <vscale x 2 x double> %c)
  ret <vscale x 2 x double> %out
}

;
; FSUB
;

define <vscale x 8 x half> @fsub_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fsub_h:
; CHECK: fsub z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fsub.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fsub_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fsub_s:
; CHECK: fsub z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fsub.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fsub_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fsub_d:
; CHECK: fsub z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fsub.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FSUBR
;

define <vscale x 8 x half> @fsubr_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fsubr_h:
; CHECK: fsubr z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fsubr.nxv8f16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fsubr_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fsubr_s:
; CHECK: fsubr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fsubr.nxv4f32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fsubr_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fsubr_d:
; CHECK: fsubr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fsubr.nxv2f64(<vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FTMAD
;

define <vscale x 8 x half> @ftmad_h(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: ftmad_h:
; CHECK: ftmad z0.h, z0.h, z1.h, #0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.ftmad.x.nxv8f16(<vscale x 8 x half> %a,
                                                                    <vscale x 8 x half> %b,
                                                                    i32 0)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @ftmad_s(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: ftmad_s:
; CHECK: ftmad z0.s, z0.s, z1.s, #0
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.ftmad.x.nxv4f32(<vscale x 4 x float> %a,
                                                                     <vscale x 4 x float> %b,
                                                                     i32 0)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @ftmad_d(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: ftmad_d:
; CHECK: ftmad z0.d, z0.d, z1.d, #7
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.ftmad.x.nxv2f64(<vscale x 2 x double> %a,
                                                                      <vscale x 2 x double> %b,
                                                                      i32 7)
  ret <vscale x 2 x double> %out
}

;
; FTSMUL
;

define <vscale x 8 x half> @ftsmul_h(<vscale x 8 x half> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: ftsmul_h:
; CHECK: ftsmul z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.ftsmul.x.nxv8f16(<vscale x 8 x half> %a,
                                                                     <vscale x 8 x i16> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @ftsmul_s(<vscale x 4 x float> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: ftsmul_s:
; CHECK: ftsmul z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.ftsmul.x.nxv4f32(<vscale x 4 x float> %a,
                                                                      <vscale x 4 x i32> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @ftsmul_d(<vscale x 2 x double> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: ftsmul_d:
; CHECK: ftsmul z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.ftsmul.x.nxv2f64(<vscale x 2 x double> %a,
                                                                       <vscale x 2 x i64> %b)
  ret <vscale x 2 x double> %out
}

;
; FTSSEL
;

define <vscale x 8 x half> @ftssel_h(<vscale x 8 x half> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: ftssel_h:
; CHECK: ftssel z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.ftssel.x.nxv8f16(<vscale x 8 x half> %a,
                                                                     <vscale x 8 x i16> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @ftssel_s(<vscale x 4 x float> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: ftssel_s:
; CHECK: ftssel z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.ftssel.x.nxv4f32(<vscale x 4 x float> %a,
                                                                      <vscale x 4 x i32> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @ftssel_d(<vscale x 2 x double> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: ftssel_d:
; CHECK: ftssel z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.ftssel.x.nxv2f64(<vscale x 2 x double> %a,
                                                                       <vscale x 2 x i64> %b)
  ret <vscale x 2 x double> %out
}

declare <vscale x 8 x half> @llvm.aarch64.sve.fabd.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fabd.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fabd.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fadd.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fcadd.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>, i32)
declare <vscale x 4 x float> @llvm.aarch64.sve.fcadd.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>, i32)
declare <vscale x 2 x double> @llvm.aarch64.sve.fcadd.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>, i32)

declare <vscale x 8 x half> @llvm.aarch64.sve.fcmla.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>, i32)
declare <vscale x 4 x float> @llvm.aarch64.sve.fcmla.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, i32)
declare <vscale x 2 x double> @llvm.aarch64.sve.fcmla.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, i32)

declare <vscale x 8 x half> @llvm.aarch64.sve.fcmla.lane.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>, i32, i32)
declare <vscale x 4 x float> @llvm.aarch64.sve.fcmla.lane.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, i32, i32)

declare <vscale x 8 x half> @llvm.aarch64.sve.fdiv.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fdiv.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fdiv.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fdivr.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fdivr.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fdivr.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmad.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmad.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmad.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmax.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmax.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmax.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmaxnm.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmaxnm.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmaxnm.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmin.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmin.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmin.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fminnm.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fminnm.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fminnm.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmla.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmla.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmla.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmla.lane.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>, i32)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmla.lane.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, i32)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmla.lane.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, i32)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmls.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmls.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmls.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmls.lane.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>, i32)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmls.lane.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, i32)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmls.lane.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, i32)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmsb.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmsb.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmsb.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmul.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmul.lane.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, i32)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmul.lane.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, i32)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmul.lane.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, i32)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmulx.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmulx.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmulx.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fnmad.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fnmad.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fnmad.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fnmla.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fnmla.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fnmla.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fnmls.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fnmls.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fnmls.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fnmsb.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fnmsb.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fnmsb.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fscale.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x i16>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fscale.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x i32>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fscale.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x i64>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fsub.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fsub.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fsub.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fsubr.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fsubr.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fsubr.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.ftmad.x.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, i32)
declare <vscale x 4 x float> @llvm.aarch64.sve.ftmad.x.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, i32)
declare <vscale x 2 x double> @llvm.aarch64.sve.ftmad.x.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, i32)

declare <vscale x 8 x half> @llvm.aarch64.sve.ftsmul.x.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i16>)
declare <vscale x 4 x float> @llvm.aarch64.sve.ftsmul.x.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i32>)
declare <vscale x 2 x double> @llvm.aarch64.sve.ftsmul.x.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i64>)

declare <vscale x 8 x half> @llvm.aarch64.sve.ftssel.x.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i16>)
declare <vscale x 4 x float> @llvm.aarch64.sve.ftssel.x.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i32>)
declare <vscale x 2 x double> @llvm.aarch64.sve.ftssel.x.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i64>)
