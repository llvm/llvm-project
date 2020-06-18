; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

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
; FABS
;

define <vscale x 8 x half> @fabs_h(<vscale x 8 x half> %a, <vscale x 8 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: fabs_h:
; CHECK: fabs z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fabs.nxv8f16(<vscale x 8 x half> %a,
                                                                 <vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fabs_s(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: fabs_s:
; CHECK: fabs z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fabs.nxv4f32(<vscale x 4 x float> %a,
                                                                  <vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fabs_d(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: fabs_d:
; CHECK: fabs z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fabs.nxv2f64(<vscale x 2 x double> %a,
                                                                   <vscale x 2 x i1> %pg,
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
; FEXPA
;

define <vscale x 8 x half> @fexpa_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: fexpa_h:
; CHECK: fexpa z0.h, z0.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fexpa.x.nxv8f16(<vscale x 8 x i16> %a)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fexpa_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: fexpa_s:
; CHECK: fexpa z0.s, z0.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fexpa.x.nxv4f32(<vscale x 4 x i32> %a)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fexpa_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: fexpa_d:
; CHECK: fexpa z0.d, z0.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fexpa.x.nxv2f64(<vscale x 2 x i64> %a)
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
; FNEG
;

define <vscale x 8 x half> @fneg_h(<vscale x 8 x half> %a, <vscale x 8 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: fneg_h:
; CHECK: fneg z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fneg.nxv8f16(<vscale x 8 x half> %a,
                                                                 <vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fneg_s(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: fneg_s:
; CHECK: fneg z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fneg.nxv4f32(<vscale x 4 x float> %a,
                                                                  <vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fneg_d(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: fneg_d:
; CHECK: fneg z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fneg.nxv2f64(<vscale x 2 x double> %a,
                                                                   <vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %b)
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
; FRECPE
;

define <vscale x 8 x half> @frecpe_h(<vscale x 8 x half> %a) {
; CHECK-LABEL: frecpe_h:
; CHECK: frecpe z0.h, z0.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.frecpe.x.nxv8f16(<vscale x 8 x half> %a)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @frecpe_s(<vscale x 4 x float> %a) {
; CHECK-LABEL: frecpe_s:
; CHECK: frecpe z0.s, z0.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.frecpe.x.nxv4f32(<vscale x 4 x float> %a)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @frecpe_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a) {
; CHECK-LABEL: frecpe_d:
; CHECK: frecpe z0.d, z0.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.frecpe.x.nxv2f64(<vscale x 2 x double> %a)
  ret <vscale x 2 x double> %out
}

;
; FRECPX
;

define <vscale x 8 x half> @frecpx_h(<vscale x 8 x half> %a, <vscale x 8 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: frecpx_h:
; CHECK: frecpx z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.frecpx.nxv8f16(<vscale x 8 x half> %a,
                                                                  <vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @frecpx_s(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: frecpx_s:
; CHECK: frecpx z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.frecpx.nxv4f32(<vscale x 4 x float> %a,
                                                                   <vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @frecpx_d(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: frecpx_d:
; CHECK: frecpx z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.frecpx.nxv2f64(<vscale x 2 x double> %a,
                                                                    <vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FRINTA
;

define <vscale x 8 x half> @frinta_h(<vscale x 8 x half> %a, <vscale x 8 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: frinta_h:
; CHECK: frinta z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.frinta.nxv8f16(<vscale x 8 x half> %a,
                                                                   <vscale x 8 x i1> %pg,
                                                                   <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @frinta_s(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: frinta_s:
; CHECK: frinta z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.frinta.nxv4f32(<vscale x 4 x float> %a,
                                                                    <vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @frinta_d(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: frinta_d:
; CHECK: frinta z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.frinta.nxv2f64(<vscale x 2 x double> %a,
                                                                     <vscale x 2 x i1> %pg,
                                                                     <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FRINTI
;

define <vscale x 8 x half> @frinti_h(<vscale x 8 x half> %a, <vscale x 8 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: frinti_h:
; CHECK: frinti z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.frinti.nxv8f16(<vscale x 8 x half> %a,
                                                                   <vscale x 8 x i1> %pg,
                                                                   <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @frinti_s(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: frinti_s:
; CHECK: frinti z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.frinti.nxv4f32(<vscale x 4 x float> %a,
                                                                    <vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @frinti_d(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: frinti_d:
; CHECK: frinti z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.frinti.nxv2f64(<vscale x 2 x double> %a,
                                                                     <vscale x 2 x i1> %pg,
                                                                     <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FRINTM
;

define <vscale x 8 x half> @frintm_h(<vscale x 8 x half> %a, <vscale x 8 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: frintm_h:
; CHECK: frintm z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.frintm.nxv8f16(<vscale x 8 x half> %a,
                                                                   <vscale x 8 x i1> %pg,
                                                                   <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @frintm_s(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: frintm_s:
; CHECK: frintm z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.frintm.nxv4f32(<vscale x 4 x float> %a,
                                                                    <vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @frintm_d(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: frintm_d:
; CHECK: frintm z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.frintm.nxv2f64(<vscale x 2 x double> %a,
                                                                     <vscale x 2 x i1> %pg,
                                                                     <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FRINTN
;

define <vscale x 8 x half> @frintn_h(<vscale x 8 x half> %a, <vscale x 8 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: frintn_h:
; CHECK: frintn z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.frintn.nxv8f16(<vscale x 8 x half> %a,
                                                                   <vscale x 8 x i1> %pg,
                                                                   <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @frintn_s(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: frintn_s:
; CHECK: frintn z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.frintn.nxv4f32(<vscale x 4 x float> %a,
                                                                    <vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @frintn_d(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: frintn_d:
; CHECK: frintn z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.frintn.nxv2f64(<vscale x 2 x double> %a,
                                                                     <vscale x 2 x i1> %pg,
                                                                     <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FRINTP
;

define <vscale x 8 x half> @frintp_h(<vscale x 8 x half> %a, <vscale x 8 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: frintp_h:
; CHECK: frintp z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.frintp.nxv8f16(<vscale x 8 x half> %a,
                                                                   <vscale x 8 x i1> %pg,
                                                                   <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @frintp_s(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: frintp_s:
; CHECK: frintp z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.frintp.nxv4f32(<vscale x 4 x float> %a,
                                                                    <vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @frintp_d(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: frintp_d:
; CHECK: frintp z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.frintp.nxv2f64(<vscale x 2 x double> %a,
                                                                     <vscale x 2 x i1> %pg,
                                                                     <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FRINTX
;

define <vscale x 8 x half> @frintx_h(<vscale x 8 x half> %a, <vscale x 8 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: frintx_h:
; CHECK: frintx z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.frintx.nxv8f16(<vscale x 8 x half> %a,
                                                                   <vscale x 8 x i1> %pg,
                                                                   <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @frintx_s(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: frintx_s:
; CHECK: frintx z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.frintx.nxv4f32(<vscale x 4 x float> %a,
                                                                    <vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @frintx_d(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: frintx_d:
; CHECK: frintx z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.frintx.nxv2f64(<vscale x 2 x double> %a,
                                                                     <vscale x 2 x i1> %pg,
                                                                     <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FRINTZ
;

define <vscale x 8 x half> @frintz_h(<vscale x 8 x half> %a, <vscale x 8 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: frintz_h:
; CHECK: frintz z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.frintz.nxv8f16(<vscale x 8 x half> %a,
                                                                   <vscale x 8 x i1> %pg,
                                                                   <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @frintz_s(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: frintz_s:
; CHECK: frintz z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.frintz.nxv4f32(<vscale x 4 x float> %a,
                                                                    <vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @frintz_d(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: frintz_d:
; CHECK: frintz z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.frintz.nxv2f64(<vscale x 2 x double> %a,
                                                                     <vscale x 2 x i1> %pg,
                                                                     <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FRSQRTE
;

define <vscale x 8 x half> @frsqrte_h(<vscale x 8 x half> %a) {
; CHECK-LABEL: frsqrte_h:
; CHECK: frsqrte z0.h, z0.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.frsqrte.x.nxv8f16(<vscale x 8 x half> %a)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @frsqrte_s(<vscale x 4 x float> %a) {
; CHECK-LABEL: frsqrte_s:
; CHECK: frsqrte z0.s, z0.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.frsqrte.x.nxv4f32(<vscale x 4 x float> %a)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @frsqrte_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a) {
; CHECK-LABEL: frsqrte_d:
; CHECK: frsqrte z0.d, z0.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.frsqrte.x.nxv2f64(<vscale x 2 x double> %a)
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
; FSQRT
;

define <vscale x 8 x half> @fsqrt_h(<vscale x 8 x half> %a, <vscale x 8 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: fsqrt_h:
; CHECK: fsqrt z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fsqrt.nxv8f16(<vscale x 8 x half> %a,
                                                                  <vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fsqrt_s(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: fsqrt_s:
; CHECK: fsqrt z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fsqrt.nxv4f32(<vscale x 4 x float> %a,
                                                                   <vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fsqrt_d(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: fsqrt_d:
; CHECK: fsqrt z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fsqrt.nxv2f64(<vscale x 2 x double> %a,
                                                                    <vscale x 2 x i1> %pg,
                                                                    <vscale x 2 x double> %b)
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

declare <vscale x 8 x half> @llvm.aarch64.sve.fabs.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fabs.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fabs.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x double>)

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

declare <vscale x 8 x half> @llvm.aarch64.sve.fexpa.x.nxv8f16(<vscale x 8 x i16>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fexpa.x.nxv4f32(<vscale x 4 x i32>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fexpa.x.nxv2f64(<vscale x 2 x i64>)

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

declare <vscale x 8 x half> @llvm.aarch64.sve.fneg.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fneg.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fneg.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x double>)

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

declare <vscale x 8 x half> @llvm.aarch64.sve.frecpe.x.nxv8f16(<vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.frecpe.x.nxv4f32(<vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.frecpe.x.nxv2f64(<vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.frecpx.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.frecpx.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.frecpx.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.frinta.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.frinta.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.frinta.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.frinti.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.frinti.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.frinti.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.frintm.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.frintm.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.frintm.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.frintn.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.frintn.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.frintn.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.frintp.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.frintp.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.frintp.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.frintx.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.frintx.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.frintx.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.frintz.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.frintz.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.frintz.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.frsqrte.x.nxv8f16(<vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.frsqrte.x.nxv4f32(<vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.frsqrte.x.nxv2f64(<vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fscale.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x i16>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fscale.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x i32>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fscale.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x i64>)

declare <vscale x 8 x half> @llvm.aarch64.sve.fsqrt.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fsqrt.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fsqrt.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x double>)

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
