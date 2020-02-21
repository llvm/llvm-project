; RUN: llc -mtriple=aarch64-linux-gnu -mattr=sve < %s | FileCheck %s

;
; FADD
;

define <vscale x 4 x float> @fadd_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fadd_s:
; CHECK:      movprfx z0.s, p0/z, z0.s
; CHECK-NEXT: fadd z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> zeroinitializer
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1> %pg,
                                                             <vscale x 4 x float> %a_z,
                                                             <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fadd_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fadd_d:
; CHECK:      movprfx z0.d, p0/z, z0.d
; CHECK-NEXT: fadd z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %a_z = select <vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> zeroinitializer
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fadd.nxv2f64(<vscale x 2 x i1> %pg,
                                                              <vscale x 2 x double> %a_z,
                                                              <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMAX
;

define <vscale x 4 x float> @fmax_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fmax_s:
; CHECK:      movprfx z0.s, p0/z, z0.s
; CHECK-NEXT: fmax z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> zeroinitializer
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmax.nxv4f32(<vscale x 4 x i1> %pg,
                                                             <vscale x 4 x float> %a_z,
                                                             <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmax_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fmax_d:
; CHECK:      movprfx z0.d, p0/z, z0.d
; CHECK-NEXT: fmax z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %a_z = select <vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> zeroinitializer
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmax.nxv2f64(<vscale x 2 x i1> %pg,
                                                              <vscale x 2 x double> %a_z,
                                                              <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMAXNM
;

define <vscale x 4 x float> @fmaxnm_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fmaxnm_s:
; CHECK:      movprfx z0.s, p0/z, z0.s
; CHECK-NEXT: fmaxnm z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> zeroinitializer
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmaxnm.nxv4f32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x float> %a_z,
                                                               <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmaxnm_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fmaxnm_d:
; CHECK:      movprfx z0.d, p0/z, z0.d
; CHECK-NEXT: fmaxnm z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %a_z = select <vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> zeroinitializer
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmaxnm.nxv2f64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x double> %a_z,
                                                                <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMIN
;

define <vscale x 4 x float> @fmin_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fmin_s:
; CHECK:      movprfx z0.s, p0/z, z0.s
; CHECK-NEXT: fmin z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> zeroinitializer
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmin.nxv4f32(<vscale x 4 x i1> %pg,
                                                             <vscale x 4 x float> %a_z,
                                                             <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmin_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fmin_d:
; CHECK:      movprfx z0.d, p0/z, z0.d
; CHECK-NEXT: fmin z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %a_z = select <vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> zeroinitializer
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmin.nxv2f64(<vscale x 2 x i1> %pg,
                                                              <vscale x 2 x double> %a_z,
                                                              <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMINNM
;

define <vscale x 4 x float> @fminnm_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fminnm_s:
; CHECK:      movprfx z0.s, p0/z, z0.s
; CHECK-NEXT: fminnm z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> zeroinitializer
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fminnm.nxv4f32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x float> %a_z,
                                                               <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fminnm_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fminnm_d:
; CHECK:      movprfx z0.d, p0/z, z0.d
; CHECK-NEXT: fminnm z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %a_z = select <vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> zeroinitializer
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fminnm.nxv2f64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x double> %a_z,
                                                                <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMUL
;

define <vscale x 4 x float> @fmul_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fmul_s:
; CHECK:      movprfx z0.s, p0/z, z0.s
; CHECK-NEXT: fmul z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> zeroinitializer
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmul.nxv4f32(<vscale x 4 x i1> %pg,
                                                             <vscale x 4 x float> %a_z,
                                                             <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmul_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fmul_d:
; CHECK:      movprfx z0.d, p0/z, z0.d
; CHECK-NEXT: fmul z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %a_z = select <vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> zeroinitializer
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1> %pg,
                                                              <vscale x 2 x double> %a_z,
                                                              <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FSUB
;

define <vscale x 4 x float> @fsub_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fsub_s:
; CHECK:      movprfx z0.s, p0/z, z0.s
; CHECK-NEXT: fsub z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> zeroinitializer
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fsub.nxv4f32(<vscale x 4 x i1> %pg,
                                                             <vscale x 4 x float> %a_z,
                                                             <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fsub_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fsub_d:
; CHECK:      movprfx z0.d, p0/z, z0.d
; CHECK-NEXT: fsub z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %a_z = select <vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> zeroinitializer
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fsub.nxv2f64(<vscale x 2 x i1> %pg,
                                                              <vscale x 2 x double> %a_z,
                                                              <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FSUBR
;

define <vscale x 4 x float> @fsubr_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fsubr_s:
; CHECK:      movprfx z0.s, p0/z, z0.s
; CHECK-NEXT: fsubr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> zeroinitializer
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fsubr.nxv4f32(<vscale x 4 x i1> %pg,
                                                              <vscale x 4 x float> %a_z,
                                                              <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fsubr_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fsubr_d:
; CHECK:      movprfx z0.d, p0/z, z0.d
; CHECK-NEXT: fsubr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %a_z = select <vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> zeroinitializer
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fsubr.nxv2f64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x double> %a_z,
                                                               <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

declare <vscale x 4 x float> @llvm.aarch64.sve.fabd.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fabd.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fadd.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 4 x float> @llvm.aarch64.sve.fdiv.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fdiv.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 4 x float> @llvm.aarch64.sve.fdivr.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fdivr.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 4 x float> @llvm.aarch64.sve.fmax.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmax.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 4 x float> @llvm.aarch64.sve.fmaxnm.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmaxnm.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 4 x float> @llvm.aarch64.sve.fmin.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmin.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 4 x float> @llvm.aarch64.sve.fminnm.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fminnm.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 4 x float> @llvm.aarch64.sve.fmul.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 4 x float> @llvm.aarch64.sve.fmulx.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmulx.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 4 x float> @llvm.aarch64.sve.fsub.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fsub.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 4 x float> @llvm.aarch64.sve.fsubr.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fsubr.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)
