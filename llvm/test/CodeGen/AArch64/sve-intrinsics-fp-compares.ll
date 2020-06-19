; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

;
; FACGE
;

define <vscale x 8 x i1> @facge_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: facge_h:
; CHECK: facge p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.facge.nxv8f16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x half> %a,
                                                                <vscale x 8 x half> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @facge_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: facge_s:
; CHECK: facge p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.facge.nxv4f32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x float> %a,
                                                                <vscale x 4 x float> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @facge_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: facge_d:
; CHECK: facge p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.facge.nxv2f64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x double> %a,
                                                                <vscale x 2 x double> %b)
  ret <vscale x 2 x i1> %out
}

;
; FACGT
;

define <vscale x 8 x i1> @facgt_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: facgt_h:
; CHECK: facgt p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.facgt.nxv8f16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x half> %a,
                                                                <vscale x 8 x half> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @facgt_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: facgt_s:
; CHECK: facgt p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.facgt.nxv4f32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x float> %a,
                                                                <vscale x 4 x float> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @facgt_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: facgt_d:
; CHECK: facgt p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.facgt.nxv2f64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x double> %a,
                                                                <vscale x 2 x double> %b)
  ret <vscale x 2 x i1> %out
}

;
; FCMEQ
;

define <vscale x 8 x i1> @fcmeq_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fcmeq_h:
; CHECK: fcmeq p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.fcmpeq.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @fcmeq_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fcmeq_s:
; CHECK: fcmeq p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.fcmpeq.nxv4f32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x float> %a,
                                                                 <vscale x 4 x float> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @fcmeq_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fcmeq_d:
; CHECK: fcmeq p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.fcmpeq.nxv2f64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x double> %a,
                                                                 <vscale x 2 x double> %b)
  ret <vscale x 2 x i1> %out
}

;
; FCMGE
;

define <vscale x 8 x i1> @fcmge_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fcmge_h:
; CHECK: fcmge p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.fcmpge.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @fcmge_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fcmge_s:
; CHECK: fcmge p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.fcmpge.nxv4f32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x float> %a,
                                                                 <vscale x 4 x float> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @fcmge_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fcmge_d:
; CHECK: fcmge p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.fcmpge.nxv2f64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x double> %a,
                                                                 <vscale x 2 x double> %b)
  ret <vscale x 2 x i1> %out
}

;
; FCMGT
;

define <vscale x 8 x i1> @fcmgt_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fcmgt_h:
; CHECK: fcmgt p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.fcmpgt.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @fcmgt_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fcmgt_s:
; CHECK: fcmgt p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.fcmpgt.nxv4f32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x float> %a,
                                                                 <vscale x 4 x float> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @fcmgt_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fcmgt_d:
; CHECK: fcmgt p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.fcmpgt.nxv2f64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x double> %a,
                                                                 <vscale x 2 x double> %b)
  ret <vscale x 2 x i1> %out
}

;
; FCMNE
;

define <vscale x 8 x i1> @fcmne_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fcmne_h:
; CHECK: fcmne p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.fcmpne.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @fcmne_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fcmne_s:
; CHECK: fcmne p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.fcmpne.nxv4f32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x float> %a,
                                                                 <vscale x 4 x float> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @fcmne_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fcmne_d:
; CHECK: fcmne p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.fcmpne.nxv2f64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x double> %a,
                                                                 <vscale x 2 x double> %b)
  ret <vscale x 2 x i1> %out
}

;
; FCMPUO
;

define <vscale x 8 x i1> @fcmuo_h(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fcmuo_h:
; CHECK: fcmuo p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.fcmpuo.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @fcmuo_s(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fcmuo_s:
; CHECK: fcmuo p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.fcmpuo.nxv4f32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x float> %a,
                                                                 <vscale x 4 x float> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @fcmuo_d(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fcmuo_d:
; CHECK: fcmuo p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.fcmpuo.nxv2f64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x double> %a,
                                                                 <vscale x 2 x double> %b)
  ret <vscale x 2 x i1> %out
}

declare <vscale x 8 x i1> @llvm.aarch64.sve.facge.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.facge.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.facge.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x i1> @llvm.aarch64.sve.facgt.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.facgt.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.facgt.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x i1> @llvm.aarch64.sve.fcmpeq.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.fcmpeq.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.fcmpeq.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x i1> @llvm.aarch64.sve.fcmpge.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.fcmpge.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.fcmpge.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x i1> @llvm.aarch64.sve.fcmpgt.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.fcmpgt.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.fcmpgt.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x i1> @llvm.aarch64.sve.fcmpne.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.fcmpne.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.fcmpne.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x i1> @llvm.aarch64.sve.fcmpuo.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.fcmpuo.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.fcmpuo.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)
