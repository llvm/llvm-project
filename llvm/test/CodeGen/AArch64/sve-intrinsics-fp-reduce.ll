; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve -asm-verbose=0 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; FADDA
;

define half @fadda_f16(<vscale x 8 x i1> %pg, half %init, <vscale x 8 x half> %a) {
; CHECK-LABEL: fadda_f16:
; CHECK: fadda h0, p0, h0, z1.h
; CHECK-NEXT: ret
  %res = call half @llvm.aarch64.sve.fadda.nxv8f16(<vscale x 8 x i1> %pg,
                                                   half %init,
                                                   <vscale x 8 x half> %a)
  ret half %res
}

define float @fadda_f32(<vscale x 4 x i1> %pg, float %init, <vscale x 4 x float> %a) {
; CHECK-LABEL: fadda_f32:
; CHECK: fadda s0, p0, s0, z1.s
; CHECK-NEXT: ret
  %res = call float @llvm.aarch64.sve.fadda.nxv4f32(<vscale x 4 x i1> %pg,
                                                    float %init,
                                                    <vscale x 4 x float> %a)
  ret float %res
}

define double @fadda_f64(<vscale x 2 x i1> %pg, double %init, <vscale x 2 x double> %a) {
; CHECK-LABEL: fadda_f64:
; CHECK: fadda d0, p0, d0, z1.d
; CHECK-NEXT: ret
  %res = call double @llvm.aarch64.sve.fadda.nxv2f64(<vscale x 2 x i1> %pg,
                                                     double %init,
                                                     <vscale x 2 x double> %a)
  ret double %res
}

;
; FADDV
;

define half @faddv_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a) {
; CHECK-LABEL: faddv_f16:
; CHECK: faddv h0, p0, z0.h
; CHECK-NEXT: ret
  %res = call half @llvm.aarch64.sve.faddv.nxv8f16(<vscale x 8 x i1> %pg,
                                                   <vscale x 8 x half> %a)
  ret half %res
}

define float @faddv_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a) {
; CHECK-LABEL: faddv_f32:
; CHECK: faddv s0, p0, z0.s
; CHECK-NEXT: ret
  %res = call float @llvm.aarch64.sve.faddv.nxv4f32(<vscale x 4 x i1> %pg,
                                                    <vscale x 4 x float> %a)
  ret float %res
}

define double @faddv_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a) {
; CHECK-LABEL: faddv_f64:
; CHECK: faddv d0, p0, z0.d
; CHECK-NEXT: ret
  %res = call double @llvm.aarch64.sve.faddv.nxv2f64(<vscale x 2 x i1> %pg,
                                                     <vscale x 2 x double> %a)
  ret double %res
}

;
; FMAXNMV
;

define half @fmaxnmv_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a) {
; CHECK-LABEL: fmaxnmv_f16:
; CHECK: fmaxnmv h0, p0, z0.h
; CHECK-NEXT: ret
  %res = call half @llvm.aarch64.sve.fmaxnmv.nxv8f16(<vscale x 8 x i1> %pg,
                                                     <vscale x 8 x half> %a)
  ret half %res
}

define float @fmaxnmv_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a) {
; CHECK-LABEL: fmaxnmv_f32:
; CHECK: fmaxnmv s0, p0, z0.s
; CHECK-NEXT: ret
  %res = call float @llvm.aarch64.sve.fmaxnmv.nxv4f32(<vscale x 4 x i1> %pg,
                                                      <vscale x 4 x float> %a)
  ret float %res
}

define double @fmaxnmv_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a) {
; CHECK-LABEL: fmaxnmv_f64:
; CHECK: fmaxnmv d0, p0, z0.d
; CHECK-NEXT: ret
  %res = call double @llvm.aarch64.sve.fmaxnmv.nxv2f64(<vscale x 2 x i1> %pg,
                                                       <vscale x 2 x double> %a)
  ret double %res
}

;
; FMAXV
;

define half @fmaxv_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a) {
; CHECK-LABEL: fmaxv_f16:
; CHECK: fmaxv h0, p0, z0.h
; CHECK-NEXT: ret
  %res = call half @llvm.aarch64.sve.fmaxv.nxv8f16(<vscale x 8 x i1> %pg,
                                                   <vscale x 8 x half> %a)
  ret half %res
}

define float @fmaxv_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a) {
; CHECK-LABEL: fmaxv_f32:
; CHECK: fmaxv s0, p0, z0.s
; CHECK-NEXT: ret
  %res = call float @llvm.aarch64.sve.fmaxv.nxv4f32(<vscale x 4 x i1> %pg,
                                                    <vscale x 4 x float> %a)
  ret float %res
}

define double @fmaxv_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a) {
; CHECK-LABEL: fmaxv_f64:
; CHECK: fmaxv d0, p0, z0.d
; CHECK-NEXT: ret
  %res = call double @llvm.aarch64.sve.fmaxv.nxv2f64(<vscale x 2 x i1> %pg,
                                                     <vscale x 2 x double> %a)
  ret double %res
}

;
; FMINNMV
;

define half @fminnmv_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a) {
; CHECK-LABEL: fminnmv_f16:
; CHECK: fminnmv h0, p0, z0.h
; CHECK-NEXT: ret
  %res = call half @llvm.aarch64.sve.fminnmv.nxv8f16(<vscale x 8 x i1> %pg,
                                                     <vscale x 8 x half> %a)
  ret half %res
}

define float @fminnmv_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a) {
; CHECK-LABEL: fminnmv_f32:
; CHECK: fminnmv s0, p0, z0.s
; CHECK-NEXT: ret
  %res = call float @llvm.aarch64.sve.fminnmv.nxv4f32(<vscale x 4 x i1> %pg,
                                                      <vscale x 4 x float> %a)
  ret float %res
}

define double @fminnmv_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a) {
; CHECK-LABEL: fminnmv_f64:
; CHECK: fminnmv d0, p0, z0.d
; CHECK-NEXT: ret
  %res = call double @llvm.aarch64.sve.fminnmv.nxv2f64(<vscale x 2 x i1> %pg,
                                                       <vscale x 2 x double> %a)
  ret double %res
}

;
; FMINV
;

define half @fminv_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a) {
; CHECK-LABEL: fminv_f16:
; CHECK: fminv h0, p0, z0.h
; CHECK-NEXT: ret
  %res = call half @llvm.aarch64.sve.fminv.nxv8f16(<vscale x 8 x i1> %pg,
                                                   <vscale x 8 x half> %a)
  ret half %res
}

define float @fminv_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a) {
; CHECK-LABEL: fminv_f32:
; CHECK: fminv s0, p0, z0.s
; CHECK-NEXT: ret
  %res = call float @llvm.aarch64.sve.fminv.nxv4f32(<vscale x 4 x i1> %pg,
                                                    <vscale x 4 x float> %a)
  ret float %res
}

define double @fminv_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a) {
; CHECK-LABEL: fminv_f64:
; CHECK: fminv d0, p0, z0.d
; CHECK-NEXT: ret
  %res = call double @llvm.aarch64.sve.fminv.nxv2f64(<vscale x 2 x i1> %pg,
                                                     <vscale x 2 x double> %a)
  ret double %res
}

declare half @llvm.aarch64.sve.fadda.nxv8f16(<vscale x 8 x i1>, half, <vscale x 8 x half>)
declare float @llvm.aarch64.sve.fadda.nxv4f32(<vscale x 4 x i1>, float, <vscale x 4 x float>)
declare double @llvm.aarch64.sve.fadda.nxv2f64(<vscale x 2 x i1>, double, <vscale x 2 x double>)

declare half @llvm.aarch64.sve.faddv.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>)
declare float @llvm.aarch64.sve.faddv.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>)
declare double @llvm.aarch64.sve.faddv.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>)

declare half @llvm.aarch64.sve.fmaxnmv.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>)
declare float @llvm.aarch64.sve.fmaxnmv.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>)
declare double @llvm.aarch64.sve.fmaxnmv.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>)

declare half @llvm.aarch64.sve.fmaxv.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>)
declare float @llvm.aarch64.sve.fmaxv.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>)
declare double @llvm.aarch64.sve.fmaxv.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>)

declare half @llvm.aarch64.sve.fminnmv.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>)
declare float @llvm.aarch64.sve.fminnmv.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>)
declare double @llvm.aarch64.sve.fminnmv.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>)

declare half @llvm.aarch64.sve.fminv.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>)
declare float @llvm.aarch64.sve.fminv.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>)
declare double @llvm.aarch64.sve.fminv.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>)
