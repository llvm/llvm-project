; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; ADDP
;

define <vscale x 16 x i8> @addp_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: addp_i8:
; CHECK: addp z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.addp.nxv16i8(<vscale x 16 x i1> %pg,
                                                                <vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @addp_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: addp_i16:
; CHECK: addp z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.addp.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @addp_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: addp_i32:
; CHECK: addp z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.addp.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @addp_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: addp_i64:
; CHECK: addp z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.addp.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

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

;
; SMAXP
;

define <vscale x 16 x i8> @smaxp_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: smaxp_i8:
; CHECK: smaxp z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.smaxp.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @smaxp_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: smaxp_i16:
; CHECK: smaxp z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.smaxp.nxv8i16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @smaxp_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: smaxp_i32:
; CHECK: smaxp z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.smaxp.nxv4i32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @smaxp_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: smaxp_i64:
; CHECK: smaxp z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.smaxp.nxv2i64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; SMINP
;

define <vscale x 16 x i8> @sminp_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sminp_i8:
; CHECK: sminp z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sminp.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sminp_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sminp_i16:
; CHECK: sminp z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sminp.nxv8i16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sminp_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sminp_i32:
; CHECK: sminp z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sminp.nxv4i32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sminp_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sminp_i64:
; CHECK: sminp z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sminp.nxv2i64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; UMINP
;

define <vscale x 16 x i8> @uminp_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uminp_i8:
; CHECK: uminp z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uminp.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uminp_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uminp_i16:
; CHECK: uminp z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uminp.nxv8i16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uminp_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uminp_i32:
; CHECK: uminp z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uminp.nxv4i32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uminp_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uminp_i64:
; CHECK: uminp z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uminp.nxv2i64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; UMAXP
;

define <vscale x 16 x i8> @umaxp_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: umaxp_i8:
; CHECK: umaxp z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.umaxp.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @umaxp_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: umaxp_i16:
; CHECK: umaxp z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.umaxp.nxv8i16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @umaxp_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: umaxp_i32:
; CHECK: umaxp z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.umaxp.nxv4i32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @umaxp_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: umaxp_i64:
; CHECK: umaxp z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.umaxp.nxv2i64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.addp.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.addp.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.addp.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.addp.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

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

declare <vscale x 16 x i8> @llvm.aarch64.sve.smaxp.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.smaxp.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smaxp.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smaxp.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sminp.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sminp.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sminp.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sminp.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.umaxp.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.umaxp.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umaxp.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umaxp.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.uminp.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uminp.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uminp.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uminp.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
