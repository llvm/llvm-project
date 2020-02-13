; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; DUP
;

define <vscale x 16 x i8> @dup_i8(<vscale x 16 x i8> %a, <vscale x 16 x i1> %pg, i8 %b) {
; CHECK-LABEL: dup_i8:
; CHECK: mov z0.b, p0/m, w0
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> %a,
                                                               <vscale x 16 x i1> %pg,
                                                               i8 %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @dup_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, i16 %b) {
; CHECK-LABEL: dup_i16:
; CHECK: mov z0.h, p0/m, w0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> %a,
                                                               <vscale x 8 x i1> %pg,
                                                               i16 %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @dup_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, i32 %b) {
; CHECK-LABEL: dup_i32:
; CHECK: mov z0.s, p0/m, w0
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32> %a,
                                                               <vscale x 4 x i1> %pg,
                                                               i32 %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @dup_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, i64 %b) {
; CHECK-LABEL: dup_i64:
; CHECK: mov z0.d, p0/m, x0
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> %a,
                                                               <vscale x 2 x i1> %pg,
                                                               i64 %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @dup_f16(<vscale x 8 x half> %a, <vscale x 8 x i1> %pg, half %b) {
; CHECK-LABEL: dup_f16:
; CHECK: mov z0.h, p0/m, h1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.dup.nxv8f16(<vscale x 8 x half> %a,
                                                                <vscale x 8 x i1> %pg,
                                                                half %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @dup_f32(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, float %b) {
; CHECK-LABEL: dup_f32:
; CHECK: mov z0.s, p0/m, s1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.dup.nxv4f32(<vscale x 4 x float> %a,
                                                                 <vscale x 4 x i1> %pg,
                                                                 float %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @dup_f64(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, double %b) {
; CHECK-LABEL: dup_f64:
; CHECK: mov z0.d, p0/m, d1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.dup.nxv2f64(<vscale x 2 x double> %a,
                                                                  <vscale x 2 x i1> %pg,
                                                                  double %b)
  ret <vscale x 2 x double> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, i8)
declare <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, i16)
declare <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64)
declare <vscale x 8 x half> @llvm.aarch64.sve.dup.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, half)
declare <vscale x 4 x float> @llvm.aarch64.sve.dup.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, float)
declare <vscale x 2 x double> @llvm.aarch64.sve.dup.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double)
