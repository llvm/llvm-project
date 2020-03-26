; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; LD1B
;

define <vscale x 16 x i8> @ld1b_i8(<vscale x 16 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1b_i8:
; CHECK: ld1b { z0.b }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1> %pred,
                                                               i8* %addr)
  ret <vscale x 16 x i8> %res
}

;
; LD1H
;

define <vscale x 8 x i16> @ld1h_i16(<vscale x 8 x i1> %pred, i16* %addr) {
; CHECK-LABEL: ld1h_i16:
; CHECK: ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.ld1.nxv8i16(<vscale x 8 x i1> %pred,
                                                               i16* %addr)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x half> @ld1h_f16(<vscale x 8 x i1> %pred, half* %addr) {
; CHECK-LABEL: ld1h_f16:
; CHECK: ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x half> @llvm.aarch64.sve.ld1.nxv8f16(<vscale x 8 x i1> %pred,
                                                                half* %addr)
  ret <vscale x 8 x half> %res
}

;
; LD1W
;

define <vscale x 4 x i32> @ld1w_i32(<vscale x 4 x i1> %pred, i32* %addr) {
; CHECK-LABEL: ld1w_i32:
; CHECK: ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1.nxv4i32(<vscale x 4 x i1> %pred,
                                                               i32* %addr)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x float> @ld1w_f32(<vscale x 4 x i1> %pred, float* %addr) {
; CHECK-LABEL: ld1w_f32:
; CHECK: ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x float> @llvm.aarch64.sve.ld1.nxv4f32(<vscale x 4 x i1> %pred,
                                                                 float* %addr)
  ret <vscale x 4 x float> %res
}

;
; LD1D
;

define <vscale x 2 x i64> @ld1d_i64(<vscale x 2 x i1> %pred, i64* %addr) {
; CHECK-LABEL: ld1d_i64:
; CHECK: ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.nxv2i64(<vscale x 2 x i1> %pred,
                                                               i64* %addr)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x double> @ld1d_f64(<vscale x 2 x i1> %pred, double* %addr) {
; CHECK-LABEL: ld1d_f64:
; CHECK: ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.nxv2f64(<vscale x 2 x i1> %pred,
                                                                  double* %addr)
  ret <vscale x 2 x double> %res
}

;
; ST1B
;

define void @st1b_i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pred, i8* %addr) {
; CHECK-LABEL: st1b_i8:
; CHECK: st1b { z0.b }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.nxv16i8(<vscale x 16 x i8> %data,
                                          <vscale x 16 x i1> %pred,
                                          i8* %addr)
  ret void
}

;
; ST1H
;

define void @st1h_i16(<vscale x 8 x i16> %data, <vscale x 8 x i1> %pred, i16* %addr) {
; CHECK-LABEL: st1h_i16:
; CHECK: st1h { z0.h }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.nxv8i16(<vscale x 8 x i16> %data,
                                          <vscale x 8 x i1> %pred,
                                          i16* %addr)
  ret void
}

define void @st1h_f16(<vscale x 8 x half> %data, <vscale x 8 x i1> %pred, half* %addr) {
; CHECK-LABEL: st1h_f16:
; CHECK: st1h { z0.h }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.nxv8f16(<vscale x 8 x half> %data,
                                          <vscale x 8 x i1> %pred,
                                          half* %addr)
  ret void
}

;
; ST1W
;

define void @st1w_i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pred, i32* %addr) {
; CHECK-LABEL: st1w_i32:
; CHECK: st1w { z0.s }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.nxv4i32(<vscale x 4 x i32> %data,
                                          <vscale x 4 x i1> %pred,
                                          i32* %addr)
  ret void
}

define void @st1w_f32(<vscale x 4 x float> %data, <vscale x 4 x i1> %pred, float* %addr) {
; CHECK-LABEL: st1w_f32:
; CHECK: st1w { z0.s }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.nxv4f32(<vscale x 4 x float> %data,
                                          <vscale x 4 x i1> %pred,
                                          float* %addr)
  ret void
}

;
; ST1D
;

define void @st1d_i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pred, i64* %addr) {
; CHECK-LABEL: st1d_i64:
; CHECK: st1d { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.nxv2i64(<vscale x 2 x i64> %data,
                                          <vscale x 2 x i1> %pred,
                                          i64* %addr)
  ret void
}

define void @st1d_f64(<vscale x 2 x double> %data, <vscale x 2 x i1> %pred, double* %addr) {
; CHECK-LABEL: st1d_f64:
; CHECK: st1d { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.nxv2f64(<vscale x 2 x double> %data,
                                          <vscale x 2 x i1> %pred,
                                          double* %addr)
  ret void
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1>, i8*)
declare <vscale x 8 x i16> @llvm.aarch64.sve.ld1.nxv8i16(<vscale x 8 x i1>, i16*)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1.nxv4i32(<vscale x 4 x i1>, i32*)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1.nxv2i64(<vscale x 2 x i1>, i64*)
declare <vscale x 8 x half> @llvm.aarch64.sve.ld1.nxv8f16(<vscale x 8 x i1>, half*)
declare <vscale x 4 x float> @llvm.aarch64.sve.ld1.nxv4f32(<vscale x 4 x i1>, float*)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.nxv2f64(<vscale x 2 x i1>, double*)

declare void @llvm.aarch64.sve.st1.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, i8*)
declare void @llvm.aarch64.sve.st1.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, i16*)
declare void @llvm.aarch64.sve.st1.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32*)
declare void @llvm.aarch64.sve.st1.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64*)
declare void @llvm.aarch64.sve.st1.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, half*)
declare void @llvm.aarch64.sve.st1.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, float*)
declare void @llvm.aarch64.sve.st1.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double*)
