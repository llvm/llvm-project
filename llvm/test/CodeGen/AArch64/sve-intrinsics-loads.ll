; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; LDNT1B
;

define <vscale x 16 x i8> @ldnt1b_i8(<vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: ldnt1b_i8:
; CHECK: ldnt1b { z0.b }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ldnt1.nxv16i8(<vscale x 16 x i1> %pred,
                                                                 <vscale x 16 x i8>* %addr)
  ret <vscale x 16 x i8> %res
}

;
; LDNT1H
;

define <vscale x 8 x i16> @ldnt1h_i16(<vscale x 8 x i1> %pred, <vscale x 8 x i16>* %addr) {
; CHECK-LABEL: ldnt1h_i16:
; CHECK: ldnt1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.ldnt1.nxv8i16(<vscale x 8 x i1> %pred,
                                                                 <vscale x 8 x i16>* %addr)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x half> @ldnt1h_f16(<vscale x 8 x i1> %pred, <vscale x 8 x half>* %addr) {
; CHECK-LABEL: ldnt1h_f16:
; CHECK: ldnt1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x half> @llvm.aarch64.sve.ldnt1.nxv8f16(<vscale x 8 x i1> %pred,
                                                                  <vscale x 8 x half>* %addr)
  ret <vscale x 8 x half> %res
}

;
; LDNT1W
;

define <vscale x 4 x i32> @ldnt1w_i32(<vscale x 4 x i1> %pred, <vscale x 4 x i32>* %addr) {
; CHECK-LABEL: ldnt1w_i32:
; CHECK: ldnt1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.nxv4i32(<vscale x 4 x i1> %pred,
                                                                 <vscale x 4 x i32>* %addr)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x float> @ldnt1w_f32(<vscale x 4 x i1> %pred, <vscale x 4 x float>* %addr) {
; CHECK-LABEL: ldnt1w_f32:
; CHECK: ldnt1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.nxv4f32(<vscale x 4 x i1> %pred,
                                                                   <vscale x 4 x float>* %addr)
  ret <vscale x 4 x float> %res
}

;
; LDNT1D
;

define <vscale x 2 x i64> @ldnt1d_i64(<vscale x 2 x i1> %pred, <vscale x 2 x i64>* %addr) {
; CHECK-LABEL: ldnt1d_i64:
; CHECK: ldnt1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1> %pred,
                                                                 <vscale x 2 x i64>* %addr)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x double> @ldnt1d_f64(<vscale x 2 x i1> %pred, <vscale x 2 x double>* %addr) {
; CHECK-LABEL: ldnt1d_f64:
; CHECK: ldnt1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.nxv2f64(<vscale x 2 x i1> %pred,
                                                                    <vscale x 2 x double>* %addr)
  ret <vscale x 2 x double> %res
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.ldnt1.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>*)
declare <vscale x 8 x i16> @llvm.aarch64.sve.ldnt1.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>*)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>*)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>*)
declare <vscale x 8 x half> @llvm.aarch64.sve.ldnt1.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>*)
declare <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>*)
declare <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>*)
