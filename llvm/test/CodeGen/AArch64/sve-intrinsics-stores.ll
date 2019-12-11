; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; STNT1B
;

define void @stnt1b_i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: stnt1b_i8:
; CHECK: stnt1b { z0.b }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.stnt1.nxv16i8(<vscale x 16 x i8> %data,
                                            <vscale x 16 x i1> %pred,
                                            <vscale x 16 x i8>* %addr)
  ret void
}

;
; STNT1H
;

define void @stnt1h_i16(<vscale x 8 x i16> %data, <vscale x 8 x i1> %pred, <vscale x 8 x i16>* %addr) {
; CHECK-LABEL: stnt1h_i16:
; CHECK: stnt1h { z0.h }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.stnt1.nxv8i16(<vscale x 8 x i16> %data,
                                            <vscale x 8 x i1> %pred,
                                            <vscale x 8 x i16>* %addr)
  ret void
}

define void @stnt1h_f16(<vscale x 8 x half> %data, <vscale x 8 x i1> %pred, <vscale x 8 x half>* %addr) {
; CHECK-LABEL: stnt1h_f16:
; CHECK: stnt1h { z0.h }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.stnt1.nxv8f16(<vscale x 8 x half> %data,
                                            <vscale x 8 x i1> %pred,
                                            <vscale x 8 x half>* %addr)
  ret void
}

;
; STNT1W
;

define void @stnt1w_i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pred, <vscale x 4 x i32>* %addr) {
; CHECK-LABEL: stnt1w_i32:
; CHECK: stnt1w { z0.s }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.stnt1.nxv4i32(<vscale x 4 x i32> %data,
                                            <vscale x 4 x i1> %pred,
                                            <vscale x 4 x i32>* %addr)
  ret void
}

define void @stnt1w_f32(<vscale x 4 x float> %data, <vscale x 4 x i1> %pred, <vscale x 4 x float>* %addr) {
; CHECK-LABEL: stnt1w_f32:
; CHECK: stnt1w { z0.s }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.stnt1.nxv4f32(<vscale x 4 x float> %data,
                                            <vscale x 4 x i1> %pred,
                                            <vscale x 4 x float>* %addr)
  ret void
}

;
; STNT1D
;

define void @stnt1d_i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pred, <vscale x 2 x i64>* %addr) {
; CHECK-LABEL: stnt1d_i64:
; CHECK: stnt1d { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64> %data,
                                            <vscale x 2 x i1> %pred,
                                            <vscale x 2 x i64>* %addr)
  ret void
}

define void @stnt1d_f64(<vscale x 2 x double> %data, <vscale x 2 x i1> %pred, <vscale x 2 x double>* %addr) {
; CHECK-LABEL: stnt1d_f64:
; CHECK: stnt1d { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.stnt1.nxv2f64(<vscale x 2 x double> %data,
                                            <vscale x 2 x i1> %pred,
                                            <vscale x 2 x double>* %addr)
  ret void
}

declare void @llvm.aarch64.sve.stnt1.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, <vscale x 16 x i8>*)
declare void @llvm.aarch64.sve.stnt1.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x i16>*)
declare void @llvm.aarch64.sve.stnt1.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>*)
declare void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>*)
declare void @llvm.aarch64.sve.stnt1.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, <vscale x 8 x half>*)
declare void @llvm.aarch64.sve.stnt1.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x float>*)
declare void @llvm.aarch64.sve.stnt1.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x double>*)
