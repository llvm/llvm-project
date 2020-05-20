; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; LD1RQB
;

define <vscale x 16 x i8> @ld1rqb_i8(<vscale x 16 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1rqb_i8:
; CHECK: ld1rqb { z0.b }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1rq.nxv16i8(<vscale x 16 x i1> %pred, i8* %addr)
  ret <vscale x 16 x i8> %res
}

define <vscale x 16 x i8> @ld1rqb_i8_imm(<vscale x 16 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1rqb_i8_imm:
; CHECK: ld1rqb { z0.b }, p0/z, [x0, #16]
; CHECK-NEXT: ret
  %ptr = getelementptr inbounds i8, i8* %addr, i8 16
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1rq.nxv16i8(<vscale x 16 x i1> %pred, i8* %ptr)
  ret <vscale x 16 x i8> %res
}

define <vscale x 16 x i8> @ld1rqb_i8_imm_lower_bound(<vscale x 16 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1rqb_i8_imm_lower_bound:
; CHECK: ld1rqb { z0.b }, p0/z, [x0, #-128]
; CHECK-NEXT: ret
  %ptr = getelementptr inbounds i8, i8* %addr, i8 -128
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1rq.nxv16i8(<vscale x 16 x i1> %pred, i8* %ptr)
  ret <vscale x 16 x i8> %res
}

define <vscale x 16 x i8> @ld1rqb_i8_imm_upper_bound(<vscale x 16 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1rqb_i8_imm_upper_bound:
; CHECK: ld1rqb { z0.b }, p0/z, [x0, #112]
; CHECK-NEXT: ret
  %ptr = getelementptr inbounds i8, i8* %addr, i8 112
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1rq.nxv16i8(<vscale x 16 x i1> %pred, i8* %ptr)
  ret <vscale x 16 x i8> %res
}

define <vscale x 16 x i8> @ld1rqb_i8_imm_out_of_lower_bound(<vscale x 16 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1rqb_i8_imm_out_of_lower_bound:
; CHECK: sub x8, x0, #129
; CHECK-NEXT: ld1rqb { z0.b }, p0/z, [x8]
; CHECK-NEXT: ret
  %ptr = getelementptr inbounds i8, i8* %addr, i64 -129
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1rq.nxv16i8(<vscale x 16 x i1> %pred, i8* %ptr)
  ret <vscale x 16 x i8> %res
}

define <vscale x 16 x i8> @ld1rqb_i8_imm_out_of_upper_bound(<vscale x 16 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1rqb_i8_imm_out_of_upper_bound:
; CHECK: add x8, x0, #113
; CHECK-NEXT: ld1rqb { z0.b }, p0/z, [x8]
; CHECK-NEXT: ret
  %ptr = getelementptr inbounds i8, i8* %addr, i64 113
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1rq.nxv16i8(<vscale x 16 x i1> %pred, i8* %ptr)
  ret <vscale x 16 x i8> %res
}

;
; LD1RQH
;

define <vscale x 8 x i16> @ld1rqh_i16(<vscale x 8 x i1> %pred, i16* %addr) {
; CHECK-LABEL: ld1rqh_i16:
; CHECK: ld1rqh { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.ld1rq.nxv8i16(<vscale x 8 x i1> %pred, i16* %addr)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x half> @ld1rqh_f16(<vscale x 8 x i1> %pred, half* %addr) {
; CHECK-LABEL: ld1rqh_f16:
; CHECK: ld1rqh { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x half> @llvm.aarch64.sve.ld1rq.nxv8f16(<vscale x 8 x i1> %pred, half* %addr)
  ret <vscale x 8 x half> %res
}

define <vscale x 8 x i16> @ld1rqh_i16_imm(<vscale x 8 x i1> %pred, i16* %addr) {
; CHECK-LABEL: ld1rqh_i16_imm:
; CHECK: ld1rqh { z0.h }, p0/z, [x0, #-64]
; CHECK-NEXT: ret
  %ptr = getelementptr inbounds i16, i16* %addr, i16 -32
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.ld1rq.nxv8i16(<vscale x 8 x i1> %pred, i16* %ptr)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x half> @ld1rqh_f16_imm(<vscale x 8 x i1> %pred, half* %addr) {
; CHECK-LABEL: ld1rqh_f16_imm:
; CHECK: ld1rqh { z0.h }, p0/z, [x0, #-16]
; CHECK-NEXT: ret
  %ptr = getelementptr inbounds half, half* %addr, i16 -8
  %res = call <vscale x 8 x half> @llvm.aarch64.sve.ld1rq.nxv8f16(<vscale x 8 x i1> %pred, half* %ptr)
  ret <vscale x 8 x half> %res
}

;
; LD1RQW
;

define <vscale x 4 x i32> @ld1rqw_i32(<vscale x 4 x i1> %pred, i32* %addr) {
; CHECK-LABEL: ld1rqw_i32:
; CHECK: ld1rqw { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1rq.nxv4i32(<vscale x 4 x i1> %pred, i32* %addr)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x float> @ld1rqw_f32(<vscale x 4 x i1> %pred, float* %addr) {
; CHECK-LABEL: ld1rqw_f32:
; CHECK: ld1rqw { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x float> @llvm.aarch64.sve.ld1rq.nxv4f32(<vscale x 4 x i1> %pred, float* %addr)
  ret <vscale x 4 x float> %res
}

define <vscale x 4 x i32> @ld1rqw_i32_imm(<vscale x 4 x i1> %pred, i32* %addr) {
; CHECK-LABEL: ld1rqw_i32_imm:
; CHECK: ld1rqw { z0.s }, p0/z, [x0, #112]
; CHECK-NEXT: ret
  %ptr = getelementptr inbounds i32, i32* %addr, i32 28
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1rq.nxv4i32(<vscale x 4 x i1> %pred, i32* %ptr)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x float> @ld1rqw_f32_imm(<vscale x 4 x i1> %pred, float* %addr) {
; CHECK-LABEL: ld1rqw_f32_imm:
; CHECK: ld1rqw { z0.s }, p0/z, [x0, #32]
; CHECK-NEXT: ret
  %ptr = getelementptr inbounds float, float* %addr, i32 8
  %res = call <vscale x 4 x float> @llvm.aarch64.sve.ld1rq.nxv4f32(<vscale x 4 x i1> %pred, float* %ptr)
  ret <vscale x 4 x float> %res
}

;
; LD1RQD
;

define <vscale x 2 x i64> @ld1rqd_i64(<vscale x 2 x i1> %pred, i64* %addr) {
; CHECK-LABEL: ld1rqd_i64:
; CHECK: ld1rqd { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1rq.nxv2i64(<vscale x 2 x i1> %pred, i64* %addr)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x double> @ld1rqd_f64(<vscale x 2 x i1> %pred, double* %addr) {
; CHECK-LABEL: ld1rqd_f64:
; CHECK: ld1rqd { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x double> @llvm.aarch64.sve.ld1rq.nxv2f64(<vscale x 2 x i1> %pred, double* %addr)
  ret <vscale x 2 x double> %res
}

define <vscale x 2 x i64> @ld1rqd_i64_imm(<vscale x 2 x i1> %pred, i64* %addr) {
; CHECK-LABEL: ld1rqd_i64_imm:
; CHECK: ld1rqd { z0.d }, p0/z, [x0, #64]
; CHECK-NEXT: ret
  %ptr = getelementptr inbounds i64, i64* %addr, i64 8
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1rq.nxv2i64(<vscale x 2 x i1> %pred, i64* %ptr)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x double> @ld1rqd_f64_imm(<vscale x 2 x i1> %pred, double* %addr) {
; CHECK-LABEL: ld1rqd_f64_imm:
; CHECK: ld1rqd { z0.d }, p0/z, [x0, #-128]
; CHECK-NEXT: ret
  %ptr = getelementptr inbounds double, double* %addr, i64 -16
  %res = call <vscale x 2 x double> @llvm.aarch64.sve.ld1rq.nxv2f64(<vscale x 2 x i1> %pred, double* %ptr)
  ret <vscale x 2 x double> %res
}

;
; LDNT1B
;

define <vscale x 16 x i8> @ldnt1b_i8(<vscale x 16 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ldnt1b_i8:
; CHECK: ldnt1b { z0.b }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ldnt1.nxv16i8(<vscale x 16 x i1> %pred,
                                                                 i8* %addr)
  ret <vscale x 16 x i8> %res
}

;
; LDNT1H
;

define <vscale x 8 x i16> @ldnt1h_i16(<vscale x 8 x i1> %pred, i16* %addr) {
; CHECK-LABEL: ldnt1h_i16:
; CHECK: ldnt1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.ldnt1.nxv8i16(<vscale x 8 x i1> %pred,
                                                                 i16* %addr)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x half> @ldnt1h_f16(<vscale x 8 x i1> %pred, half* %addr) {
; CHECK-LABEL: ldnt1h_f16:
; CHECK: ldnt1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x half> @llvm.aarch64.sve.ldnt1.nxv8f16(<vscale x 8 x i1> %pred,
                                                                  half* %addr)
  ret <vscale x 8 x half> %res
}

;
; LDNT1W
;

define <vscale x 4 x i32> @ldnt1w_i32(<vscale x 4 x i1> %pred, i32* %addr) {
; CHECK-LABEL: ldnt1w_i32:
; CHECK: ldnt1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.nxv4i32(<vscale x 4 x i1> %pred,
                                                                 i32* %addr)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x float> @ldnt1w_f32(<vscale x 4 x i1> %pred, float* %addr) {
; CHECK-LABEL: ldnt1w_f32:
; CHECK: ldnt1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.nxv4f32(<vscale x 4 x i1> %pred,
                                                                   float* %addr)
  ret <vscale x 4 x float> %res
}

;
; LDNT1D
;

define <vscale x 2 x i64> @ldnt1d_i64(<vscale x 2 x i1> %pred, i64* %addr) {
; CHECK-LABEL: ldnt1d_i64:
; CHECK: ldnt1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1> %pred,
                                                                 i64* %addr)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x double> @ldnt1d_f64(<vscale x 2 x i1> %pred, double* %addr) {
; CHECK-LABEL: ldnt1d_f64:
; CHECK: ldnt1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.nxv2f64(<vscale x 2 x i1> %pred,
                                                                    double* %addr)
  ret <vscale x 2 x double> %res
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.ld1rq.nxv16i8(<vscale x 16 x i1>, i8*)
declare <vscale x 8 x i16> @llvm.aarch64.sve.ld1rq.nxv8i16(<vscale x 8 x i1>, i16*)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1rq.nxv4i32(<vscale x 4 x i1>, i32*)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1rq.nxv2i64(<vscale x 2 x i1>, i64*)
declare <vscale x 8 x half> @llvm.aarch64.sve.ld1rq.nxv8f16(<vscale x 8 x i1>, half*)
declare <vscale x 4 x float> @llvm.aarch64.sve.ld1rq.nxv4f32(<vscale x 4 x i1>, float*)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1rq.nxv2f64(<vscale x 2 x i1>, double*)

declare <vscale x 16 x i8> @llvm.aarch64.sve.ldnt1.nxv16i8(<vscale x 16 x i1>, i8*)
declare <vscale x 8 x i16> @llvm.aarch64.sve.ldnt1.nxv8i16(<vscale x 8 x i1>, i16*)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.nxv4i32(<vscale x 4 x i1>, i32*)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1>, i64*)
declare <vscale x 8 x half> @llvm.aarch64.sve.ldnt1.nxv8f16(<vscale x 8 x i1>, half*)
declare <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.nxv4f32(<vscale x 4 x i1>, float*)
declare <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.nxv2f64(<vscale x 2 x i1>, double*)
