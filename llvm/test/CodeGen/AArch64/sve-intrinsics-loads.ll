; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve,+bf16 -asm-verbose=0 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

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

define <vscale x 8 x bfloat> @ld1rqh_bf16(<vscale x 8 x i1> %pred, bfloat* %addr) {
; CHECK-LABEL: ld1rqh_bf16:
; CHECK: ld1rqh { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x bfloat> @llvm.aarch64.sve.ld1rq.nxv8bf16(<vscale x 8 x i1> %pred, bfloat* %addr)
  ret <vscale x 8 x bfloat> %res
}

define <vscale x 8 x bfloat> @ld1rqh_bf16_imm(<vscale x 8 x i1> %pred, bfloat* %addr) {
; CHECK-LABEL: ld1rqh_bf16_imm:
; CHECK: ld1rqh { z0.h }, p0/z, [x0, #-16]
; CHECK-NEXT: ret
  %ptr = getelementptr inbounds bfloat, bfloat* %addr, i16 -8
  %res = call <vscale x 8 x bfloat> @llvm.aarch64.sve.ld1rq.nxv8bf16(<vscale x 8 x i1> %pred, bfloat* %ptr)
  ret <vscale x 8 x bfloat> %res
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

define <vscale x 8 x bfloat> @ldnt1h_bf16(<vscale x 8 x i1> %pred, bfloat* %addr) {
; CHECK-LABEL: ldnt1h_bf16:
; CHECK: ldnt1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x bfloat> @llvm.aarch64.sve.ldnt1.nxv8bf16(<vscale x 8 x i1> %pred,
                                                                     bfloat* %addr)
  ret <vscale x 8 x bfloat> %res
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

;
; LD2B
;

define <vscale x 32 x i8> @ld2b_i8(<vscale x 16 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld2b_i8:
; CHECK: ld2b { z0.b, z1.b }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1.p0i8(<vscale x 16 x i1> %pred, i8* %addr)
  ret <vscale x 32 x i8> %res
}

;
; LD2H
;

define <vscale x 16 x i16> @ld2h_i16(<vscale x 8 x i1> %pred, i16* %addr) {
; CHECK-LABEL: ld2h_i16:
; CHECK: ld2h { z0.h, z1.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i16> @llvm.aarch64.sve.ld2.nxv16i16.nxv8i1.p0i16(<vscale x 8 x i1> %pred, i16* %addr)
  ret <vscale x 16 x i16> %res
}

define <vscale x 16 x half> @ld2h_f16(<vscale x 8 x i1> %pred, half* %addr) {
; CHECK-LABEL: ld2h_f16:
; CHECK: ld2h { z0.h, z1.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 16 x half> @llvm.aarch64.sve.ld2.nxv16f16.nxv8i1.p0f16(<vscale x 8 x i1> %pred, half* %addr)
  ret <vscale x 16 x half> %res
}

define <vscale x 16 x bfloat> @ld2h_bf16(<vscale x 8 x i1> %pred, bfloat* %addr) {
; CHECK-LABEL: ld2h_bf16:
; CHECK: ld2h { z0.h, z1.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 16 x bfloat> @llvm.aarch64.sve.ld2.nxv16bf16.nxv8i1.p0bf16(<vscale x 8 x i1> %pred, bfloat* %addr)
  ret <vscale x 16 x bfloat> %res
}

;
; LD2W
;

define <vscale x 8 x i32> @ld2w_i32(<vscale x 4 x i1> %pred, i32* %addr) {
; CHECK-LABEL: ld2w_i32:
; CHECK: ld2w { z0.s, z1.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i32> @llvm.aarch64.sve.ld2.nxv8i32.nxv4i1.p0i32(<vscale x 4 x i1> %pred, i32* %addr)
  ret <vscale x 8 x i32> %res
}

define <vscale x 8 x float> @ld2w_f32(<vscale x 4 x i1> %pred, float* %addr) {
; CHECK-LABEL: ld2w_f32:
; CHECK: ld2w { z0.s, z1.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x float> @llvm.aarch64.sve.ld2.nxv8f32.nxv4i1.p0f32(<vscale x 4 x i1> %pred, float* %addr)
  ret <vscale x 8 x float> %res
}

;
; LD2D
;

define <vscale x 4 x i64> @ld2d_i64(<vscale x 2 x i1> %pred, i64* %addr) {
; CHECK-LABEL: ld2d_i64:
; CHECK: ld2d { z0.d, z1.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i64> @llvm.aarch64.sve.ld2.nxv4i64.nxv2i1.p0i64(<vscale x 2 x i1> %pred, i64* %addr)
  ret <vscale x 4 x i64> %res
}

define <vscale x 4 x double> @ld2d_f64(<vscale x 2 x i1> %pred, double* %addr) {
; CHECK-LABEL: ld2d_f64:
; CHECK: ld2d { z0.d, z1.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x double> @llvm.aarch64.sve.ld2.nxv4f64.nxv2i1.p0f64(<vscale x 2 x i1> %pred, double* %addr)
  ret <vscale x 4 x double> %res
}

;
; LD3B
;

define <vscale x 48 x i8> @ld3b_i8(<vscale x 16 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld3b_i8:
; CHECK: ld3b { z0.b, z1.b, z2.b }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 48 x i8> @llvm.aarch64.sve.ld3.nxv48i8.nxv16i1.p0i8(<vscale x 16 x i1> %pred, i8* %addr)
  ret <vscale x 48 x i8> %res
}

;
; LD3H
;

define <vscale x 24 x i16> @ld3h_i16(<vscale x 8 x i1> %pred, i16* %addr) {
; CHECK-LABEL: ld3h_i16:
; CHECK: ld3h { z0.h, z1.h, z2.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 24 x i16> @llvm.aarch64.sve.ld3.nxv24i16.nxv8i1.p0i16(<vscale x 8 x i1> %pred, i16* %addr)
  ret <vscale x 24 x i16> %res
}

define <vscale x 24 x half> @ld3h_f16(<vscale x 8 x i1> %pred, half* %addr) {
; CHECK-LABEL: ld3h_f16:
; CHECK: ld3h { z0.h, z1.h, z2.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 24 x half> @llvm.aarch64.sve.ld3.nxv24f16.nxv8i1.p0f16(<vscale x 8 x i1> %pred, half* %addr)
  ret <vscale x 24 x half> %res
}

define <vscale x 24 x bfloat> @ld3h_bf16(<vscale x 8 x i1> %pred, bfloat* %addr) {
; CHECK-LABEL: ld3h_bf16:
; CHECK: ld3h { z0.h, z1.h, z2.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 24 x bfloat> @llvm.aarch64.sve.ld3.nxv24bf16.nxv8i1.p0bf16(<vscale x 8 x i1> %pred, bfloat* %addr)
  ret <vscale x 24 x bfloat> %res
}

;
; LD3W
;

define <vscale x 12 x i32> @ld3w_i32(<vscale x 4 x i1> %pred, i32* %addr) {
; CHECK-LABEL: ld3w_i32:
; CHECK: ld3w { z0.s, z1.s, z2.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 12 x i32> @llvm.aarch64.sve.ld3.nxv12i32.nxv4i1.p0i32(<vscale x 4 x i1> %pred, i32* %addr)
  ret <vscale x 12 x i32> %res
}

define <vscale x 12 x float> @ld3w_f32(<vscale x 4 x i1> %pred, float* %addr) {
; CHECK-LABEL: ld3w_f32:
; CHECK: ld3w { z0.s, z1.s, z2.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 12 x float> @llvm.aarch64.sve.ld3.nxv12f32.nxv4i1.p0f32(<vscale x 4 x i1> %pred, float* %addr)
  ret <vscale x 12 x float> %res
}

;
; LD3D
;

define <vscale x 6 x i64> @ld3d_i64(<vscale x 2 x i1> %pred, i64* %addr) {
; CHECK-LABEL: ld3d_i64:
; CHECK: ld3d { z0.d, z1.d, z2.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 6 x i64> @llvm.aarch64.sve.ld3.nxv6i64.nxv2i1.p0i64(<vscale x 2 x i1> %pred, i64* %addr)
  ret <vscale x 6 x i64> %res
}

define <vscale x 6 x double> @ld3d_f64(<vscale x 2 x i1> %pred, double* %addr) {
; CHECK-LABEL: ld3d_f64:
; CHECK: ld3d { z0.d, z1.d, z2.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 6 x double> @llvm.aarch64.sve.ld3.nxv6f64.nxv2i1.p0f64(<vscale x 2 x i1> %pred, double* %addr)
  ret <vscale x 6 x double> %res
}

;
; LD4B
;

define <vscale x 64 x i8> @ld4b_i8(<vscale x 16 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld4b_i8:
; CHECK: ld4b { z0.b, z1.b, z2.b, z3.b }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 64 x i8> @llvm.aarch64.sve.ld4.nxv64i8.nxv16i1.p0i8(<vscale x 16 x i1> %pred, i8* %addr)
  ret <vscale x 64 x i8> %res
}

;
; LD4H
;

define <vscale x 32 x i16> @ld4h_i16(<vscale x 8 x i1> %pred, i16* %addr) {
; CHECK-LABEL: ld4h_i16:
; CHECK: ld4h { z0.h, z1.h, z2.h, z3.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 32 x i16> @llvm.aarch64.sve.ld4.nxv32i16.nxv8i1.p0i16(<vscale x 8 x i1> %pred, i16* %addr)
  ret <vscale x 32 x i16> %res
}

define <vscale x 32 x half> @ld4h_f16(<vscale x 8 x i1> %pred, half* %addr) {
; CHECK-LABEL: ld4h_f16:
; CHECK: ld4h { z0.h, z1.h, z2.h, z3.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 32 x half> @llvm.aarch64.sve.ld4.nxv32f16.nxv8i1.p0f16(<vscale x 8 x i1> %pred, half* %addr)
  ret <vscale x 32 x half> %res
}

define <vscale x 32 x bfloat> @ld4h_bf16(<vscale x 8 x i1> %pred, bfloat* %addr) {
; CHECK-LABEL: ld4h_bf16:
; CHECK: ld4h { z0.h, z1.h, z2.h, z3.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 32 x bfloat> @llvm.aarch64.sve.ld4.nxv32bf16.nxv8i1.p0bf16(<vscale x 8 x i1> %pred, bfloat* %addr)
  ret <vscale x 32 x bfloat> %res
}

;
; LD4W
;

define <vscale x 16 x i32> @ld4w_i32(<vscale x 4 x i1> %pred, i32* %addr) {
; CHECK-LABEL: ld4w_i32:
; CHECK: ld4w { z0.s, z1.s, z2.s, z3.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i32> @llvm.aarch64.sve.ld4.nxv16i32.nxv4i1.p0i32(<vscale x 4 x i1> %pred, i32* %addr)
  ret <vscale x 16 x i32> %res
}

define <vscale x 16 x float> @ld4w_f32(<vscale x 4 x i1> %pred, float* %addr) {
; CHECK-LABEL: ld4w_f32:
; CHECK: ld4w { z0.s, z1.s, z2.s, z3.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 16 x float> @llvm.aarch64.sve.ld4.nxv16f32.nxv4i1.p0f32(<vscale x 4 x i1> %pred, float* %addr)
  ret <vscale x 16 x float> %res
}

;
; LD4D
;

define <vscale x 8 x i64> @ld4d_i64(<vscale x 2 x i1> %pred, i64* %addr) {
; CHECK-LABEL: ld4d_i64:
; CHECK: ld4d { z0.d, z1.d, z2.d, z3.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i64> @llvm.aarch64.sve.ld4.nxv8i64.nxv2i1.p0i64(<vscale x 2 x i1> %pred, i64* %addr)
  ret <vscale x 8 x i64> %res
}

define <vscale x 8 x double> @ld4d_f64(<vscale x 2 x i1> %pred, double* %addr) {
; CHECK-LABEL: ld4d_f64:
; CHECK: ld4d { z0.d, z1.d, z2.d, z3.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x double> @llvm.aarch64.sve.ld4.nxv8f64.nxv2i1.p0f64(<vscale x 2 x i1> %pred, double* %addr)
  ret <vscale x 8 x double> %res
}


declare <vscale x 16 x i8> @llvm.aarch64.sve.ld1rq.nxv16i8(<vscale x 16 x i1>, i8*)
declare <vscale x 8 x i16> @llvm.aarch64.sve.ld1rq.nxv8i16(<vscale x 8 x i1>, i16*)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1rq.nxv4i32(<vscale x 4 x i1>, i32*)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1rq.nxv2i64(<vscale x 2 x i1>, i64*)
declare <vscale x 8 x half> @llvm.aarch64.sve.ld1rq.nxv8f16(<vscale x 8 x i1>, half*)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.ld1rq.nxv8bf16(<vscale x 8 x i1>, bfloat*)
declare <vscale x 4 x float> @llvm.aarch64.sve.ld1rq.nxv4f32(<vscale x 4 x i1>, float*)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1rq.nxv2f64(<vscale x 2 x i1>, double*)

declare <vscale x 16 x i8> @llvm.aarch64.sve.ldnt1.nxv16i8(<vscale x 16 x i1>, i8*)
declare <vscale x 8 x i16> @llvm.aarch64.sve.ldnt1.nxv8i16(<vscale x 8 x i1>, i16*)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.nxv4i32(<vscale x 4 x i1>, i32*)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1>, i64*)
declare <vscale x 8 x half> @llvm.aarch64.sve.ldnt1.nxv8f16(<vscale x 8 x i1>, half*)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.ldnt1.nxv8bf16(<vscale x 8 x i1>, bfloat*)
declare <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.nxv4f32(<vscale x 4 x i1>, float*)
declare <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.nxv2f64(<vscale x 2 x i1>, double*)

declare <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1.p0i8(<vscale x 16 x i1>, i8*)
declare <vscale x 16 x i16> @llvm.aarch64.sve.ld2.nxv16i16.nxv8i1.p0i16(<vscale x 8 x i1>, i16*)
declare <vscale x 8 x i32> @llvm.aarch64.sve.ld2.nxv8i32.nxv4i1.p0i32(<vscale x 4 x i1>, i32*)
declare <vscale x 4 x i64> @llvm.aarch64.sve.ld2.nxv4i64.nxv2i1.p0i64(<vscale x 2 x i1>, i64*)
declare <vscale x 16 x half> @llvm.aarch64.sve.ld2.nxv16f16.nxv8i1.p0f16(<vscale x 8 x i1>, half*)
declare <vscale x 16 x bfloat> @llvm.aarch64.sve.ld2.nxv16bf16.nxv8i1.p0bf16(<vscale x 8 x i1>, bfloat*)
declare <vscale x 8 x float> @llvm.aarch64.sve.ld2.nxv8f32.nxv4i1.p0f32(<vscale x 4 x i1>, float*)
declare <vscale x 4 x double> @llvm.aarch64.sve.ld2.nxv4f64.nxv2i1.p0f64(<vscale x 2 x i1>, double*)

declare <vscale x 48 x i8> @llvm.aarch64.sve.ld3.nxv48i8.nxv16i1.p0i8(<vscale x 16 x i1>, i8*)
declare <vscale x 24 x i16> @llvm.aarch64.sve.ld3.nxv24i16.nxv8i1.p0i16(<vscale x 8 x i1>, i16*)
declare <vscale x 12 x i32> @llvm.aarch64.sve.ld3.nxv12i32.nxv4i1.p0i32(<vscale x 4 x i1>, i32*)
declare <vscale x 6 x i64> @llvm.aarch64.sve.ld3.nxv6i64.nxv2i1.p0i64(<vscale x 2 x i1>, i64*)
declare <vscale x 24 x half> @llvm.aarch64.sve.ld3.nxv24f16.nxv8i1.p0f16(<vscale x 8 x i1>, half*)
declare <vscale x 24 x bfloat> @llvm.aarch64.sve.ld3.nxv24bf16.nxv8i1.p0bf16(<vscale x 8 x i1>, bfloat*)
declare <vscale x 12 x float> @llvm.aarch64.sve.ld3.nxv12f32.nxv4i1.p0f32(<vscale x 4 x i1>, float*)
declare <vscale x 6 x double> @llvm.aarch64.sve.ld3.nxv6f64.nxv2i1.p0f64(<vscale x 2 x i1>, double*)

declare <vscale x 64 x i8> @llvm.aarch64.sve.ld4.nxv64i8.nxv16i1.p0i8(<vscale x 16 x i1>, i8*)
declare <vscale x 32 x i16> @llvm.aarch64.sve.ld4.nxv32i16.nxv8i1.p0i16(<vscale x 8 x i1>, i16*)
declare <vscale x 16 x i32> @llvm.aarch64.sve.ld4.nxv16i32.nxv4i1.p0i32(<vscale x 4 x i1>, i32*)
declare <vscale x 8 x i64> @llvm.aarch64.sve.ld4.nxv8i64.nxv2i1.p0i64(<vscale x 2 x i1>, i64*)
declare <vscale x 32 x half> @llvm.aarch64.sve.ld4.nxv32f16.nxv8i1.p0f16(<vscale x 8 x i1>, half*)
declare <vscale x 32 x bfloat> @llvm.aarch64.sve.ld4.nxv32bf16.nxv8i1.p0bf16(<vscale x 8 x i1>, bfloat*)
declare <vscale x 16 x float> @llvm.aarch64.sve.ld4.nxv16f32.nxv4i1.p0f32(<vscale x 4 x i1>, float*)
declare <vscale x 8 x double> @llvm.aarch64.sve.ld4.nxv8f64.nxv2i1.p0f64(<vscale x 2 x i1>, double*)
