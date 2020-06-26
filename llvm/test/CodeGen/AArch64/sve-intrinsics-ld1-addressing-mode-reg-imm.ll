; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; LD1B
;

define <vscale x 16 x i8> @ld1b_upper_bound(<vscale x 16 x i1> %pg, i8* %a) {
; CHECK-LABEL: ld1b_upper_bound:
; CHECK: ld1b { z0.b }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 16 x i8>*
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 16 x i8>* %base to i8*
  %load = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1> %pg, i8* %base_scalar)
  ret <vscale x 16 x i8> %load
}

define <vscale x 16 x i8> @ld1b_inbound(<vscale x 16 x i1> %pg, i8* %a) {
; CHECK-LABEL: ld1b_inbound:
; CHECK: ld1b { z0.b }, p0/z, [x0, #1, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 16 x i8>*
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %base_scalable, i64 1
  %base_scalar = bitcast <vscale x 16 x i8>* %base to i8*
  %load = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1> %pg, i8* %base_scalar)
  ret <vscale x 16 x i8> %load
}

define <vscale x 4 x i32> @ld1b_s_inbound(<vscale x 4 x i1> %pg, i8* %a) {
; CHECK-LABEL: ld1b_s_inbound:
; CHECK: ld1b { z0.s }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 4 x i8>*
  %base = getelementptr <vscale x 4 x i8>, <vscale x 4 x i8>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 4 x i8>* %base to i8*
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ld1.nxv4i8(<vscale x 4 x i1> %pg, i8* %base_scalar)
  %res = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @ld1sb_s_inbound(<vscale x 4 x i1> %pg, i8* %a) {
; CHECK-LABEL: ld1sb_s_inbound:
; CHECK: ld1sb { z0.s }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 4 x i8>*
  %base = getelementptr <vscale x 4 x i8>, <vscale x 4 x i8>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 4 x i8>* %base to i8*
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ld1.nxv4i8(<vscale x 4 x i1> %pg, i8* %base_scalar)
  %res = sext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 16 x i8> @ld1b_lower_bound(<vscale x 16 x i1> %pg, i8* %a) {
; CHECK-LABEL: ld1b_lower_bound:
; CHECK: ld1b { z0.b }, p0/z, [x0, #-8, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 16 x i8>*
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %base_scalable, i64 -8
  %base_scalar = bitcast <vscale x 16 x i8>* %base to i8*
  %load = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1> %pg, i8* %base_scalar)
  ret <vscale x 16 x i8> %load
}

define <vscale x 16 x i8> @ld1b_out_of_upper_bound(<vscale x 16 x i1> %pg, i8* %a) {
; CHECK-LABEL: ld1b_out_of_upper_bound:
; CHECK: rdvl x[[OFFSET:[0-9]+]], #8
; CHECK-NEXT: ld1b { z0.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT:  ret
  %base_scalable = bitcast i8* %a to <vscale x 16 x i8>*
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %base_scalable, i64 8
  %base_scalar = bitcast <vscale x 16 x i8>* %base to i8*
  %load = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1> %pg, i8* %base_scalar)
  ret <vscale x 16 x i8> %load
}

define <vscale x 16 x i8> @ld1b_out_of_lower_bound(<vscale x 16 x i1> %pg, i8* %a) {
; CHECK-LABEL: ld1b_out_of_lower_bound:
; CHECK: rdvl x[[OFFSET:[0-9]+]], #-9
; CHECK-NEXT: ld1b { z0.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT:  ret
  %base_scalable = bitcast i8* %a to <vscale x 16 x i8>*
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %base_scalable, i64 -9
  %base_scalar = bitcast <vscale x 16 x i8>* %base to i8*
  %load = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1> %pg, i8* %base_scalar)
  ret <vscale x 16 x i8> %load
}

;
; LD1H
;

define <vscale x 8 x i16> @ld1b_h_inbound(<vscale x 8 x i1> %pg, i8* %a) {
; CHECK-LABEL: ld1b_h_inbound:
; CHECK: ld1b { z0.h }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 8 x i8>*
  %base = getelementptr <vscale x 8 x i8>, <vscale x 8 x i8>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 8 x i8>* %base to i8*
  %load = call <vscale x 8 x i8> @llvm.aarch64.sve.ld1.nxv8i8(<vscale x 8 x i1> %pg, i8* %base_scalar)
  %res = zext <vscale x 8 x i8> %load to <vscale x 8 x i16>
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @ld1sb_h_inbound(<vscale x 8 x i1> %pg, i8* %a) {
; CHECK-LABEL: ld1sb_h_inbound:
; CHECK: ld1sb { z0.h }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 8 x i8>*
  %base = getelementptr <vscale x 8 x i8>, <vscale x 8 x i8>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 8 x i8>* %base to i8*
  %load = call <vscale x 8 x i8> @llvm.aarch64.sve.ld1.nxv8i8(<vscale x 8 x i1> %pg, i8* %base_scalar)
  %res = sext <vscale x 8 x i8> %load to <vscale x 8 x i16>
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @ld1h_inbound(<vscale x 8 x i1> %pg, i16* %a) {
; CHECK-LABEL: ld1h_inbound:
; CHECK: ld1h { z0.h }, p0/z, [x0, #1, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i16* %a to <vscale x 8 x i16>*
  %base = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %base_scalable, i64 1
  %base_scalar = bitcast <vscale x 8 x i16>* %base to i16*
  %load = call <vscale x 8 x i16> @llvm.aarch64.sve.ld1.nxv8i16(<vscale x 8 x i1> %pg, i16* %base_scalar)
  ret <vscale x 8 x i16> %load
}

define <vscale x 4 x i32> @ld1h_s_inbound(<vscale x 4 x i1> %pg, i16* %a) {
; CHECK-LABEL: ld1h_s_inbound:
; CHECK: ld1h { z0.s }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i16* %a to <vscale x 4 x i16>*
  %base = getelementptr <vscale x 4 x i16>, <vscale x 4 x i16>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 4 x i16>* %base to i16*
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.nxv4i16(<vscale x 4 x i1> %pg, i16* %base_scalar)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @ld1sh_s_inbound(<vscale x 4 x i1> %pg, i16* %a) {
; CHECK-LABEL: ld1sh_s_inbound:
; CHECK: ld1sh { z0.s }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i16* %a to <vscale x 4 x i16>*
  %base = getelementptr <vscale x 4 x i16>, <vscale x 4 x i16>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 4 x i16>* %base to i16*
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.nxv4i16(<vscale x 4 x i1> %pg, i16* %base_scalar)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @ld1b_d_inbound(<vscale x 2 x i1> %pg, i8* %a) {
; CHECK-LABEL: ld1b_d_inbound:
; CHECK: ld1b { z0.d }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 2 x i8>*
  %base = getelementptr <vscale x 2 x i8>, <vscale x 2 x i8>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 2 x i8>* %base to i8*
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.nxv2i8(<vscale x 2 x i1> %pg, i8* %base_scalar)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @ld1sb_d_inbound(<vscale x 2 x i1> %pg, i8* %a) {
; CHECK-LABEL: ld1sb_d_inbound:
; CHECK: ld1sb { z0.d }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 2 x i8>*
  %base = getelementptr <vscale x 2 x i8>, <vscale x 2 x i8>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 2 x i8>* %base to i8*
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.nxv2i8(<vscale x 2 x i1> %pg, i8* %base_scalar)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @ld1h_d_inbound(<vscale x 2 x i1> %pg, i16* %a) {
; CHECK-LABEL: ld1h_d_inbound:
; CHECK: ld1h { z0.d }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i16* %a to <vscale x 2 x i16>*
  %base = getelementptr <vscale x 2 x i16>, <vscale x 2 x i16>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 2 x i16>* %base to i16*
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.nxv2i16(<vscale x 2 x i1> %pg, i16* %base_scalar)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @ld1sh_d_inbound(<vscale x 2 x i1> %pg, i16* %a) {
; CHECK-LABEL: ld1sh_d_inbound:
; CHECK: ld1sh { z0.d }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i16* %a to <vscale x 2 x i16>*
  %base = getelementptr <vscale x 2 x i16>, <vscale x 2 x i16>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 2 x i16>* %base to i16*
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.nxv2i16(<vscale x 2 x i1> %pg, i16* %base_scalar)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 8 x half> @ld1h_f16_inbound(<vscale x 8 x i1> %pg, half* %a) {
; CHECK-LABEL: ld1h_f16_inbound:
; CHECK: ld1h { z0.h }, p0/z, [x0, #1, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast half* %a to <vscale x 8 x half>*
  %base = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %base_scalable, i64 1
  %base_scalar = bitcast <vscale x 8 x half>* %base to half*
  %load = call <vscale x 8 x half> @llvm.aarch64.sve.ld1.nxv8f16(<vscale x 8 x i1> %pg, half* %base_scalar)
  ret <vscale x 8 x half> %load
}

define <vscale x 8 x bfloat> @ld1h_bf16_inbound(<vscale x 8 x i1> %pg, bfloat* %a) #0 {
; CHECK-LABEL: ld1h_bf16_inbound:
; CHECK: ld1h { z0.h }, p0/z, [x0, #1, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast bfloat* %a to <vscale x 8 x bfloat>*
  %base = getelementptr <vscale x 8 x bfloat>, <vscale x 8 x bfloat>* %base_scalable, i64 1
  %base_scalar = bitcast <vscale x 8 x bfloat>* %base to bfloat*
  %load = call <vscale x 8 x bfloat> @llvm.aarch64.sve.ld1.nxv8bf16(<vscale x 8 x i1> %pg, bfloat* %base_scalar)
  ret <vscale x 8 x bfloat> %load
}

;
; LD1W
;

define <vscale x 4 x i32> @ld1w_inbound(<vscale x 4 x i1> %pg, i32* %a) {
; CHECK-LABEL: ld1w_inbound:
; CHECK: ld1w { z0.s }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i32* %a to <vscale x 4 x i32>*
  %base = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 4 x i32>* %base to i32*
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1.nxv4i32(<vscale x 4 x i1> %pg, i32* %base_scalar)
  ret <vscale x 4 x i32> %load
}

define <vscale x 4 x float> @ld1w_f32_inbound(<vscale x 4 x i1> %pg, float* %a) {
; CHECK-LABEL: ld1w_f32_inbound:
; CHECK: ld1w { z0.s }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast float* %a to <vscale x 4 x float>*
  %base = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 4 x float>* %base to float*
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ld1.nxv4f32(<vscale x 4 x i1> %pg, float* %base_scalar)
  ret <vscale x 4 x float> %load
}

;
; LD1D
;

define <vscale x 2 x i64> @ld1d_inbound(<vscale x 2 x i1> %pg, i64* %a) {
; CHECK-LABEL: ld1d_inbound:
; CHECK: ld1d { z0.d }, p0/z, [x0, #1, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i64* %a to <vscale x 2 x i64>*
  %base = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %base_scalable, i64 1
  %base_scalar = bitcast <vscale x 2 x i64>* %base to i64*
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.nxv2i64(<vscale x 2 x i1> %pg, i64* %base_scalar)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x i64> @ld1w_d_inbound(<vscale x 2 x i1> %pg, i32* %a) {
; CHECK-LABEL: ld1w_d_inbound:
; CHECK: ld1w { z0.d }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i32* %a to <vscale x 2 x i32>*
  %base = getelementptr <vscale x 2 x i32>, <vscale x 2 x i32>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 2 x i32>* %base to i32*
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.nxv2i32(<vscale x 2 x i1> %pg, i32* %base_scalar)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @ld1sw_d_inbound(<vscale x 2 x i1> %pg, i32* %a) {
; CHECK-LABEL: ld1sw_d_inbound:
; CHECK: ld1sw { z0.d }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i32* %a to <vscale x 2 x i32>*
  %base = getelementptr <vscale x 2 x i32>, <vscale x 2 x i32>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 2 x i32>* %base to i32*
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.nxv2i32(<vscale x 2 x i1> %pg, i32* %base_scalar)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x double> @ld1d_f64_inbound(<vscale x 2 x i1> %pg, double* %a) {
; CHECK-LABEL: ld1d_f64_inbound:
; CHECK: ld1d { z0.d }, p0/z, [x0, #1, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast double* %a to <vscale x 2 x double>*
  %base = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %base_scalable, i64 1
  %base_scalar = bitcast <vscale x 2 x double>* %base to double*
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.nxv2f64(<vscale x 2 x i1> %pg, double* %base_scalar)
  ret <vscale x 2 x double> %load
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1>, i8*)

declare <vscale x 8 x i8> @llvm.aarch64.sve.ld1.nxv8i8(<vscale x 8 x i1>, i8*)
declare <vscale x 8 x i16> @llvm.aarch64.sve.ld1.nxv8i16(<vscale x 8 x i1>, i16*)
declare <vscale x 8 x half> @llvm.aarch64.sve.ld1.nxv8f16(<vscale x 8 x i1>, half*)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.ld1.nxv8bf16(<vscale x 8 x i1>, bfloat*)

declare <vscale x 4 x i8> @llvm.aarch64.sve.ld1.nxv4i8(<vscale x 4 x i1>, i8*)
declare <vscale x 4 x i16> @llvm.aarch64.sve.ld1.nxv4i16(<vscale x 4 x i1>, i16*)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1.nxv4i32(<vscale x 4 x i1>, i32*)
declare <vscale x 4 x float> @llvm.aarch64.sve.ld1.nxv4f32(<vscale x 4 x i1>, float*)

declare <vscale x 2 x i8> @llvm.aarch64.sve.ld1.nxv2i8(<vscale x 2 x i1>, i8*)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ld1.nxv2i16(<vscale x 2 x i1>, i16*)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ld1.nxv2i32(<vscale x 2 x i1>, i32*)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1.nxv2i64(<vscale x 2 x i1>, i64*)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.nxv2f64(<vscale x 2 x i1>, double*)

; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+sve,+bf16" }
