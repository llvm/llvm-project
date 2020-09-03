; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s

;
; LDNT1B, LDNT1W, LDNT1H, LDNT1D: base + 64-bit unscaled offsets
;   e.g. ldnt1h { z0.d }, p0/z, [z0.d, x0]
;

define <vscale x 2 x i64> @gldnt1b_d(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldnt1b_d:
; CHECK: ldnt1b { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldnt1.gather.nxv2i8(<vscale x 2 x i1> %pg,
                                                                       i8* %base,
                                                                       <vscale x 2 x i64> %b)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldnt1h_d(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldnt1h_d:
; CHECK: ldnt1h { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldnt1.gather.nxv2i16(<vscale x 2 x i1> %pg,
                                                                         i16* %base,
                                                                         <vscale x 2 x i64> %b)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldnt1w_d(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: gldnt1w_d:
; CHECK: ldnt1w { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldnt1.gather.nxv2i32(<vscale x 2 x i1> %pg,
                                                                         i32* %base,
                                                                         <vscale x 2 x i64> %offsets)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldnt1d_d(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldnt1d_d:
; CHECK: ldnt1d { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.nxv2i64(<vscale x 2 x i1> %pg,
                                                                         i64* %base,
                                                                         <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gldnt1d_d_double(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldnt1d_d_double:
; CHECK: ldnt1d { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.gather.nxv2f64(<vscale x 2 x i1> %pg,
                                                                            double* %base,
                                                                            <vscale x 2 x i64> %b)
  ret <vscale x 2 x double> %load
}

;
; LDNT1SB, LDNT1SW, LDNT1SH: base + 64-bit unscaled offsets
;   e.g. ldnt1sh { z0.d }, p0/z, [z0.d, x0]
;

define <vscale x 2 x i64> @gldnt1sb_d(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldnt1sb_d:
; CHECK: ldnt1sb { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldnt1.gather.nxv2i8(<vscale x 2 x i1> %pg,
                                                                       i8* %base,
                                                                       <vscale x 2 x i64> %b)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldnt1sh_d(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldnt1sh_d:
; CHECK: ldnt1sh { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldnt1.gather.nxv2i16(<vscale x 2 x i1> %pg,
                                                                         i16* %base,
                                                                         <vscale x 2 x i64> %b)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldnt1sw_d(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: gldnt1sw_d:
; CHECK: ldnt1sw { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldnt1.gather.nxv2i32(<vscale x 2 x i1> %pg,
                                                                         i32* %base,
                                                                         <vscale x 2 x i64> %offsets)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

declare <vscale x 2 x i8> @llvm.aarch64.sve.ldnt1.gather.nxv2i8(<vscale x 2 x i1>, i8*, <vscale x 2 x i64>)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ldnt1.gather.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i64>)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ldnt1.gather.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i64>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i64>)
declare <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.gather.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i64>)
