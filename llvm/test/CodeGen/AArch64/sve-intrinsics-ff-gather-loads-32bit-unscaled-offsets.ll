; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; LDFF1B, LDFF1W, LDFF1H, LDFF1D: base + 32-bit unscaled offset, sign (sxtw) or zero
; (uxtw) extended to 64 bits.
;   e.g. ldff1h { z0.d }, p0/z, [x0, z0.d, uxtw]
;

; LDFF1B
define <vscale x 4 x i32> @gldff1b_s_uxtw(<vscale x 4 x i1> %pg, i8* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1b_s_uxtw:
; CHECK: ldff1b { z0.s }, p0/z, [x0, z0.s, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv4i8(<vscale x 4 x i1> %pg,
                                                                            i8* %base,
                                                                            <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @gldff1b_s_sxtw(<vscale x 4 x i1> %pg, i8* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1b_s_sxtw:
; CHECK: ldff1b { z0.s }, p0/z, [x0, z0.s, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv4i8(<vscale x 4 x i1> %pg,
                                                                            i8* %base,
                                                                            <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1b_d_uxtw(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1b_d_uxtw:
; CHECK: ldff1b { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv2i8(<vscale x 2 x i1> %pg,
                                                                            i8* %base,
                                                                            <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1b_d_sxtw(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1b_d_sxtw:
; CHECK: ldff1b { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv2i8(<vscale x 2 x i1> %pg,
                                                                            i8* %base,
                                                                            <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1H
define <vscale x 4 x i32> @gldff1h_s_uxtw(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1h_s_uxtw:
; CHECK: ldff1h { z0.s }, p0/z, [x0, z0.s, uxtw]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv4i16(<vscale x 4 x i1> %pg,
                                                                              i16* %base,
                                                                              <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @gldff1h_s_sxtw(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1h_s_sxtw:
; CHECK: ldff1h { z0.s }, p0/z, [x0, z0.s, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv4i16(<vscale x 4 x i1> %pg,
                                                                              i16* %base,
                                                                              <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1h_d_uxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1h_d_uxtw:
; CHECK: ldff1h { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv2i16(<vscale x 2 x i1> %pg,
                                                                              i16* %base,
                                                                              <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1h_d_sxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1h_d_sxtw:
; CHECK: ldff1h { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv2i16(<vscale x 2 x i1> %pg,
                                                                              i16* %base,
                                                                              <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1W
define <vscale x 4 x i32> @gldff1w_s_uxtw(<vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1w_s_uxtw:
; CHECK: ldff1w { z0.s }, p0/z, [x0, z0.s, uxtw]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv4i32(<vscale x 4 x i1> %pg,
                                                                              i32* %base,
                                                                              <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %load
}

define <vscale x 4 x i32> @gldff1w_s_sxtw(<vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1w_s_sxtw:
; CHECK: ldff1w { z0.s }, p0/z, [x0, z0.s, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv4i32(<vscale x 4 x i1> %pg,
                                                                              i32* %base,
                                                                              <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %load
}

define <vscale x 2 x i64> @gldff1w_d_uxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1w_d_uxtw:
; CHECK: ldff1w { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv2i32(<vscale x 2 x i1> %pg,
                                                                              i32* %base,
                                                                              <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1w_d_sxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1w_d_sxtw:
; CHECK: ldff1w { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv2i32(<vscale x 2 x i1> %pg,
                                                                              i32* %base,
                                                                              <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x float> @gldff1w_s_uxtw_float(<vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1w_s_uxtw_float:
; CHECK: ldff1w { z0.s }, p0/z, [x0, z0.s, uxtw]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv4f32(<vscale x 4 x i1> %pg,
                                                                                float* %base,
                                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x float> %load
}

define <vscale x 4 x float> @gldff1w_s_sxtw_float(<vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1w_s_sxtw_float:
; CHECK: ldff1w { z0.s }, p0/z, [x0, z0.s, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv4f32(<vscale x 4 x i1> %pg,
                                                                                float* %base,
                                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x float> %load
}

; LDFF1D
define <vscale x 2 x i64> @gldff1d_d_uxtw(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1d_d_uxtw:
; CHECK: ldff1d { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv2i64(<vscale x 2 x i1> %pg,
                                                                              i64* %base,
                                                                              <vscale x 2 x i32> %b)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x i64> @gldff1d_d_sxtw(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1d_d_sxtw:
; CHECK: ldff1d { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv2i64(<vscale x 2 x i1> %pg,
                                                                              i64* %base,
                                                                              <vscale x 2 x i32> %b)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gldff1d_d_uxtw_double(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1d_d_uxtw_double:
; CHECK: ldff1d { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv2f64(<vscale x 2 x i1> %pg,
                                                                                 double* %base,
                                                                                 <vscale x 2 x i32> %b)
  ret <vscale x 2 x double> %load
}

define <vscale x 2 x double> @gldff1d_d_sxtw_double(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1d_d_sxtw_double:
; CHECK: ldff1d { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv2f64(<vscale x 2 x i1> %pg,
                                                                                 double* %base,
                                                                                 <vscale x 2 x i32> %b)
  ret <vscale x 2 x double> %load
}

;
; LDFF1SB, LDFF1SW, LDFF1SH: base + 32-bit unscaled offset, sign (sxtw) or zero
; (uxtw) extended to 64 bits.
;   e.g. ldff1sh { z0.d }, p0/z, [x0, z0.d, uxtw]
;

; LDFF1SB
define <vscale x 4 x i32> @gldff1sb_s_uxtw(<vscale x 4 x i1> %pg, i8* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1sb_s_uxtw:
; CHECK: ldff1sb { z0.s }, p0/z, [x0, z0.s, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv4i8(<vscale x 4 x i1> %pg,
                                                                            i8* %base,
                                                                            <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @gldff1sb_s_sxtw(<vscale x 4 x i1> %pg, i8* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1sb_s_sxtw:
; CHECK: ldff1sb { z0.s }, p0/z, [x0, z0.s, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv4i8(<vscale x 4 x i1> %pg,
                                                                            i8* %base,
                                                                            <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1sb_d_uxtw(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1sb_d_uxtw:
; CHECK: ldff1sb { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv2i8(<vscale x 2 x i1> %pg,
                                                                            i8* %base,
                                                                            <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1sb_d_sxtw(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1sb_d_sxtw:
; CHECK: ldff1sb { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv2i8(<vscale x 2 x i1> %pg,
                                                                            i8* %base,
                                                                            <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1SH
define <vscale x 4 x i32> @gldff1sh_s_uxtw(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1sh_s_uxtw:
; CHECK: ldff1sh { z0.s }, p0/z, [x0, z0.s, uxtw]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv4i16(<vscale x 4 x i1> %pg,
                                                                              i16* %base,
                                                                              <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @gldff1sh_s_sxtw(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1sh_s_sxtw:
; CHECK: ldff1sh { z0.s }, p0/z, [x0, z0.s, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv4i16(<vscale x 4 x i1> %pg,
                                                                              i16* %base,
                                                                              <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1sh_d_uxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1sh_d_uxtw:
; CHECK: ldff1sh { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv2i16(<vscale x 2 x i1> %pg,
                                                                              i16* %base,
                                                                              <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1sh_d_sxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1sh_d_sxtw:
; CHECK: ldff1sh { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv2i16(<vscale x 2 x i1> %pg,
                                                                              i16* %base,
                                                                              <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1SW
define <vscale x 2 x i64> @gldff1sw_d_uxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1sw_d_uxtw:
; CHECK: ldff1sw { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv2i32(<vscale x 2 x i1> %pg,
                                                                              i32* %base,
                                                                              <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1sw_d_sxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1sw_d_sxtw:
; CHECK: ldff1sw { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv2i32(<vscale x 2 x i1> %pg,
                                                                              i32* %base,
                                                                              <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1B/LDFF1SB
declare <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv4i8(<vscale x 4 x i1>, i8*, <vscale x 4 x i32>)
declare <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv2i8(<vscale x 2 x i1>, i8*, <vscale x 2 x i32>)
declare <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv4i8(<vscale x 4 x i1>, i8*, <vscale x 4 x i32>)
declare <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv2i8(<vscale x 2 x i1>, i8*, <vscale x 2 x i32>)

; LDFF1H/LDFF1SH
declare <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv4i16(<vscale x 4 x i1>, i16*, <vscale x 4 x i32>)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i32>)
declare <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv4i16(<vscale x 4 x i1>, i16*, <vscale x 4 x i32>)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i32>)

; LDFF1W/LDFF1SW
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv4i32(<vscale x 4 x i1>, i32*, <vscale x 4 x i32>)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv4i32(<vscale x 4 x i1>, i32*, <vscale x 4 x i32>)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i32>)

declare <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv4f32(<vscale x 4 x i1>, float*, <vscale x 4 x i32>)
declare <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv4f32(<vscale x 4 x i1>, float*, <vscale x 4 x i32>)

; LDFF1D
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i32>)

declare <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i32>)
declare <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i32>)
