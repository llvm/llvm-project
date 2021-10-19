; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; LD1B, LD1W, LD1H, LD1D: base + 64-bit unscaled offset
;   e.g. ld1h { z0.d }, p0/z, [x0, z0.d]
;

define <vscale x 2 x i64> @gld1b_d(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1b_d:
; CHECK: ld1b { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.nxv2i8(<vscale x 2 x i1> %pg,
                                                                     i8* %base,
                                                                     <vscale x 2 x i64> %b)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1h_d(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1h_d:
; CHECK: ld1h { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.nxv2i16(<vscale x 2 x i1> %pg,
                                                                       i16* %base,
                                                                       <vscale x 2 x i64> %b)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1w_d(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: gld1w_d:
; CHECK: ld1w { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.nxv2i32(<vscale x 2 x i1> %pg,
                                                                       i32* %base,
                                                                       <vscale x 2 x i64> %offsets)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1d_d(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1d_d:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.nxv2i64(<vscale x 2 x i1> %pg,
                                                                       i64* %base,
                                                                       <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gld1d_d_double(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1d_d_double:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.nxv2f64(<vscale x 2 x i1> %pg,
                                                                       double* %base,
                                                                       <vscale x 2 x i64> %b)
  ret <vscale x 2 x double> %load
}

;
; LD1SB, LD1SW, LD1SH: base + 64-bit unscaled offset
;   e.g. ld1sh { z0.d }, p0/z, [x0, z0.d]
;

define <vscale x 2 x i64> @gld1sb_d(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1sb_d:
; CHECK: ld1sb { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.nxv2i8(<vscale x 2 x i1> %pg,
                                                                     i8* %base,
                                                                     <vscale x 2 x i64> %b)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sh_d(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1sh_d:
; CHECK: ld1sh { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.nxv2i16(<vscale x 2 x i1> %pg,
                                                                       i16* %base,
                                                                       <vscale x 2 x i64> %b)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sw_d(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: gld1sw_d:
; CHECK: ld1sw { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.nxv2i32(<vscale x 2 x i1> %pg,
                                                                       i32* %base,
                                                                       <vscale x 2 x i64> %offsets)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

;
; LD1B, LD1W, LD1H, LD1D: base + 64-bit sxtw'd unscaled offset
;   e.g. ld1h { z0.d }, p0/z, [x0, z0.d, sxtw]
;

define <vscale x 2 x i64> @gld1b_d_sxtw(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1b_d_sxtw:
; CHECK: ld1b { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.nxv2i8(<vscale x 2 x i1> %pg,
                                                                     i8* %base,
                                                                     <vscale x 2 x i64> %sxtw)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1h_d_sxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1h_d_sxtw:
; CHECK: ld1h { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.nxv2i16(<vscale x 2 x i1> %pg,
                                                                       i16* %base,
                                                                       <vscale x 2 x i64> %sxtw)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1w_d_sxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: gld1w_d_sxtw:
; CHECK: ld1w { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %offsets)
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.nxv2i32(<vscale x 2 x i1> %pg,
                                                                       i32* %base,
                                                                       <vscale x 2 x i64> %sxtw)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1d_d_sxtw(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1d_d_sxtw:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.nxv2i64(<vscale x 2 x i1> %pg,
                                                                       i64* %base,
                                                                       <vscale x 2 x i64> %sxtw)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gld1d_d_double_sxtw(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1d_d_double_sxtw:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.nxv2f64(<vscale x 2 x i1> %pg,
                                                                       double* %base,
                                                                       <vscale x 2 x i64> %sxtw)
  ret <vscale x 2 x double> %load
}

;
; LD1SB, LD1SW, LD1SH: base + 64-bit sxtw'd unscaled offset
;   e.g. ld1sh { z0.d }, p0/z, [x0, z0.d]
;

define <vscale x 2 x i64> @gld1sb_d_sxtw(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1sb_d_sxtw:
; CHECK: ld1sb { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.nxv2i8(<vscale x 2 x i1> %pg,
                                                                     i8* %base,
                                                                     <vscale x 2 x i64> %sxtw)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sh_d_sxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1sh_d_sxtw:
; CHECK: ld1sh { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.nxv2i16(<vscale x 2 x i1> %pg,
                                                                       i16* %base,
                                                                       <vscale x 2 x i64> %sxtw)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sw_d_sxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: gld1sw_d_sxtw:
; CHECK: ld1sw { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %offsets)
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.nxv2i32(<vscale x 2 x i1> %pg,
                                                                       i32* %base,
                                                                       <vscale x 2 x i64> %sxtw)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

;
; LD1B, LD1W, LD1H, LD1D: base + 64-bit uxtw'd unscaled offset
;   e.g. ld1h { z0.d }, p0/z, [x0, z0.d, uxtw]
;

define <vscale x 2 x i64> @gld1b_d_uxtw(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1b_d_uxtw:
; CHECK: ld1b { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.nxv2i8(<vscale x 2 x i1> %pg,
                                                                     i8* %base,
                                                                     <vscale x 2 x i64> %uxtw)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1h_d_uxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1h_d_uxtw:
; CHECK: ld1h { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.nxv2i16(<vscale x 2 x i1> %pg,
                                                                       i16* %base,
                                                                       <vscale x 2 x i64> %uxtw)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1w_d_uxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: gld1w_d_uxtw:
; CHECK: ld1w { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %offsets)
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.nxv2i32(<vscale x 2 x i1> %pg,
                                                                       i32* %base,
                                                                       <vscale x 2 x i64> %uxtw)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1d_d_uxtw(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1d_d_uxtw:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.nxv2i64(<vscale x 2 x i1> %pg,
                                                                       i64* %base,
                                                                       <vscale x 2 x i64> %uxtw)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gld1d_d_double_uxtw(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1d_d_double_uxtw:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.nxv2f64(<vscale x 2 x i1> %pg,
                                                                       double* %base,
                                                                       <vscale x 2 x i64> %uxtw)
  ret <vscale x 2 x double> %load
}

;
; LD1SB, LD1SW, LD1SH: base + 64-bit uxtw'd unscaled offset
;   e.g. ld1sh { z0.d }, p0/z, [x0, z0.d]
;

define <vscale x 2 x i64> @gld1sb_d_uxtw(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1sb_d_uxtw:
; CHECK: ld1sb { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.nxv2i8(<vscale x 2 x i1> %pg,
                                                                     i8* %base,
                                                                     <vscale x 2 x i64> %uxtw)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sh_d_uxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1sh_d_uxtw:
; CHECK: ld1sh { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.nxv2i16(<vscale x 2 x i1> %pg,
                                                                       i16* %base,
                                                                       <vscale x 2 x i64> %uxtw)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sw_d_uxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: gld1sw_d_uxtw:
; CHECK: ld1sw { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %offsets)
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.nxv2i32(<vscale x 2 x i1> %pg,
                                                                       i32* %base,
                                                                       <vscale x 2 x i64> %uxtw)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

declare <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.nxv2i8(<vscale x 2 x i1>, i8*, <vscale x 2 x i64>)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i64>)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i64>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i64>)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i64>)

declare <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)
