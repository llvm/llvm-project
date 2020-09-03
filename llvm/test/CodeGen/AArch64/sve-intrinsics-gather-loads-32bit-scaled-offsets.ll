; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; LD1H, LD1W, LD1D: base + 32-bit scaled offset, sign (sxtw) or zero (uxtw)
; extended to 64 bits
;   e.g. ld1h z0.d, p0/z, [x0, z0.d, uxtw #1]
;

; LD1H
define <vscale x 4 x i32> @gld1h_s_uxtw_index(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1h_s_uxtw_index:
; CHECK: ld1h { z0.s }, p0/z, [x0, z0.s, uxtw #1]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv4i16(<vscale x 4 x i1> %pg,
                                                                                  i16* %base,
                                                                                  <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @gld1h_s_sxtw_index(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1h_s_sxtw_index:
; CHECK: ld1h { z0.s }, p0/z, [x0, z0.s, sxtw #1]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv4i16(<vscale x 4 x i1> %pg,
                                                                                  i16* %base,
                                                                                  <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gld1h_d_uxtw_index(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1h_d_uxtw_index:
; CHECK: ld1h { z0.d }, p0/z, [x0, z0.d, uxtw #1]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                                  i16* %base,
                                                                                  <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1h_d_sxtw_index(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1h_d_sxtw_index:
; CHECK: ld1h { z0.d }, p0/z, [x0, z0.d, sxtw #1]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                                  i16* %base,
                                                                                  <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LD1W
define <vscale x 4 x i32> @gld1w_s_uxtw_index(<vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1w_s_uxtw_index:
; CHECK: ld1w { z0.s }, p0/z, [x0, z0.s, uxtw #2]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                  i32* %base,
                                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %load
}

define <vscale x 4 x i32> @gld1w_s_sxtw_index(<vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1w_s_sxtw_index:
; CHECK: ld1w { z0.s }, p0/z, [x0, z0.s, sxtw #2]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                  i32* %base,
                                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %load
}

define <vscale x 2 x i64> @gld1w_d_uxtw_index(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1w_d_uxtw_index:
; CHECK: ld1w { z0.d }, p0/z, [x0, z0.d, uxtw #2]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                                  i32* %base,
                                                                                  <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1w_d_sxtw_index(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1w_d_sxtw_index:
; CHECK: ld1w { z0.d }, p0/z, [x0, z0.d, sxtw #2]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                                  i32* %base,
                                                                                  <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x float> @gld1w_s_uxtw_index_float(<vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1w_s_uxtw_index_float:
; CHECK: ld1w { z0.s }, p0/z, [x0, z0.s, uxtw #2]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv4f32(<vscale x 4 x i1> %pg,
                                                                                    float* %base,
                                                                                    <vscale x 4 x i32> %b)
  ret <vscale x 4 x float> %load
}

define <vscale x 4 x float> @gld1w_s_sxtw_index_float(<vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1w_s_sxtw_index_float:
; CHECK: ld1w { z0.s }, p0/z, [x0, z0.s, sxtw #2]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv4f32(<vscale x 4 x i1> %pg,
                                                                                    float* %base,
                                                                                    <vscale x 4 x i32> %b)
  ret <vscale x 4 x float> %load
}

; LD1D
define <vscale x 2 x i64> @gld1d_s_uxtw_index(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1d_s_uxtw_index:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d, uxtw #3]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                  i64* %base,
                                                                                  <vscale x 2 x i32> %b)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x i64> @gld1d_sxtw_index(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1d_sxtw_index:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d, sxtw #3]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                  i64* %base,
                                                                                  <vscale x 2 x i32> %b)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gld1d_uxtw_index_double(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1d_uxtw_index_double:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d, uxtw #3]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv2f64(<vscale x 2 x i1> %pg,
                                                                                     double* %base,
                                                                                     <vscale x 2 x i32> %b)
  ret <vscale x 2 x double> %load
}

define <vscale x 2 x double> @gld1d_sxtw_index_double(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1d_sxtw_index_double:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d, sxtw #3]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv2f64(<vscale x 2 x i1> %pg,
                                                                                     double* %base,
                                                                                     <vscale x 2 x i32> %b)
  ret <vscale x 2 x double> %load
}

;
; LD1SH, LD1SW, LD1SD: base + 32-bit scaled offset, sign (sxtw) or zero (uxtw)
; extended to 64 bits
;   e.g. ld1sh z0.d, p0/z, [x0, z0.d, uxtw #1]
;

; LD1SH
define <vscale x 4 x i32> @gld1sh_s_uxtw_index(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1sh_s_uxtw_index:
; CHECK: ld1sh { z0.s }, p0/z, [x0, z0.s, uxtw #1]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv4i16(<vscale x 4 x i1> %pg,
                                                                                  i16* %base,
                                                                                  <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @gld1sh_s_sxtw_index(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1sh_s_sxtw_index:
; CHECK: ld1sh { z0.s }, p0/z, [x0, z0.s, sxtw #1]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv4i16(<vscale x 4 x i1> %pg,
                                                                                  i16* %base,
                                                                                  <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gld1sh_d_uxtw_index(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1sh_d_uxtw_index:
; CHECK: ld1sh { z0.d }, p0/z, [x0, z0.d, uxtw #1]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                                  i16* %base,
                                                                                  <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sh_d_sxtw_index(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1sh_d_sxtw_index:
; CHECK: ld1sh { z0.d }, p0/z, [x0, z0.d, sxtw #1]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                                  i16* %base,
                                                                                  <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LD1SW
define <vscale x 2 x i64> @gld1sw_d_uxtw_index(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1sw_d_uxtw_index:
; CHECK: ld1sw { z0.d }, p0/z, [x0, z0.d, uxtw #2]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                                  i32* %base,
                                                                                  <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sw_d_sxtw_index(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1sw_d_sxtw_index:
; CHECK: ld1sw { z0.d }, p0/z, [x0, z0.d, sxtw #2]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                                  i32* %base,
                                                                                  <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}


; LD1H/LD1SH
declare <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv4i16(<vscale x 4 x i1>, i16*, <vscale x 4 x i32>)
declare <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv4i16(<vscale x 4 x i1>, i16*, <vscale x 4 x i32>)

declare <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i32>)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i32>)

; LD1W/LD1SW
declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv4i32(<vscale x 4 x i1>, i32*, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv4i32(<vscale x 4 x i1>, i32*, <vscale x 4 x i32>)

declare <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i32>)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i32>)

declare <vscale x 4 x float> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv4f32(<vscale x 4 x i1>, float*, <vscale x 4 x i32>)
declare <vscale x 4 x float> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv4f32(<vscale x 4 x i1>, float*, <vscale x 4 x i32>)

; LD1D
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i32>)

declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i32>)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i32>)
