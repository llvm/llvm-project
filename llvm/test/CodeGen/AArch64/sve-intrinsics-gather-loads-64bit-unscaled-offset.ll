; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

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

declare <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.nxv2i8(<vscale x 2 x i1>, i8*, <vscale x 2 x i64>)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i64>)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i64>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i64>)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i64>)
