; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; LD1B, LD1W, LD1H, LD1D: base + 64-bit unscaled offset
;   e.g. ld1h { z0.d }, p0/z, [x0, z0.d]
;

define <vscale x 2 x i64> @gld1b_d(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1b_d:
; CHECK: ld1b { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT:	mov	w8, #255
; CHECK-NEXT: mov	z1.d, x8
; CHECK-NEXT: and	z0.d, z0.d, z1.d
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
; CHECK-NEXT:	mov	w8, #65535
; CHECK-NEXT: mov	z1.d, x8
; CHECK-NEXT: and	z0.d, z0.d, z1.d
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
; CHECK-NEXT:	mov	w8, #-1
; CHECK-NEXT: mov	z1.d, x8
; CHECK-NEXT: and	z0.d, z0.d, z1.d
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

declare <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.nxv2i8(<vscale x 2 x i1>, i8*, <vscale x 2 x i64>)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i64>)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i64>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i64>)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i64>)
