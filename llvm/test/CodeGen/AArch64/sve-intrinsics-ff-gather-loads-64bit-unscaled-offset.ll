; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; LDFF1B, LDFF1W, LDFF1H, LDFF1D: base + 64-bit unscaled offset
;   e.g. ldff1h { z0.d }, p0/z, [x0, z0.d]
;

define <vscale x 2 x i64> @gldff1b_d(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldff1b_d:
; CHECK: ldff1b { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.nxv2i8(<vscale x 2 x i1> %pg,
                                                                       i8* %base,
                                                                       <vscale x 2 x i64> %b)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1h_d(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldff1h_d:
; CHECK: ldff1h { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.nxv2i16(<vscale x 2 x i1> %pg,
                                                                         i16* %base,
                                                                         <vscale x 2 x i64> %b)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1w_d(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: gldff1w_d:
; CHECK: ldff1w { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.nxv2i32(<vscale x 2 x i1> %pg,
                                                                         i32* %base,
                                                                         <vscale x 2 x i64> %offsets)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1d_d(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldff1d_d:
; CHECK: ldff1d { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.nxv2i64(<vscale x 2 x i1> %pg,
                                                                         i64* %base,
                                                                         <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gldff1d_d_double(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldff1d_d_double:
; CHECK: ldff1d { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.nxv2f64(<vscale x 2 x i1> %pg,
                                                                            double* %base,
                                                                            <vscale x 2 x i64> %b)
  ret <vscale x 2 x double> %load
}

;
; LDFF1SB, LDFF1SW, LDFF1SH: base + 64-bit unscaled offset
;   e.g. ldff1sh { z0.d }, p0/z, [x0, z0.d]
;

define <vscale x 2 x i64> @gldff1sb_d(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldff1sb_d:
; CHECK: ldff1sb { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.nxv2i8(<vscale x 2 x i1> %pg,
                                                                       i8* %base,
                                                                       <vscale x 2 x i64> %b)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1sh_d(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldff1sh_d:
; CHECK: ldff1sh { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.nxv2i16(<vscale x 2 x i1> %pg,
                                                                         i16* %base,
                                                                         <vscale x 2 x i64> %b)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1sw_d(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: gldff1sw_d:
; CHECK: ldff1sw { z0.d }, p0/z, [x0, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.nxv2i32(<vscale x 2 x i1> %pg,
                                                                         i32* %base,
                                                                         <vscale x 2 x i64> %offsets)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

declare <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.nxv2i8(<vscale x 2 x i1>, i8*, <vscale x 2 x i64>)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i64>)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i64>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i64>)
declare <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i64>)
