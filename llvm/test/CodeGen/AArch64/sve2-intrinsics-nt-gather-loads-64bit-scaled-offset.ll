; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; LDNT1H, LDNT1W, LDNT1D: base + 64-bit index
;   e.g.
;     lsl z0.d, z0.d, #1
;     ldnt1h z0.d, p0/z, [z0.d, x0]
;

define <vscale x 2 x i64> @gldnt1h_index(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldnt1h_index
; CHECK:        lsl z0.d, z0.d, #1
; CHECK-NEXT:   ldnt1h  { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:   ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldnt1.gather.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                               i16* %base,
                                                                               <vscale x 2 x i64> %b)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldnt1w_index(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldnt1w_index
; CHECK:        lsl z0.d, z0.d, #2
; CHECK-NEXT:   ldnt1w  { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:   ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldnt1.gather.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                               i32* %base,
                                                                               <vscale x 2 x i64> %b)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldnt1d_index(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldnt1d_index
; CHECK:        lsl z0.d, z0.d, #3
; CHECK-NEXT:   ldnt1d  { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:   ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.index.nxv2i64(<vscale x 2 x i1> %pg,
                                                                               i64* %base,
                                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gldnt1d_index_double(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldnt1d_index_double
; CHECK:        lsl z0.d, z0.d, #3
; CHECK-NEXT:   ldnt1d  { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:   ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.gather.index.nxv2f64(<vscale x 2 x i1> %pg,
                                                                                  double* %base,
                                                                                  <vscale x 2 x i64> %b)
  ret <vscale x 2 x double> %load
}

;
; LDNT1SH, LDNT1SW: base + 64-bit index
;   e.g.
;     lsl z0.d, z0.d, #1
;     ldnt1sh z0.d, p0/z, [z0.d, x0]
;

define <vscale x 2 x i64> @gldnt1sh_index(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldnt1sh_index
; CHECK:        lsl z0.d, z0.d, #1
; CHECK-NEXT:   ldnt1sh { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:   ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldnt1.gather.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                               i16* %base,
                                                                               <vscale x 2 x i64> %b)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldnt1sw_index(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gldnt1sw_index
; CHECK:        lsl z0.d, z0.d, #2
; CHECK-NEXT:   ldnt1sw { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:   ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldnt1.gather.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                               i32* %base,
                                                                               <vscale x 2 x i64> %b)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

declare <vscale x 2 x i16> @llvm.aarch64.sve.ldnt1.gather.index.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i64>)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ldnt1.gather.index.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i64>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.index.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i64>)
declare <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.gather.index.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i64>)
