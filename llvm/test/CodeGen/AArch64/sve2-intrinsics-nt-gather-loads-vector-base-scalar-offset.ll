; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; LDNT1B, LDNT1W, LDNT1H, LDNT1D: vector base + scalar offset
;   ldnt1b { z0.s }, p0/z, [z0.s, x0]
;

; LDNT1B
define <vscale x 4 x i32> @gldnt1b_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base, i64 %offset) {
; CHECK-LABEL: gldnt1b_s:
; CHECK:    ldnt1b { z0.s }, p0/z, [z0.s, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                             <vscale x 4 x i32> %base,
                                                                                             i64 %offset)
  %res = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldnt1b_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base, i64 %offset) {
; CHECK-LABEL: gldnt1b_d:
; CHECK:    ldnt1b { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                             <vscale x 2 x i64> %base,
                                                                                             i64 %offset)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDNT1H
define <vscale x 4 x i32> @gldnt1h_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base, i64 %offset) {
; CHECK-LABEL: gldnt1h_s:
; CHECK:    ldnt1h { z0.s }, p0/z, [z0.s, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv416.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                              <vscale x 4 x i32> %base,
                                                                                              i64 %offset)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldnt1h_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base, i64 %offset) {
; CHECK-LABEL: gldnt1h_d:
; CHECK:    ldnt1h { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 %offset)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDNT1W
define <vscale x 4 x i32> @gldnt1w_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base, i64 %offset) {
; CHECK-LABEL: gldnt1w_s:
; CHECK:    ldnt1w { z0.s }, p0/z, [z0.s, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                               <vscale x 4 x i32> %base,
                                                                                               i64 %offset)
  ret <vscale x 4 x i32> %load
}

define <vscale x 4 x float> @gldnt1w_s_float(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base, i64 %offset) {
; CHECK-LABEL: gldnt1w_s_float:
; CHECK:    ldnt1w { z0.s }, p0/z, [z0.s, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                                 <vscale x 4 x i32> %base,
                                                                                                 i64 %offset)
  ret <vscale x 4 x float> %load
}

define <vscale x 2 x i64> @gldnt1w_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base, i64 %offset) {
; CHECK-LABEL: gldnt1w_d:
; CHECK:    ldnt1w { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 %offset)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDNT1D
define <vscale x 2 x i64> @gldnt1d_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base, i64 %offset) {
; CHECK-LABEL: gldnt1d_d:
; CHECK:    ldnt1d { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 %offset)
  ret <vscale x 2 x i64> %load
}

; LDNT1D
define <vscale x 2 x double> @gldnt1d_d_double(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base, i64 %offset) {
; CHECK-LABEL: gldnt1d_d_double:
; CHECK:    ldnt1d { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                                  <vscale x 2 x i64> %base,
                                                                                                  i64 %offset)
  ret <vscale x 2 x double> %load
}

;
; LDNT1SB, LDNT1SW, LDNT1SH, LDNT1SD: vector base + scalar offset
;   ldnt1sb { z0.s }, p0/z, [z0.s, x0]
;

; LDNT1SB
define <vscale x 4 x i32> @gldnt1sb_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base, i64 %offset) {
; CHECK-LABEL: gldnt1sb_s:
; CHECK:    ldnt1sb { z0.s }, p0/z, [z0.s, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                             <vscale x 4 x i32> %base,
                                                                                             i64 %offset)
  %res = sext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldnt1sb_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base, i64 %offset) {
; CHECK-LABEL: gldnt1sb_d:
; CHECK:    ldnt1sb { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                             <vscale x 2 x i64> %base,
                                                                                             i64 %offset)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDNT1SH
define <vscale x 4 x i32> @gldnt1sh_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base, i64 %offset) {
; CHECK-LABEL: gldnt1sh_s:
; CHECK:    ldnt1sh { z0.s }, p0/z, [z0.s, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv416.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                              <vscale x 4 x i32> %base,
                                                                                              i64 %offset)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldnt1sh_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base, i64 %offset) {
; CHECK-LABEL: gldnt1sh_d:
; CHECK:    ldnt1sh { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 %offset)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDNT1SW
define <vscale x 2 x i64> @gldnt1sw_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base, i64 %offset) {
; CHECK-LABEL: gldnt1sw_d:
; CHECK:    ldnt1sw { z0.d }, p0/z, [z0.d, x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 %offset)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDNT1B/LDNT1SB
declare <vscale x 4 x i8> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i64)
declare <vscale x 2 x i8> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)

; LDNT1H/LDNT1SH
declare <vscale x 4 x i16> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv416.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i64)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)

; LDNT1W/LDNT1SW
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i64)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)

declare <vscale x 4 x float>  @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x i1>,  <vscale x 4 x i32>, i64)

; LDNT1D
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)

declare <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)
