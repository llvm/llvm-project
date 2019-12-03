; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; LD1B, LD1W, LD1H, LD1D: vector + immediate (index)
;   e.g. ld1h { z0.s }, p0/z, [z0.s, #16]
;

; LD1B
define <vscale x 4 x i32> @gld1b_s_imm(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gld1b_s_imm:
; CHECK: ld1b { z0.s }, p0/z, [z0.s, #16]
; CHECK-NEXT: mov	w8, #255
; CHECK-NEXT: mov	z1.s, w8
; CHECK-NEXT:	and	z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ld1.gather.imm.nxv4i8.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                 <vscale x 4 x i32> %base,
                                                                                 i64 16)
  %res = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gld1b_d_imm(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gld1b_d_imm:
; CHECK: ld1b { z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT: mov	w8, #255
; CHECK-NEXT: mov	z1.d, x8
; CHECK-NEXT:	and	z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.imm.nxv2i8.nxv2i64(<vscale x 2 x i1> %pg,
                                                                           <vscale x 2 x i64> %base,
                                                                           i64 16)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LD1H
define <vscale x 4 x i32> @gld1h_s_imm(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gld1h_s_imm:
; CHECK: ld1h { z0.s }, p0/z, [z0.s, #16]
; CHECK-NEXT: mov	w8, #65535
; CHECK-NEXT: mov	z1.s, w8
; CHECK-NEXT:	and	z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.imm.nxv4i16.nxv4i32(<vscale x 4 x i1> %pg,
                                                                            <vscale x 4 x i32> %base,
                                                                            i64 16)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gld1h_d_imm(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gld1h_d_imm:
; CHECK: ld1h { z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT: mov	w8, #65535
; CHECK-NEXT: mov	z1.d, x8
; CHECK-NEXT:	and	z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.imm.nxv2i16.nxv2i64(<vscale x 2 x i1> %pg,
                                                                           <vscale x 2 x i64> %base,
                                                                           i64 16)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LD1W
define <vscale x 4 x i32> @gld1w_s_imm(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gld1w_s_imm:
; CHECK: ld1w { z0.s }, p0/z, [z0.s, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.imm.nxv4i32.nxv4i32(<vscale x 4 x i1> %pg,
                                                                            <vscale x 4 x i32> %base,
                                                                            i64 16)
  ret <vscale x 4 x i32> %load
}

define <vscale x 2 x i64> @gld1w_d_imm(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gld1w_d_imm:
; CHECK: ld1w { z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT:	mov	w8, #-1
; CHECK-NEXT:	mov	z1.d, x8
; CHECK-NEXT:	and	z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.imm.nxv2i32.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                   <vscale x 2 x i64> %base,
                                                                                   i64 16)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x float> @gld1w_s_imm_float(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gld1w_s_imm_float:
; CHECK: ld1w { z0.s }, p0/z, [z0.s, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ld1.gather.imm.nxv4f32.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                     <vscale x 4 x i32> %base,
                                                                                     i64 16)
  ret <vscale x 4 x float> %load
}

; LD1D
define <vscale x 2 x i64> @gld1d_d_imm(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gld1d_d_imm:
; CHECK: ld1d { z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.imm.nxv2i64.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                   <vscale x 2 x i64> %base,
                                                                                   i64 16)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gld1d_d_imm_double(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gld1d_d_imm_double:
; CHECK: ld1d { z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.imm.nxv2f64.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                      <vscale x 2 x i64> %base,
                                                                                      i64 16)
  ret <vscale x 2 x double> %load
}

; LD1B
declare <vscale x 4 x i8> @llvm.aarch64.sve.ld1.gather.imm.nxv4i8.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i64)
declare <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.imm.nxv2i8.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)

; LD1H
declare <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.imm.nxv4i16.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i64)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.imm.nxv2i16.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)

; LD1W
declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.imm.nxv4i32.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i64)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.imm.nxv2i32.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)

declare <vscale x 4 x float> @llvm.aarch64.sve.ld1.gather.imm.nxv4f32.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i64)

; LD1D
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.imm.nxv2i64.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)

declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.imm.nxv2f64.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)
