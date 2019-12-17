; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; ST1B, ST1W, ST1H, ST1D: vector + immediate (index)
;   e.g. st1h { z0.s }, p0, [z1.s, #16]
;

; ST1B
define void @sst1b_s_imm(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: sst1b_s_imm:
; CHECK: st1b { z0.s }, p0, [z1.s, #16]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  call void @llvm.aarch64.sve.st1.scatter.imm.nxv4i8.nxv4i32(<vscale x 4 x i8> %data_trunc,
                                                             <vscale x 4 x i1> %pg,
                                                             <vscale x 4 x i32> %base,
                                                             i64 16)
  ret void
}

define void @sst1b_d_imm(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: sst1b_d_imm:
; CHECK: st1b { z0.d }, p0, [z1.d, #16]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  call void @llvm.aarch64.sve.st1.scatter.imm.nxv2i8.nxv2i64(<vscale x 2 x i8> %data_trunc,
                                                             <vscale x 2 x i1> %pg,
                                                             <vscale x 2 x i64> %base,
                                                             i64 16)
  ret void
}

; ST1H
define void @sst1h_s_imm(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: sst1h_s_imm:
; CHECK: st1h { z0.s }, p0, [z1.s, #16]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i16>
  call void @llvm.aarch64.sve.st1.scatter.imm.nxv4i16.nxv4i32(<vscale x 4 x i16> %data_trunc,
                                                              <vscale x 4 x i1> %pg,
                                                              <vscale x 4 x i32> %base,
                                                              i64 16)
  ret void
}

define void @sst1h_d_imm(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: sst1h_d_imm:
; CHECK: st1h { z0.d }, p0, [z1.d, #16]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  call void @llvm.aarch64.sve.st1.scatter.imm.nxv2i16.nxv2i64(<vscale x 2 x i16> %data_trunc,
                                                              <vscale x 2 x i1> %pg,
                                                              <vscale x 2 x i64> %base,
                                                              i64 16)
  ret void
}

; ST1W
define void @sst1w_s_imm(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: sst1w_s_imm:
; CHECK: st1w { z0.s }, p0, [z1.s, #16]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.imm.nxv4i32.nxv4i32(<vscale x 4 x i32> %data,
                                                              <vscale x 4 x i1> %pg,
                                                              <vscale x 4 x i32> %base,
                                                              i64 16)
  ret void
}

define void @sst1w_d_imm(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: sst1w_d_imm:
; CHECK: st1w { z0.d }, p0, [z1.d, #16]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  call void @llvm.aarch64.sve.st1.scatter.imm.nxv2i32.nxv2i64(<vscale x 2 x i32> %data_trunc,
                                                              <vscale x 2 x i1> %pg,
                                                              <vscale x 2 x i64> %base,
                                                              i64 16)
  ret void
}

define void @sst1w_s_imm_float(<vscale x 4 x float> %data, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: sst1w_s_imm_float:
; CHECK: st1w { z0.s }, p0, [z1.s, #16]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.imm.nxv4f32.nxv4i32(<vscale x 4 x float> %data,
                                                              <vscale x 4 x i1> %pg,
                                                              <vscale x 4 x i32> %base,
                                                              i64 16)
  ret void
}

; ST1D
define void @sst1d_d_imm(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: sst1d_d_imm:
; CHECK: st1d { z0.d }, p0, [z1.d, #16]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.imm.nxv2i64.nxv2i64(<vscale x 2 x i64> %data,
                                                              <vscale x 2 x i1> %pg,
                                                              <vscale x 2 x i64> %base,
                                                              i64 16)
  ret void
}

define void @sst1d_d_imm_double(<vscale x 2 x double> %data, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: sst1d_d_imm_double:
; CHECK: st1d { z0.d }, p0, [z1.d, #16]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.imm.nxv2f64.nxv2i64(<vscale x 2 x double> %data,
                                                              <vscale x 2 x i1> %pg,
                                                              <vscale x 2 x i64> %base,
                                                              i64 16)
  ret void
}

; ST1B
declare void @llvm.aarch64.sve.st1.scatter.imm.nxv4i8.nxv4i32(<vscale x 4 x i8>, <vscale x 4 x i1>, <vscale x 4 x i32>, i64)
declare void @llvm.aarch64.sve.st1.scatter.imm.nxv2i8.nxv2i64(<vscale x 2 x i8>, <vscale x 2 x i1>, <vscale x 2 x i64>, i64)

; ST1H
declare void @llvm.aarch64.sve.st1.scatter.imm.nxv4i16.nxv4i32(<vscale x 4 x i16>, <vscale x 4 x i1>, <vscale x 4 x i32>, i64)
declare void @llvm.aarch64.sve.st1.scatter.imm.nxv2i16.nxv2i64(<vscale x 2 x i16>, <vscale x 2 x i1>, <vscale x 2 x i64>, i64)

; ST1W
declare void @llvm.aarch64.sve.st1.scatter.imm.nxv4i32.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>, i64)
declare void @llvm.aarch64.sve.st1.scatter.imm.nxv2i32.nxv2i64(<vscale x 2 x i32>, <vscale x 2 x i1>, <vscale x 2 x i64>, i64)

declare void @llvm.aarch64.sve.st1.scatter.imm.nxv4f32.nxv4i32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x i32>, i64)

; ST1D
declare void @llvm.aarch64.sve.st1.scatter.imm.nxv2i64.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>, i64)

declare void @llvm.aarch64.sve.st1.scatter.imm.nxv2f64.nxv2i64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x i64>, i64)
