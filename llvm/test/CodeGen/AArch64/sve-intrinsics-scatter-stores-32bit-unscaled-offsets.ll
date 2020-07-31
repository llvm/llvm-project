; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; ST1B, ST1W, ST1H, ST1D: base + 32-bit unscaled offset, sign (sxtw) or zero
; (uxtw) extended to 64 bits.
;   e.g. st1h { z0.d }, p0, [x0, z1.d, uxtw]
;

; ST1B
define void @sst1b_s_uxtw(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i8* %base, <vscale x 4 x i32> %offsets) {
; CHECK-LABEL: sst1b_s_uxtw:
; CHECK: st1b { z0.s }, p0, [x0, z1.s, uxtw]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  call void  @llvm.aarch64.sve.st1.scatter.uxtw.nxv4i8(<vscale x 4 x i8> %data_trunc,
                                                       <vscale x 4 x i1> %pg,
                                                       i8* %base,
                                                       <vscale x 4 x i32> %offsets)
  ret void
}

define void @sst1b_s_sxtw(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i8* %base, <vscale x 4 x i32> %offsets) {
; CHECK-LABEL: sst1b_s_sxtw:
; CHECK: st1b { z0.s }, p0, [x0, z1.s, sxtw]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  call void @llvm.aarch64.sve.st1.scatter.sxtw.nxv4i8(<vscale x 4 x i8> %data_trunc,
                                                      <vscale x 4 x i1> %pg,
                                                      i8* %base,
                                                      <vscale x 4 x i32> %offsets)
  ret void
}

define void @sst1b_d_uxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i32> %offsets) {
; CHECK-LABEL: sst1b_d_uxtw:
; CHECK: st1b { z0.d }, p0, [x0, z1.d, uxtw]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  call void @llvm.aarch64.sve.st1.scatter.uxtw.nxv2i8(<vscale x 2 x i8> %data_trunc,
                                                      <vscale x 2 x i1> %pg,
                                                      i8* %base,
                                                      <vscale x 2 x i32> %offsets)
  ret void
}

define void @sst1b_d_sxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i32> %offsets) {
; CHECK-LABEL: sst1b_d_sxtw:
; CHECK: st1b { z0.d }, p0, [x0, z1.d, sxtw]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  call void  @llvm.aarch64.sve.st1.scatter.sxtw.nxv2i8(<vscale x 2 x i8> %data_trunc,
                                                       <vscale x 2 x i1> %pg,
                                                       i8* %base,
                                                       <vscale x 2 x i32> %offsets)
  ret void
}

; ST1H
define void @sst1h_s_uxtw(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %offsets) {
; CHECK-LABEL: sst1h_s_uxtw:
; CHECK: st1h { z0.s }, p0, [x0, z1.s, uxtw]
; CHECK-NEXT:	ret
  %data_trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i16>
  call void @llvm.aarch64.sve.st1.scatter.uxtw.nxv4i16(<vscale x 4 x i16> %data_trunc,
                                                       <vscale x 4 x i1> %pg,
                                                       i16* %base,
                                                       <vscale x 4 x i32> %offsets)
  ret void
}

define void @sst1h_s_sxtw(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %offsets) {
; CHECK-LABEL: sst1h_s_sxtw:
; CHECK: st1h { z0.s }, p0, [x0, z1.s, sxtw]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i16>
  call void @llvm.aarch64.sve.st1.scatter.sxtw.nxv4i16(<vscale x 4 x i16> %data_trunc,
                                                       <vscale x 4 x i1> %pg,
                                                       i16* %base,
                                                       <vscale x 4 x i32> %offsets)
  ret void
}

define void @sst1h_d_uxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %offsets) {
; CHECK-LABEL: sst1h_d_uxtw:
; CHECK: st1h { z0.d }, p0, [x0, z1.d, uxtw]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  call void @llvm.aarch64.sve.st1.scatter.uxtw.nxv2i16(<vscale x 2 x i16> %data_trunc,
                                                       <vscale x 2 x i1> %pg,
                                                       i16* %base,
                                                       <vscale x 2 x i32> %offsets)
  ret void
}

define void @sst1h_d_sxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %offsets) {
; CHECK-LABEL: sst1h_d_sxtw:
; CHECK: st1h { z0.d }, p0, [x0, z1.d, sxtw]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  call void @llvm.aarch64.sve.st1.scatter.sxtw.nxv2i16(<vscale x 2 x i16> %data_trunc,
                                                       <vscale x 2 x i1> %pg,
                                                       i16* %base,
                                                       <vscale x 2 x i32> %offsets)
  ret void
}

; ST1W
define void @sst1w_s_uxtw(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %offsets) {
; CHECK-LABEL: sst1w_s_uxtw:
; CHECK: st1w { z0.s }, p0, [x0, z1.s, uxtw]
; CHECK-NEXT:	ret
  call void @llvm.aarch64.sve.st1.scatter.uxtw.nxv4i32(<vscale x 4 x i32> %data,
                                                       <vscale x 4 x i1> %pg,
                                                       i32* %base,
                                                       <vscale x 4 x i32> %offsets)
  ret void
}

define void @sst1w_s_sxtw(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %offsets) {
; CHECK-LABEL: sst1w_s_sxtw:
; CHECK: st1w { z0.s }, p0, [x0, z1.s, sxtw]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.sxtw.nxv4i32(<vscale x 4 x i32> %data,
                                                       <vscale x 4 x i1> %pg,
                                                       i32* %base,
                                                       <vscale x 4 x i32> %offsets)
  ret void
}

define void @sst1w_d_uxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %offsets) {
; CHECK-LABEL: sst1w_d_uxtw:
; CHECK: st1w { z0.d }, p0, [x0, z1.d, uxtw]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  call void @llvm.aarch64.sve.st1.scatter.uxtw.nxv2i32(<vscale x 2 x i32> %data_trunc,
                                                       <vscale x 2 x i1> %pg,
                                                       i32* %base,
                                                       <vscale x 2 x i32> %offsets)
  ret void
}

define void @sst1w_d_sxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %offsets) {
; CHECK-LABEL: sst1w_d_sxtw:
; CHECK: st1w { z0.d }, p0, [x0, z1.d, sxtw]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  call void @llvm.aarch64.sve.st1.scatter.sxtw.nxv2i32(<vscale x 2 x i32> %data_trunc,
                                                       <vscale x 2 x i1> %pg,
                                                       i32* %base,
                                                       <vscale x 2 x i32> %offsets)
  ret void
}

define void @sst1w_s_uxtw_float(<vscale x 4 x float> %data, <vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %offsets) {
; CHECK-LABEL: sst1w_s_uxtw_float:
; CHECK: st1w { z0.s }, p0, [x0, z1.s, uxtw]
; CHECK-NEXT:	ret
  call void @llvm.aarch64.sve.st1.scatter.uxtw.nxv4f32(<vscale x 4 x float> %data,
                                                       <vscale x 4 x i1> %pg,
                                                       float* %base,
                                                       <vscale x 4 x i32> %offsets)
  ret void
}

define void @sst1w_s_sxtw_float(<vscale x 4 x float> %data, <vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %offsets) {
; CHECK-LABEL: sst1w_s_sxtw_float:
; CHECK: st1w { z0.s }, p0, [x0, z1.s, sxtw]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.sxtw.nxv4f32(<vscale x 4 x float> %data,
                                                       <vscale x 4 x i1> %pg,
                                                       float* %base,
                                                       <vscale x 4 x i32> %offsets)
  ret void
}

; ST1D
define void @sst1d_d_uxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i32> %offsets) {
; CHECK-LABEL: sst1d_d_uxtw:
; CHECK: st1d { z0.d }, p0, [x0, z1.d, uxtw]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.uxtw.nxv2i64(<vscale x 2 x i64> %data,
                                                       <vscale x 2 x i1> %pg,
                                                       i64* %base,
                                                       <vscale x 2 x i32> %offsets)
  ret void
}

define void @sst1d_d_sxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i32> %offsets) {
; CHECK-LABEL: sst1d_d_sxtw:
; CHECK: st1d { z0.d }, p0, [x0, z1.d, sxtw]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.sxtw.nxv2i64(<vscale x 2 x i64> %data,
                                                       <vscale x 2 x i1> %pg,
                                                       i64* %base,
                                                       <vscale x 2 x i32> %offsets)
  ret void
}

define void @sst1d_d_uxtw_double(<vscale x 2 x double> %data, <vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i32> %offsets) {
; CHECK-LABEL: sst1d_d_uxtw_double:
; CHECK: st1d { z0.d }, p0, [x0, z1.d, uxtw]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.uxtw.nxv2f64(<vscale x 2 x double> %data,
                                                       <vscale x 2 x i1> %pg,
                                                       double* %base,
                                                       <vscale x 2 x i32> %offsets)
  ret void
}

define void @sst1d_d_sxtw_double(<vscale x 2 x double> %data, <vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i32> %offsets) {
; CHECK-LABEL: sst1d_d_sxtw_double:
; CHECK: st1d { z0.d }, p0, [x0, z1.d, sxtw]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.sxtw.nxv2f64(<vscale x 2 x double> %data,
                                                       <vscale x 2 x i1> %pg,
                                                       double* %base,
                                                       <vscale x 2 x i32> %offsets)
  ret void
}


; ST1B
declare void @llvm.aarch64.sve.st1.scatter.uxtw.nxv4i8(<vscale x 4 x i8>, <vscale x 4 x i1>, i8*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.nxv2i8(<vscale x 2 x i8>, <vscale x 2 x i1>, i8*, <vscale x 2 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.sxtw.nxv4i8(<vscale x 4 x i8>, <vscale x 4 x i1>, i8*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.sxtw.nxv2i8(<vscale x 2 x i8>, <vscale x 2 x i1>, i8*, <vscale x 2 x i32>)

; ST1H
declare void @llvm.aarch64.sve.st1.scatter.sxtw.nxv4i16(<vscale x 4 x i16>, <vscale x 4 x i1>, i16*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.sxtw.nxv2i16(<vscale x 2 x i16>, <vscale x 2 x i1>, i16*, <vscale x 2 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.nxv4i16(<vscale x 4 x i16>, <vscale x 4 x i1>, i16*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.nxv2i16(<vscale x 2 x i16>, <vscale x 2 x i1>, i16*, <vscale x 2 x i32>)

; ST1W
declare void @llvm.aarch64.sve.st1.scatter.sxtw.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.sxtw.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i1>, i32*, <vscale x 2 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i1>, i32*, <vscale x 2 x i32>)

declare void @llvm.aarch64.sve.st1.scatter.sxtw.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, float*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, float*, <vscale x 4 x i32>)

; ST1D
declare void @llvm.aarch64.sve.st1.scatter.sxtw.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64*, <vscale x 2 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64*, <vscale x 2 x i32>)

declare void @llvm.aarch64.sve.st1.scatter.sxtw.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double*, <vscale x 2 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double*, <vscale x 2 x i32>)
