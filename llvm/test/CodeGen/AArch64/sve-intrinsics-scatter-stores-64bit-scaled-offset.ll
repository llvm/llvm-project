; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

;
; ST1H, ST1W, ST1D: base + 64-bit scaled offset
;   e.g. st1h { z0.d }, p0, [x0, z0.d, lsl #1]
;

define void @sst1h_index(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: sst1h_index
; CHECK:	    st1h	{ z0.d }, p0, [x0, z1.d, lsl #1]
; CHECK-NEXT:	ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  call void @llvm.aarch64.sve.st1.scatter.index.nxv2i16(<vscale x 2 x i16> %data_trunc,
                                                        <vscale x 2 x i1> %pg,
                                                        i16* %base,
                                                        <vscale x 2 x i64> %offsets)
  ret void
}

define void @sst1w_index(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: sst1w_index
; CHECK:	    st1w	{ z0.d }, p0, [x0, z1.d, lsl #2]
; CHECK-NEXT:	ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  call void @llvm.aarch64.sve.st1.scatter.index.nxv2i32(<vscale x 2 x i32> %data_trunc,
                                                        <vscale x 2 x i1> %pg,
                                                        i32* %base,
                                                        <vscale x 2 x i64> %offsets)
  ret void
}

define void  @sst1d_index(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: sst1d_index
; CHECK:	    st1d	{ z0.d }, p0, [x0, z1.d, lsl #3]
; CHECK-NEXT:	ret
  call void @llvm.aarch64.sve.st1.scatter.index.nxv2i64(<vscale x 2 x i64> %data,
                                                        <vscale x 2 x i1> %pg,
                                                        i64* %base,
                                                        <vscale x 2 x i64> %offsets)
  ret void
}

define void  @sst1d_index_double(<vscale x 2 x double> %data, <vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: sst1d_index_double
; CHECK:	    st1d	{ z0.d }, p0, [x0, z1.d, lsl #3]
; CHECK-NEXT:	ret
  call void @llvm.aarch64.sve.st1.scatter.index.nxv2f64(<vscale x 2 x double> %data,
                                                        <vscale x 2 x i1> %pg,
                                                        double* %base,
                                                        <vscale x 2 x i64> %offsets)
  ret void
}


declare void @llvm.aarch64.sve.st1.scatter.index.nxv2i16(<vscale x 2 x i16>, <vscale x 2 x i1>, i16*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.st1.scatter.index.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i1>, i32*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.st1.scatter.index.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.st1.scatter.index.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double*, <vscale x 2 x i64>)
