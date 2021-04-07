; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s

;
; STNT1H, STNT1W, STNT1D: base + 64-bit index
;   e.g.
;     lsl z1.d, z1.d, #1
;     stnt1h { z0.d }, p0, [z0.d, x0]
;

define void @sstnt1h_index(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: sstnt1h_index
; CHECK:        lsl z1.d, z1.d, #1
; CHECK-NEXT:   stnt1h  { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT:   ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  call void @llvm.aarch64.sve.stnt1.scatter.index.nxv2i16(<vscale x 2 x i16> %data_trunc,
                                                          <vscale x 2 x i1> %pg,
                                                          i16* %base,
                                                          <vscale x 2 x i64> %offsets)
  ret void
}

define void @sstnt1w_index(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: sstnt1w_index
; CHECK:        lsl z1.d, z1.d, #2
; CHECK-NEXT:   stnt1w  { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT:   ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  call void @llvm.aarch64.sve.stnt1.scatter.index.nxv2i32(<vscale x 2 x i32> %data_trunc,
                                                          <vscale x 2 x i1> %pg,
                                                          i32* %base,
                                                          <vscale x 2 x i64> %offsets)
  ret void
}

define void  @sstnt1d_index(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: sstnt1d_index
; CHECK:        lsl z1.d, z1.d, #3
; CHECK-NEXT:   stnt1d  { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT:   ret
  call void @llvm.aarch64.sve.stnt1.scatter.index.nxv2i64(<vscale x 2 x i64> %data,
                                                          <vscale x 2 x i1> %pg,
                                                          i64* %base,
                                                          <vscale x 2 x i64> %offsets)
  ret void
}

define void  @sstnt1d_index_double(<vscale x 2 x double> %data, <vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i64> %offsets) {
; CHECK-LABEL: sstnt1d_index_double
; CHECK:        lsl z1.d, z1.d, #3
; CHECK-NEXT:   stnt1d  { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT:   ret
  call void @llvm.aarch64.sve.stnt1.scatter.index.nxv2f64(<vscale x 2 x double> %data,
                                                          <vscale x 2 x i1> %pg,
                                                          double* %base,
                                                          <vscale x 2 x i64> %offsets)
  ret void
}


declare void @llvm.aarch64.sve.stnt1.scatter.index.nxv2i16(<vscale x 2 x i16>, <vscale x 2 x i1>, i16*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.stnt1.scatter.index.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i1>, i32*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.stnt1.scatter.index.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.stnt1.scatter.index.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double*, <vscale x 2 x i64>)
