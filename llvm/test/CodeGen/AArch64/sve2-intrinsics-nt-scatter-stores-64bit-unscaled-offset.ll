; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; STNT1B, STNT1W, STNT1H, STNT1D: base + 64-bit unscaled offset
;   e.g. stnt1h { z0.d }, p0, [z1.d, x0]
;

define void @sstnt1b_d(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sstnt1b_d:
; CHECK: stnt1b { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  call void @llvm.aarch64.sve.stnt1.scatter.nxv2i8(<vscale x 2 x i8> %data_trunc,
                                                   <vscale x 2 x i1> %pg,
                                                   i8* %base,
                                                   <vscale x 2 x i64> %b)
  ret void
}

define void @sstnt1h_d(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sstnt1h_d:
; CHECK: stnt1h { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  call void @llvm.aarch64.sve.stnt1.scatter.nxv2i16(<vscale x 2 x i16> %data_trunc,
                                                    <vscale x 2 x i1> %pg,
                                                    i16* %base,
                                                    <vscale x 2 x i64> %b)
  ret void
}

define void @sstnt1w_d(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sstnt1w_d:
; CHECK: stnt1w { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  call void @llvm.aarch64.sve.stnt1.scatter.nxv2i32(<vscale x 2 x i32> %data_trunc,
                                                    <vscale x 2 x i1> %pg,
                                                    i32* %base,
                                                    <vscale x 2 x i64> %b)
  ret void
}

define void @sstnt1d_d(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sstnt1d_d:
; CHECK: stnt1d { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.stnt1.scatter.nxv2i64(<vscale x 2 x i64> %data,
                                                    <vscale x 2 x i1> %pg,
                                                    i64* %base,
                                                    <vscale x 2 x i64> %b)
  ret void
}

define void @sstnt1d_d_double(<vscale x 2 x double> %data, <vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sstnt1d_d_double:
; CHECK: stnt1d { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.stnt1.scatter.nxv2f64(<vscale x 2 x double> %data,
                                                    <vscale x 2 x i1> %pg,
                                                    double* %base,
                                                    <vscale x 2 x i64> %b)
  ret void
}

declare void @llvm.aarch64.sve.stnt1.scatter.nxv2i8(<vscale x 2 x i8>, <vscale x 2 x i1>, i8*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.stnt1.scatter.nxv2i16(<vscale x 2 x i16>, <vscale x 2 x i1>, i16*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.stnt1.scatter.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i1>, i32*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.stnt1.scatter.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.stnt1.scatter.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double*, <vscale x 2 x i64>)
