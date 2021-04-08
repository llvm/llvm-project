; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve < %s | FileCheck %s

;
; Masked Loads
;

define <vscale x 2 x i64> @masked_sload_nxv2i8(<vscale x 2 x i8> *%a, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_sload_nxv2i8:
; CHECK: ld1sb { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.masked.load.nxv2i8(<vscale x 2 x i8> *%a, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i8> undef)
  %ext = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 2 x i64> @masked_sload_nxv2i16(<vscale x 2 x i16> *%a, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_sload_nxv2i16:
; CHECK: ld1sh { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.masked.load.nxv2i16(<vscale x 2 x i16> *%a, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i16> undef)
  %ext = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 2 x i64> @masked_sload_nxv2i32(<vscale x 2 x i32> *%a, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_sload_nxv2i32:
; CHECK: ld1sw { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32> *%a, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i32> undef)
  %ext = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 4 x i32> @masked_sload_nxv4i8(<vscale x 4 x i8> *%a, <vscale x 4 x i1> %mask) {
; CHECK-LABEL: masked_sload_nxv4i8:
; CHECK: ld1sb { [[IN:z[0-9]+]].s }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.masked.load.nxv4i8(<vscale x 4 x i8> *%a, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x i8> undef)
  %ext = sext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %ext
}

define <vscale x 4 x i32> @masked_sload_nxv4i16(<vscale x 4 x i16> *%a, <vscale x 4 x i1> %mask) {
; CHECK-LABEL: masked_sload_nxv4i16:
; CHECK: ld1sh { [[IN:z[0-9]+]].s }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.masked.load.nxv4i16(<vscale x 4 x i16> *%a, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x i16> undef)
  %ext = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %ext
}

define <vscale x 8 x i16> @masked_sload_nxv8i8(<vscale x 8 x i8> *%a, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: masked_sload_nxv8i8:
; CHECK: ld1sb { [[IN:z[0-9]+]].h }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x i8> @llvm.masked.load.nxv8i8(<vscale x 8 x i8> *%a, i32 1, <vscale x 8 x i1> %mask, <vscale x 8 x i8> undef)
  %ext = sext <vscale x 8 x i8> %load to <vscale x 8 x i16>
  ret <vscale x 8 x i16> %ext
}

declare <vscale x 2 x i8> @llvm.masked.load.nxv2i8(<vscale x 2 x i8>*, i32, <vscale x 2 x i1>, <vscale x 2 x i8>)
declare <vscale x 2 x i16> @llvm.masked.load.nxv2i16(<vscale x 2 x i16>*, i32, <vscale x 2 x i1>, <vscale x 2 x i16>)
declare <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32>*, i32, <vscale x 2 x i1>, <vscale x 2 x i32>)
declare <vscale x 4 x i8> @llvm.masked.load.nxv4i8(<vscale x 4 x i8>*, i32, <vscale x 4 x i1>, <vscale x 4 x i8>)
declare <vscale x 4 x i16> @llvm.masked.load.nxv4i16(<vscale x 4 x i16>*, i32, <vscale x 4 x i1>, <vscale x 4 x i16>)
declare <vscale x 8 x i8> @llvm.masked.load.nxv8i8(<vscale x 8 x i8>*, i32, <vscale x 8 x i1>, <vscale x 8 x i8>)
