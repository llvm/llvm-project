; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve < %s | FileCheck %s

;
; Masked Loads
;

define <vscale x 2 x i64> @masked_zload_nxv2i8(<vscale x 2 x i8>* %src, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_zload_nxv2i8:
; CHECK-NOT: ld1sb
; CHECK: ld1b { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.masked.load.nxv2i8(<vscale x 2 x i8>* %src, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i8> undef)
  %ext = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 2 x i64> @masked_zload_nxv2i16(<vscale x 2 x i16>* %src, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_zload_nxv2i16:
; CHECK-NOT: ld1sh
; CHECK: ld1h { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.masked.load.nxv2i16(<vscale x 2 x i16>* %src, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i16> undef)
  %ext = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 2 x i64> @masked_zload_nxv2i32(<vscale x 2 x i32>* %src, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_zload_nxv2i32:
; CHECK-NOT: ld1sw
; CHECK: ld1w { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32>* %src, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i32> undef)
  %ext = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 4 x i32> @masked_zload_nxv4i8(<vscale x 4 x i8>* %src, <vscale x 4 x i1> %mask) {
; CHECK-LABEL: masked_zload_nxv4i8:
; CHECK-NOT: ld1sb
; CHECK: ld1b { [[IN:z[0-9]+]].s }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.masked.load.nxv4i8(<vscale x 4 x i8>* %src, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x i8> undef)
  %ext = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %ext
}

define <vscale x 4 x i32> @masked_zload_nxv4i16(<vscale x 4 x i16>* %src, <vscale x 4 x i1> %mask) {
; CHECK-LABEL: masked_zload_nxv4i16:
; CHECK-NOT: ld1sh
; CHECK: ld1h { [[IN:z[0-9]+]].s }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.masked.load.nxv4i16(<vscale x 4 x i16>* %src, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x i16> undef)
  %ext = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %ext
}

define <vscale x 8 x i16> @masked_zload_nxv8i8(<vscale x 8 x i8>* %src, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: masked_zload_nxv8i8:
; CHECK-NOT: ld1sb
; CHECK: ld1b { [[IN:z[0-9]+]].h }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x i8> @llvm.masked.load.nxv8i8(<vscale x 8 x i8>* %src, i32 1, <vscale x 8 x i1> %mask, <vscale x 8 x i8> undef)
  %ext = zext <vscale x 8 x i8> %load to <vscale x 8 x i16>
  ret <vscale x 8 x i16> %ext
}

define <vscale x 2 x i64> @masked_zload_passthru(<vscale x 2 x i32>* %src, <vscale x 2 x i1> %mask, <vscale x 2 x i32> %passthru) {
; CHECK-LABEL: masked_zload_passthru:
; CHECK-NOT: ld1sw
; CHECK: ld1w { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: and z0.d, z0.d, #0xffffffff
; CHECK-NEXT: mov z0.d, [[PG]]/m, [[IN]].d
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32>* %src, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i32> %passthru)
  %ext = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

; Return type requires splitting
define <vscale x 8 x i64> @masked_zload_nxv8i16(<vscale x 8 x i16>* %a, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: masked_zload_nxv8i16:
; CHECK:         ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    uunpklo z1.s, z0.h
; CHECK-NEXT:    uunpkhi z3.s, z0.h
; CHECK-NEXT:    uunpklo z0.d, z1.s
; CHECK-NEXT:    uunpkhi z1.d, z1.s
; CHECK-NEXT:    uunpklo z2.d, z3.s
; CHECK-NEXT:    uunpkhi z3.d, z3.s
; CHECK-NEXT:    ret
  %load = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16(<vscale x 8 x i16>* %a, i32 2, <vscale x 8 x i1> %mask, <vscale x 8 x i16> undef)
  %ext = zext <vscale x 8 x i16> %load to <vscale x 8 x i64>
  ret <vscale x 8 x i64> %ext
}

; Masked load requires promotion
define <vscale x 2 x double> @masked_zload_2i16_2f64(<vscale x 2 x i16>* noalias %in, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_zload_2i16_2f64:
; CHECK:       ld1h { z0.d }, p0/z, [x0]
; CHECK-NEXT:  ptrue p0.d
; CHECK-NEXT:  ucvtf z0.d, p0/m, z0.s
; CHECK-NEXT:  ret
  %wide.load = call <vscale x 2 x i16> @llvm.masked.load.nxv2i16(<vscale x 2 x i16>* %in, i32 2, <vscale x 2 x i1> %mask, <vscale x 2 x i16> undef)
  %zext = zext <vscale x 2 x i16> %wide.load to <vscale x 2 x i32>
  %res = uitofp <vscale x 2 x i32> %zext to <vscale x 2 x double>
  ret <vscale x 2 x double> %res
}

declare <vscale x 2 x i8> @llvm.masked.load.nxv2i8(<vscale x 2 x i8>*, i32, <vscale x 2 x i1>, <vscale x 2 x i8>)
declare <vscale x 2 x i16> @llvm.masked.load.nxv2i16(<vscale x 2 x i16>*, i32, <vscale x 2 x i1>, <vscale x 2 x i16>)
declare <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32>*, i32, <vscale x 2 x i1>, <vscale x 2 x i32>)
declare <vscale x 4 x i8> @llvm.masked.load.nxv4i8(<vscale x 4 x i8>*, i32, <vscale x 4 x i1>, <vscale x 4 x i8>)
declare <vscale x 4 x i16> @llvm.masked.load.nxv4i16(<vscale x 4 x i16>*, i32, <vscale x 4 x i1>, <vscale x 4 x i16>)
declare <vscale x 8 x i8> @llvm.masked.load.nxv8i8(<vscale x 8 x i8>*, i32, <vscale x 8 x i1>, <vscale x 8 x i8>)
declare <vscale x 8 x i16> @llvm.masked.load.nxv8i16(<vscale x 8 x i16>*, i32, <vscale x 8 x i1>, <vscale x 8 x i16>)
