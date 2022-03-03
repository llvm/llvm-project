; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+streaming-sve -asm-verbose=0 < %s | FileCheck %s

;
; EORBT
;

define <vscale x 16 x i8> @eorbt_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: eorbt_i8:
; CHECK: eorbt z0.b, z1.b, z2.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.eorbt.nxv16i8(<vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b,
                                                                 <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @eorbt_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: eorbt_i16:
; CHECK: eorbt z0.h, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.eorbt.nxv8i16(<vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b,
                                                                 <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @eorbt_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: eorbt_i32:
; CHECK: eorbt z0.s, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.eorbt.nxv4i32(<vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b,
                                                                 <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @eorbt_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: eorbt_i64:
; CHECK: eorbt z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.eorbt.nxv2i64(<vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b,
                                                                 <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %out
}

;
; EORTB
;

define <vscale x 16 x i8> @eortb_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: eortb_i8:
; CHECK: eortb z0.b, z1.b, z2.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.eortb.nxv16i8(<vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b,
                                                                 <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @eortb_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: eortb_i16:
; CHECK: eortb z0.h, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.eortb.nxv8i16(<vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b,
                                                                 <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @eortb_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: eortb_i32:
; CHECK: eortb z0.s, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.eortb.nxv4i32(<vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b,
                                                                 <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @eortb_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: eortb_i64:
; CHECK: eortb z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.eortb.nxv2i64(<vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b,
                                                                 <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %out
}

;
; PMULLB
;

define <vscale x 16 x i8> @pmullb_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: pmullb_i8:
; CHECK: pmullb z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.pmullb.pair.nxv16i8(<vscale x 16 x i8> %a,
                                                                       <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 4 x i32> @pmullb_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: pmullb_i32:
; CHECK: pmullb z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.pmullb.pair.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

;
; PMULLT
;

define <vscale x 16 x i8> @pmullt_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: pmullt_i8:
; CHECK: pmullt z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.pmullt.pair.nxv16i8(<vscale x 16 x i8> %a,
                                                                       <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 4 x i32> @pmullt_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: pmullt_i32:
; CHECK: pmullt z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.pmullt.pair.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.eorbt.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.eorbt.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.eorbt.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.eorbt.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.eortb.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.eortb.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.eortb.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.eortb.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.pmullb.pair.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.pmullb.pair.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.pmullt.pair.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.pmullt.pair.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
