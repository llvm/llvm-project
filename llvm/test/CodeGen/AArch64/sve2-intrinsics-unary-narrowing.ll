; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme < %s | FileCheck %s

;
; SQXTNB
;

define <vscale x 16 x i8> @sqxtnb_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqxtnb_h:
; CHECK: sqxtnb z0.b, z0.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqxtnb.nxv8i16(<vscale x 8 x i16> %a)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqxtnb_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqxtnb_s:
; CHECK: sqxtnb z0.h, z0.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqxtnb.nxv4i32(<vscale x 4 x i32> %a)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqxtnb_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqxtnb_d:
; CHECK: sqxtnb z0.s, z0.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqxtnb.nxv2i64(<vscale x 2 x i64> %a)
  ret <vscale x 4 x i32> %out
}

;
; UQXTNB
;

define <vscale x 16 x i8> @uqxtnb_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqxtnb_h:
; CHECK: uqxtnb z0.b, z0.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uqxtnb.nxv8i16(<vscale x 8 x i16> %a)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uqxtnb_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqxtnb_s:
; CHECK: uqxtnb z0.h, z0.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqxtnb.nxv4i32(<vscale x 4 x i32> %a)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uqxtnb_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqxtnb_d:
; CHECK: uqxtnb z0.s, z0.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqxtnb.nxv2i64(<vscale x 2 x i64> %a)
  ret <vscale x 4 x i32> %out
}

;
; SQXTUNB
;

define <vscale x 16 x i8> @sqxtunb_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqxtunb_h:
; CHECK: sqxtunb z0.b, z0.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqxtunb.nxv8i16(<vscale x 8 x i16> %a)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqxtunb_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqxtunb_s:
; CHECK: sqxtunb z0.h, z0.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqxtunb.nxv4i32(<vscale x 4 x i32> %a)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqxtunb_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqxtunb_d:
; CHECK: sqxtunb z0.s, z0.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqxtunb.nxv2i64(<vscale x 2 x i64> %a)
  ret <vscale x 4 x i32> %out
}

;
; SQXTNT
;

define <vscale x 16 x i8> @sqxtnt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqxtnt_h:
; CHECK: sqxtnt z0.b, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqxtnt.nxv8i16(<vscale x 16 x i8> %a,
                                                             <vscale x 8 x i16> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqxtnt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqxtnt_s:
; CHECK: sqxtnt z0.h, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqxtnt.nxv4i32(<vscale x 8 x i16> %a,
                                                             <vscale x 4 x i32> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqxtnt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqxtnt_d:
; CHECK: sqxtnt z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqxtnt.nxv2i64(<vscale x 4 x i32> %a,
                                                             <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

;
; UQXTNT
;

define <vscale x 16 x i8> @uqxtnt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uqxtnt_h:
; CHECK: uqxtnt z0.b, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uqxtnt.nxv8i16(<vscale x 16 x i8> %a,
                                                             <vscale x 8 x i16> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uqxtnt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uqxtnt_s:
; CHECK: uqxtnt z0.h, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqxtnt.nxv4i32(<vscale x 8 x i16> %a,
                                                             <vscale x 4 x i32> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uqxtnt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uqxtnt_d:
; CHECK: uqxtnt z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqxtnt.nxv2i64(<vscale x 4 x i32> %a,
                                                             <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

;
; SQXTUNT
;

define <vscale x 16 x i8> @sqxtunt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqxtunt_h:
; CHECK: sqxtunt z0.b, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqxtunt.nxv8i16(<vscale x 16 x i8> %a,
                                                              <vscale x 8 x i16> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqxtunt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqxtunt_s:
; CHECK: sqxtunt z0.h, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqxtunt.nxv4i32(<vscale x 8 x i16> %a,
                                                              <vscale x 4 x i32> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqxtunt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqxtunt_d:
; CHECK: sqxtunt z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqxtunt.nxv2i64(<vscale x 4 x i32> %a,
                                                              <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqxtnb.nxv8i16(<vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqxtnb.nxv4i32(<vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqxtnb.nxv2i64(<vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.uqxtnb.nxv8i16(<vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqxtnb.nxv4i32(<vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqxtnb.nxv2i64(<vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqxtunb.nxv8i16(<vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqxtunb.nxv4i32(<vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqxtunb.nxv2i64(<vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqxtnt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqxtnt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqxtnt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.uqxtnt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqxtnt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqxtnt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqxtunt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqxtunt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqxtunt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>)
