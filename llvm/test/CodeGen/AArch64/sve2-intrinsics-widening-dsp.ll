; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s

;
; SABALB
;

define <vscale x 8 x i16> @sabalb_b(<vscale x 8 x i16> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: sabalb_b:
; CHECK: sabalb z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sabalb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sabalb_h(<vscale x 4 x i32> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: sabalb_h:
; CHECK: sabalb z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sabalb.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sabalb_s(<vscale x 2 x i64> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: sabalb_s:
; CHECK: sabalb z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sabalb.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %out
}

;
; SABALT
;

define <vscale x 8 x i16> @sabalt_b(<vscale x 8 x i16> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: sabalt_b:
; CHECK: sabalt z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sabalt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sabalt_h(<vscale x 4 x i32> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: sabalt_h:
; CHECK: sabalt z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sabalt.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sabalt_s(<vscale x 2 x i64> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: sabalt_s:
; CHECK: sabalt z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sabalt.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %out
}

;
; SABDLB
;

define <vscale x 8 x i16> @sabdlb_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sabdlb_b:
; CHECK: sabdlb z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sabdlb.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sabdlb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sabdlb_h:
; CHECK: sabdlb z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sabdlb.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sabdlb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sabdlb_s:
; CHECK: sabdlb z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sabdlb.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SABDLT
;

define <vscale x 8 x i16> @sabdlt_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sabdlt_b:
; CHECK: sabdlt z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sabdlt.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sabdlt_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sabdlt_h:
; CHECK: sabdlt z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sabdlt.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sabdlt_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sabdlt_s:
; CHECK: sabdlt z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sabdlt.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SADDLB
;

define <vscale x 8 x i16> @saddlb_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: saddlb_b:
; CHECK: saddlb z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.saddlb.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @saddlb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: saddlb_h:
; CHECK: saddlb z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.saddlb.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @saddlb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: saddlb_s:
; CHECK: saddlb z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.saddlb.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SADDLT
;

define <vscale x 8 x i16> @saddlt_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: saddlt_b:
; CHECK: saddlt z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.saddlt.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @saddlt_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: saddlt_h:
; CHECK: saddlt z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.saddlt.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @saddlt_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: saddlt_s:
; CHECK: saddlt z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.saddlt.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SADDWB
;

define <vscale x 8 x i16> @saddwb_b(<vscale x 8 x i16> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: saddwb_b:
; CHECK: saddwb z0.h, z0.h, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.saddwb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @saddwb_h(<vscale x 4 x i32> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: saddwb_h:
; CHECK: saddwb z0.s, z0.s, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.saddwb.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @saddwb_s(<vscale x 2 x i64> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: saddwb_s:
; CHECK: saddwb z0.d, z0.d, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.saddwb.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SADDWT
;

define <vscale x 8 x i16> @saddwt_b(<vscale x 8 x i16> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: saddwt_b:
; CHECK: saddwt z0.h, z0.h, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.saddwt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @saddwt_h(<vscale x 4 x i32> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: saddwt_h:
; CHECK: saddwt z0.s, z0.s, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.saddwt.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @saddwt_s(<vscale x 2 x i64> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: saddwt_s:
; CHECK: saddwt z0.d, z0.d, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.saddwt.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}


;
; SMULLB (Vectors)
;

define <vscale x 8 x i16> @smullb_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: smullb_b:
; CHECK: smullb z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.smullb.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @smullb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: smullb_h:
; CHECK: smullb z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.smullb.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @smullb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: smullb_s:
; CHECK: smullb z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.smullb.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SMULLB (Indexed)
;

define <vscale x 4 x i32> @smullb_lane_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: smullb_lane_h:
; CHECK: smullb z0.s, z0.h, z1.h[4]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.smullb.lane.nxv4i32(<vscale x 8 x i16> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       i32 4)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @smullb_lane_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: smullb_lane_s:
; CHECK: smullb z0.d, z0.s, z1.s[3]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.smullb.lane.nxv2i64(<vscale x 4 x i32> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       i32 3)
  ret <vscale x 2 x i64> %out
}

;
; SMULLT (Vectors)
;

define <vscale x 8 x i16> @smullt_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: smullt_b:
; CHECK: smullt z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.smullt.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @smullt_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: smullt_h:
; CHECK: smullt z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.smullt.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @smullt_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: smullt_s:
; CHECK: smullt z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.smullt.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SMULLT (Indexed)
;

define <vscale x 4 x i32> @smullt_lane_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: smullt_lane_h:
; CHECK: smullt z0.s, z0.h, z1.h[5]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.smullt.lane.nxv4i32(<vscale x 8 x i16> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       i32 5)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @smullt_lane_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: smullt_lane_s:
; CHECK: smullt z0.d, z0.s, z1.s[2]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.smullt.lane.nxv2i64(<vscale x 4 x i32> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       i32 2)
  ret <vscale x 2 x i64> %out
}

;
; SQDMULLB (Vectors)
;

define <vscale x 8 x i16> @sqdmullb_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sqdmullb_b:
; CHECK: sqdmullb z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdmullb.nxv8i16(<vscale x 16 x i8> %a,
                                                                    <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqdmullb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqdmullb_h:
; CHECK: sqdmullb z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullb.nxv4i32(<vscale x 8 x i16> %a,
                                                                    <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqdmullb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqdmullb_s:
; CHECK: sqdmullb z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullb.nxv2i64(<vscale x 4 x i32> %a,
                                                                    <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SQDMULLB (Indexed)
;

define <vscale x 4 x i32> @sqdmullb_lane_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqdmullb_lane_h:
; CHECK: sqdmullb z0.s, z0.h, z1.h[2]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullb.lane.nxv4i32(<vscale x 8 x i16> %a,
                                                                         <vscale x 8 x i16> %b,
                                                                         i32 2)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqdmullb_lane_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqdmullb_lane_s:
; CHECK: sqdmullb z0.d, z0.s, z1.s[1]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullb.lane.nxv2i64(<vscale x 4 x i32> %a,
                                                                         <vscale x 4 x i32> %b,
                                                                         i32 1)
  ret <vscale x 2 x i64> %out
}

;
; SQDMULLT (Vectors)
;

define <vscale x 8 x i16> @sqdmullt_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sqdmullt_b:
; CHECK: sqdmullt z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdmullt.nxv8i16(<vscale x 16 x i8> %a,
                                                                    <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqdmullt_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqdmullt_h:
; CHECK: sqdmullt z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullt.nxv4i32(<vscale x 8 x i16> %a,
                                                                    <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqdmullt_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqdmullt_s:
; CHECK: sqdmullt z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullt.nxv2i64(<vscale x 4 x i32> %a,
                                                                    <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SQDMULLT (Indexed)
;

define <vscale x 4 x i32> @sqdmullt_lane_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqdmullt_lane_h:
; CHECK: sqdmullt z0.s, z0.h, z1.h[3]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullt.lane.nxv4i32(<vscale x 8 x i16> %a,
                                                                         <vscale x 8 x i16> %b,
                                                                         i32 3)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqdmullt_lane_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqdmullt_lane_s:
; CHECK: sqdmullt z0.d, z0.s, z1.s[0]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullt.lane.nxv2i64(<vscale x 4 x i32> %a,
                                                                         <vscale x 4 x i32> %b,
                                                                         i32 0)
  ret <vscale x 2 x i64> %out
}

;
; SSUBLB
;

define <vscale x 8 x i16> @ssublb_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: ssublb_b:
; CHECK: ssublb z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.ssublb.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @ssublb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: ssublb_h:
; CHECK: ssublb z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.ssublb.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @ssublb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: ssublb_s:
; CHECK: ssublb z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.ssublb.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SSHLLB
;

define <vscale x 8 x i16> @sshllb_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sshllb_b:
; CHECK: sshllb z0.h, z0.b, #0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sshllb.nxv8i16(<vscale x 16 x i8> %a, i32 0)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sshllb_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sshllb_h:
; CHECK: sshllb z0.s, z0.h, #1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sshllb.nxv4i32(<vscale x 8 x i16> %a, i32 1)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sshllb_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sshllb_s:
; CHECK: sshllb z0.d, z0.s, #2
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sshllb.nxv2i64(<vscale x 4 x i32> %a, i32 2)
  ret <vscale x 2 x i64> %out
}

;
; SSHLLT
;

define <vscale x 8 x i16> @sshllt_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sshllt_b:
; CHECK: sshllt z0.h, z0.b, #3
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sshllt.nxv8i16(<vscale x 16 x i8> %a, i32 3)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sshllt_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sshllt_h:
; CHECK: sshllt z0.s, z0.h, #4
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sshllt.nxv4i32(<vscale x 8 x i16> %a, i32 4)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sshllt_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sshllt_s:
; CHECK: sshllt z0.d, z0.s, #5
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sshllt.nxv2i64(<vscale x 4 x i32> %a, i32 5)
  ret <vscale x 2 x i64> %out
}

;
; SSUBLT
;

define <vscale x 8 x i16> @ssublt_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: ssublt_b:
; CHECK: ssublt z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.ssublt.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @ssublt_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: ssublt_h:
; CHECK: ssublt z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.ssublt.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @ssublt_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: ssublt_s:
; CHECK: ssublt z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.ssublt.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SSUBWB
;

define <vscale x 8 x i16> @ssubwb_b(<vscale x 8 x i16> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: ssubwb_b:
; CHECK: ssubwb z0.h, z0.h, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.ssubwb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @ssubwb_h(<vscale x 4 x i32> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: ssubwb_h:
; CHECK: ssubwb z0.s, z0.s, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.ssubwb.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @ssubwb_s(<vscale x 2 x i64> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: ssubwb_s:
; CHECK: ssubwb z0.d, z0.d, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.ssubwb.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SSUBWT
;

define <vscale x 8 x i16> @ssubwt_b(<vscale x 8 x i16> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: ssubwt_b:
; CHECK: ssubwt z0.h, z0.h, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.ssubwt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @ssubwt_h(<vscale x 4 x i32> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: ssubwt_h:
; CHECK: ssubwt z0.s, z0.s, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.ssubwt.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @ssubwt_s(<vscale x 2 x i64> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: ssubwt_s:
; CHECK: ssubwt z0.d, z0.d, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.ssubwt.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; UABALB
;

define <vscale x 8 x i16> @uabalb_b(<vscale x 8 x i16> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: uabalb_b:
; CHECK: uabalb z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uabalb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uabalb_h(<vscale x 4 x i32> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: uabalb_h:
; CHECK: uabalb z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uabalb.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uabalb_s(<vscale x 2 x i64> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: uabalb_s:
; CHECK: uabalb z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uabalb.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %out
}

;
; UABALT
;

define <vscale x 8 x i16> @uabalt_b(<vscale x 8 x i16> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: uabalt_b:
; CHECK: uabalt z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uabalt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uabalt_h(<vscale x 4 x i32> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: uabalt_h:
; CHECK: uabalt z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uabalt.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uabalt_s(<vscale x 2 x i64> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: uabalt_s:
; CHECK: uabalt z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uabalt.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %out
}

;
; UABDLB
;

define <vscale x 8 x i16> @uabdlb_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uabdlb_b:
; CHECK: uabdlb z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uabdlb.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uabdlb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uabdlb_h:
; CHECK: uabdlb z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uabdlb.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uabdlb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uabdlb_s:
; CHECK: uabdlb z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uabdlb.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; UABDLT
;

define <vscale x 8 x i16> @uabdlt_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uabdlt_b:
; CHECK: uabdlt z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uabdlt.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uabdlt_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uabdlt_h:
; CHECK: uabdlt z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uabdlt.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uabdlt_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uabdlt_s:
; CHECK: uabdlt z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uabdlt.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; UADDLB
;

define <vscale x 8 x i16> @uaddlb_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uaddlb_b:
; CHECK: uaddlb z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uaddlb.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uaddlb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uaddlb_h:
; CHECK: uaddlb z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uaddlb.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uaddlb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uaddlb_s:
; CHECK: uaddlb z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uaddlb.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; UADDLT
;

define <vscale x 8 x i16> @uaddlt_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uaddlt_b:
; CHECK: uaddlt z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uaddlt.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uaddlt_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uaddlt_h:
; CHECK: uaddlt z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uaddlt.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uaddlt_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uaddlt_s:
; CHECK: uaddlt z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uaddlt.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; UADDWB
;

define <vscale x 8 x i16> @uaddwb_b(<vscale x 8 x i16> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uaddwb_b:
; CHECK: uaddwb z0.h, z0.h, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uaddwb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uaddwb_h(<vscale x 4 x i32> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uaddwb_h:
; CHECK: uaddwb z0.s, z0.s, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uaddwb.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uaddwb_s(<vscale x 2 x i64> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uaddwb_s:
; CHECK: uaddwb z0.d, z0.d, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uaddwb.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; UADDWT
;

define <vscale x 8 x i16> @uaddwt_b(<vscale x 8 x i16> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uaddwt_b:
; CHECK: uaddwt z0.h, z0.h, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uaddwt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uaddwt_h(<vscale x 4 x i32> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uaddwt_h:
; CHECK: uaddwt z0.s, z0.s, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uaddwt.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uaddwt_s(<vscale x 2 x i64> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uaddwt_s:
; CHECK: uaddwt z0.d, z0.d, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uaddwt.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; UMULLB (Vectors)
;

define <vscale x 8 x i16> @umullb_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: umullb_b:
; CHECK: umullb z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.umullb.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @umullb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: umullb_h:
; CHECK: umullb z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.umullb.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @umullb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: umullb_s:
; CHECK: umullb z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.umullb.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; UMULLB (Indexed)
;

define <vscale x 4 x i32> @umullb_lane_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: umullb_lane_h:
; CHECK: umullb z0.s, z0.h, z1.h[0]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.umullb.lane.nxv4i32(<vscale x 8 x i16> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       i32 0)
  ret <vscale x 4 x i32> %out
}


define <vscale x 2 x i64> @umullb_lane_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: umullb_lane_s:
; CHECK: umullb z0.d, z0.s, z1.s[3]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.umullb.lane.nxv2i64(<vscale x 4 x i32> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       i32 3)
  ret <vscale x 2 x i64> %out
}

;
; UMULLT (Vectors)
;

define <vscale x 8 x i16> @umullt_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: umullt_b:
; CHECK: umullt z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.umullt.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @umullt_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: umullt_h:
; CHECK: umullt z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.umullt.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @umullt_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: umullt_s:
; CHECK: umullt z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.umullt.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; UMULLT (Indexed)
;

define <vscale x 4 x i32> @umullt_lane_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: umullt_lane_h:
; CHECK: umullt z0.s, z0.h, z1.h[1]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.umullt.lane.nxv4i32(<vscale x 8 x i16> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       i32 1)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @umullt_lane_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: umullt_lane_s:
; CHECK: umullt z0.d, z0.s, z1.s[2]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.umullt.lane.nxv2i64(<vscale x 4 x i32> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       i32 2)
  ret <vscale x 2 x i64> %out
}

;
; USHLLB
;

define <vscale x 8 x i16> @ushllb_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: ushllb_b:
; CHECK: ushllb z0.h, z0.b, #6
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.ushllb.nxv8i16(<vscale x 16 x i8> %a, i32 6)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @ushllb_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: ushllb_h:
; CHECK: ushllb z0.s, z0.h, #7
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.ushllb.nxv4i32(<vscale x 8 x i16> %a, i32 7)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @ushllb_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: ushllb_s:
; CHECK: ushllb z0.d, z0.s, #8
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.ushllb.nxv2i64(<vscale x 4 x i32> %a, i32 8)
  ret <vscale x 2 x i64> %out
}

;
; USHLLT
;

define <vscale x 8 x i16> @ushllt_b(<vscale x 16 x i8> %a) {
; CHECK-LABEL: ushllt_b:
; CHECK: ushllt z0.h, z0.b, #7
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.ushllt.nxv8i16(<vscale x 16 x i8> %a, i32 7)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @ushllt_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: ushllt_h:
; CHECK: ushllt z0.s, z0.h, #15
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.ushllt.nxv4i32(<vscale x 8 x i16> %a, i32 15)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @ushllt_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: ushllt_s:
; CHECK: ushllt z0.d, z0.s, #31
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.ushllt.nxv2i64(<vscale x 4 x i32> %a, i32 31)
  ret <vscale x 2 x i64> %out
}

;
; USUBLB
;

define <vscale x 8 x i16> @usublb_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: usublb_b:
; CHECK: usublb z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.usublb.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @usublb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: usublb_h:
; CHECK: usublb z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.usublb.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @usublb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: usublb_s:
; CHECK: usublb z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.usublb.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; USUBLT
;

define <vscale x 8 x i16> @usublt_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: usublt_b:
; CHECK: usublt z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.usublt.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @usublt_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: usublt_h:
; CHECK: usublt z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.usublt.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @usublt_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: usublt_s:
; CHECK: usublt z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.usublt.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; USUBWB
;

define <vscale x 8 x i16> @usubwb_b(<vscale x 8 x i16> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: usubwb_b:
; CHECK: usubwb z0.h, z0.h, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.usubwb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @usubwb_h(<vscale x 4 x i32> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: usubwb_h:
; CHECK: usubwb z0.s, z0.s, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.usubwb.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @usubwb_s(<vscale x 2 x i64> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: usubwb_s:
; CHECK: usubwb z0.d, z0.d, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.usubwb.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; USUBWT
;

define <vscale x 8 x i16> @usubwt_b(<vscale x 8 x i16> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: usubwt_b:
; CHECK: usubwt z0.h, z0.h, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.usubwt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @usubwt_h(<vscale x 4 x i32> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: usubwt_h:
; CHECK: usubwt z0.s, z0.s, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.usubwt.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @usubwt_s(<vscale x 2 x i64> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: usubwt_s:
; CHECK: usubwt z0.d, z0.d, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.usubwt.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.sabalb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sabalb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sabalb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sabalt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sabalt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sabalt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sabdlb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sabdlb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sabdlb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sabdlt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sabdlt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sabdlt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.saddlb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.saddlb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.saddlb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.saddlt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.saddlt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.saddlt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.saddwb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.saddwb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.saddwb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.saddwt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.saddwt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.saddwt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.smullb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smullb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smullb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.smullb.lane.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smullb.lane.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>, i32)

declare <vscale x 8 x i16> @llvm.aarch64.sve.smullt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smullt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smullt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.smullt.lane.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smullt.lane.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>, i32)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdmullb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullb.lane.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullb.lane.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>, i32)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdmullt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullt.lane.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullt.lane.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>, i32)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sshllb.nxv8i16(<vscale x 16 x i8>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sshllb.nxv4i32(<vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sshllb.nxv2i64(<vscale x 4 x i32>, i32)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sshllt.nxv8i16(<vscale x 16 x i8>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sshllt.nxv4i32(<vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sshllt.nxv2i64(<vscale x 4 x i32>, i32)

declare <vscale x 8 x i16> @llvm.aarch64.sve.ssublb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ssublb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ssublb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.ssublt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ssublt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ssublt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.ssubwb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ssubwb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ssubwb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.ssubwt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ssubwt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ssubwt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uabalb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uabalb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uabalb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uabalt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uabalt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uabalt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uabdlb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uabdlb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uabdlb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uabdlt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uabdlt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uabdlt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uaddlb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uaddlb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uaddlb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uaddlt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uaddlt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uaddlt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uaddwb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uaddwb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uaddwb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uaddwt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uaddwt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uaddwt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.umullb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umullb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umullb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.umullb.lane.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umullb.lane.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>, i32)

declare <vscale x 8 x i16> @llvm.aarch64.sve.umullt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umullt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umullt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.umullt.lane.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umullt.lane.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>, i32)

declare <vscale x 8 x i16> @llvm.aarch64.sve.ushllb.nxv8i16(<vscale x 16 x i8>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ushllb.nxv4i32(<vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ushllb.nxv2i64(<vscale x 4 x i32>, i32)

declare <vscale x 8 x i16> @llvm.aarch64.sve.ushllt.nxv8i16(<vscale x 16 x i8>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ushllt.nxv4i32(<vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ushllt.nxv2i64(<vscale x 4 x i32>, i32)

declare <vscale x 8 x i16> @llvm.aarch64.sve.usublb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.usublb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.usublb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.usublt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.usublt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.usublt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.usubwb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.usubwb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.usubwb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.usubwt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.usubwt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.usubwt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>)
