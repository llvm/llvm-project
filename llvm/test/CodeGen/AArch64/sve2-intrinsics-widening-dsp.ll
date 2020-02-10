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

declare <vscale x 8 x i16> @llvm.aarch64.sve.smullb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smullb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smullb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.smullt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smullt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smullt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdmullb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdmullt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.ssublb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ssublb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ssublb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.ssublt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ssublt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ssublt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

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

declare <vscale x 8 x i16> @llvm.aarch64.sve.umullb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umullb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umullb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.umullt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umullt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umullt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.usublb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.usublb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.usublb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.usublt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.usublt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.usublt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)
