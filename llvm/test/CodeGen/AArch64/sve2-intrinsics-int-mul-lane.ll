; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s

;
; MUL
;

define <vscale x 2 x i64> @mul_lane_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: mul_lane_d:
; CHECK: mul z0.d, z0.d, z1.d[1]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.mul.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                    <vscale x 2 x i64> %b,
                                                                    i32 1)
  ret <vscale x 2 x i64> %out
}

define <vscale x 4 x i32> @mul_lane_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: mul_lane_s:
; CHECK: mul z0.s, z0.s, z1.s[1]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.mul.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                    <vscale x 4 x i32> %b,
                                                                    i32 1)
  ret <vscale x 4 x i32> %out
}

define <vscale x 8 x i16> @mul_lane_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: mul_lane_h:
; CHECK: mul z0.h, z0.h, z1.h[1]
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.mul.lane.nxv8i16(<vscale x 8 x i16> %a,
                                                                    <vscale x 8 x i16> %b,
                                                                    i32 1)
  ret <vscale x 8 x i16> %out
}

;
; MLA
;

define <vscale x 2 x i64> @mla_lane_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: mla_lane_d:
; CHECK: mla z0.d, z1.d, z2.d[1]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.mla.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                    <vscale x 2 x i64> %b,
                                                                    <vscale x 2 x i64> %c,
                                                                    i32 1)
  ret <vscale x 2 x i64> %out
}

define <vscale x 4 x i32> @mla_lane_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: mla_lane_s:
; CHECK: mla z0.s, z1.s, z2.s[1]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.mla.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                    <vscale x 4 x i32> %b,
                                                                    <vscale x 4 x i32> %c,
                                                                    i32 1)
  ret <vscale x 4 x i32> %out
}

define <vscale x 8 x i16> @mla_lane_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: mla_lane_h:
; CHECK: mla z0.h, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.mla.lane.nxv8i16(<vscale x 8 x i16> %a,
                                                                    <vscale x 8 x i16> %b,
                                                                    <vscale x 8 x i16> %c,
                                                                    i32 1)
  ret <vscale x 8 x i16> %out
}

;
; MLS
;

define <vscale x 2 x i64> @mls_lane_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: mls_lane_d:
; CHECK: mls z0.d, z1.d, z2.d[1]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                    <vscale x 2 x i64> %b,
                                                                    <vscale x 2 x i64> %c,
                                                                    i32 1)
  ret <vscale x 2 x i64> %out
}

define <vscale x 4 x i32> @mls_lane_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: mls_lane_s:
; CHECK: mls z0.s, z1.s, z2.s[1]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                    <vscale x 4 x i32> %b,
                                                                    <vscale x 4 x i32> %c,
                                                                    i32 1)
  ret <vscale x 4 x i32> %out
}

define <vscale x 8 x i16> @mls_lane_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: mls_lane_h:
; CHECK: mls z0.h, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.lane.nxv8i16(<vscale x 8 x i16> %a,
                                                                    <vscale x 8 x i16> %b,
                                                                    <vscale x 8 x i16> %c,
                                                                    i32 1)
  ret <vscale x 8 x i16> %out
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.mul.lane.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.mul.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.mul.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.mla.lane.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.mla.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.mla.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.mls.lane.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.mls.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.mls.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, i32)
