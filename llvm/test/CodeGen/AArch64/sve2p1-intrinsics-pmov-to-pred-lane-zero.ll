; RUN: llc -mattr=+sve2p1 < %s | FileCheck %s
; RUN: llc -mattr=+sme2p1 -force-streaming < %s | FileCheck %s

target triple = "aarch64"

define <vscale x 16 x i1> @pmov_b(<vscale x 16 x i8> %src) {
; CHECK-LABEL: pmov_b:
; CHECK:       pmov p0.b, z0
; CHECK-NEXT:  ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.pmov.to.pred.lane.zero.nxv16i8(<vscale x 16 x i8> %src)
  ret <vscale x 16 x i1> %res
}

define <vscale x 8 x i1> @pmov_h(<vscale x 8 x i16> %src) {
; CHECK-LABEL: pmov_h:
; CHECK:       pmov p0.h, z0[0]
; CHECK-NEXT:  ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.pmov.to.pred.lane.zero.nxv8i16(<vscale x 8 x i16> %src)
  ret <vscale x 8 x i1> %res
}

define <vscale x 4 x i1> @pmov_s(<vscale x 4 x i32> %src) {
; CHECK-LABEL: pmov_s:
; CHECK:       pmov p0.s, z0[0]
; CHECK-NEXT:  ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.pmov.to.pred.lane.zero.nxv4i32(<vscale x 4 x i32> %src)
  ret <vscale x 4 x i1> %res
}

define <vscale x 2 x i1> @pmov_d(<vscale x 2 x i64> %src) {
; CHECK-LABEL: pmov_d:
; CHECK:       pmov p0.d, z0[0]
; CHECK-NEXT:  ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.pmov.to.pred.lane.zero.nxv2i64(<vscale x 2 x i64> %src)
  ret <vscale x 2 x i1> %res
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.pmov.to.pred.lane.zero.nxv16i8(<vscale x 16 x i8>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.pmov.to.pred.lane.zero.nxv8i16(<vscale x 8 x i16>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.pmov.to.pred.lane.zero.nxv4i32(<vscale x 4 x i32>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.pmov.to.pred.lane.zero.nxv2i64(<vscale x 2 x i64>)
