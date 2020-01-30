; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s

;
; SMLALB
;
define <vscale x 4 x i32> @smlalb_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: smlalb_i32
; CHECK: smlalb z0.s, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.smlalb.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 1)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @smlalb_i32_2(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: smlalb_i32_2
; CHECK: smlalb z0.s, z1.h, z2.h[7]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.smlalb.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 7)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @smlalb_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: smlalb_i64
; CHECK: smlalb z0.d, z1.s, z2.s[0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.smlalb.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 0)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @smlalb_i64_2(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: smlalb_i64_2
; CHECK: smlalb z0.d, z1.s, z2.s[3]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.smlalb.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 3)
  ret <vscale x 2 x i64> %res
}

;
; SMLALT
;
define <vscale x 4 x i32> @smlalt_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: smlalt_i32
; CHECK: smlalt z0.s, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.smlalt.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 1)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @smlalt_i32_2(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: smlalt_i32_2
; CHECK: smlalt z0.s, z1.h, z2.h[7]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.smlalt.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 7)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @smlalt_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: smlalt_i64
; CHECK: smlalt z0.d, z1.s, z2.s[0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.smlalt.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 0)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @smlalt_i64_2(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: smlalt_i64_2
; CHECK: smlalt z0.d, z1.s, z2.s[3]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.smlalt.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 3)
  ret <vscale x 2 x i64> %res
}

;
; UMLALB
;
define <vscale x 4 x i32> @umlalb_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: umlalb_i32
; CHECK: umlalb z0.s, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.umlalb.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 1)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @umlalb_i32_2(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: umlalb_i32_2
; CHECK: umlalb z0.s, z1.h, z2.h[7]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.umlalb.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 7)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @umlalb_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: umlalb_i64
; CHECK: umlalb z0.d, z1.s, z2.s[0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.umlalb.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 0)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @umlalb_i64_2(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: umlalb_i64_2
; CHECK: umlalb z0.d, z1.s, z2.s[3]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.umlalb.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 3)
  ret <vscale x 2 x i64> %res
}

;
; UMLALT
;
define <vscale x 4 x i32> @umlalt_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: umlalt_i32
; CHECK: umlalt z0.s, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.umlalt.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 1)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @umlalt_i32_2(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: umlalt_i32_2
; CHECK: umlalt z0.s, z1.h, z2.h[7]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.umlalt.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 7)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @umlalt_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: umlalt_i64
; CHECK: umlalt z0.d, z1.s, z2.s[0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.umlalt.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 0)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @umlalt_i64_2(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: umlalt_i64_2
; CHECK: umlalt z0.d, z1.s, z2.s[3]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.umlalt.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 3)
  ret <vscale x 2 x i64> %res
}

;
; SMLSLB
;
define <vscale x 4 x i32> @smlslb_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: smlslb_i32
; CHECK: smlslb z0.s, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.smlslb.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 1)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @smlslb_i32_2(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: smlslb_i32_2
; CHECK: smlslb z0.s, z1.h, z2.h[7]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.smlslb.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 7)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @smlslb_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: smlslb_i64
; CHECK: smlslb z0.d, z1.s, z2.s[0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.smlslb.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 0)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @smlslb_i64_2(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: smlslb_i64_2
; CHECK: smlslb z0.d, z1.s, z2.s[3]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.smlslb.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 3)
  ret <vscale x 2 x i64> %res
}

;
; SMLSLT
;
define <vscale x 4 x i32> @smlslt_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: smlslt_i32
; CHECK: smlslt z0.s, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.smlslt.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 1)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @smlslt_i32_2(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: smlslt_i32_2
; CHECK: smlslt z0.s, z1.h, z2.h[7]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.smlslt.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 7)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @smlslt_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: smlslt_i64
; CHECK: smlslt z0.d, z1.s, z2.s[0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.smlslt.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 0)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @smlslt_i64_2(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: smlslt_i64_2
; CHECK: smlslt z0.d, z1.s, z2.s[3]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.smlslt.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 3)
  ret <vscale x 2 x i64> %res
}

;
; UMLSLB
;
define <vscale x 4 x i32> @umlslb_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: umlslb_i32
; CHECK: umlslb z0.s, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.umlslb.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 1)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @umlslb_i32_2(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: umlslb_i32_2
; CHECK: umlslb z0.s, z1.h, z2.h[7]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.umlslb.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 7)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @umlslb_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: umlslb_i64
; CHECK: umlslb z0.d, z1.s, z2.s[0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.umlslb.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 0)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @umlslb_i64_2(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: umlslb_i64_2
; CHECK: umlslb z0.d, z1.s, z2.s[3]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.umlslb.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 3)
  ret <vscale x 2 x i64> %res
}

;
; UMLSLT
;
define <vscale x 4 x i32> @umlslt_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: umlslt_i32
; CHECK: umlslt z0.s, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.umlslt.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 1)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @umlslt_i32_2(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: umlslt_i32_2
; CHECK: umlslt z0.s, z1.h, z2.h[7]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.umlslt.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 7)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @umlslt_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: umlslt_i64
; CHECK: umlslt z0.d, z1.s, z2.s[0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.umlslt.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 0)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @umlslt_i64_2(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: umlslt_i64_2
; CHECK: umlslt z0.d, z1.s, z2.s[3]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.umlslt.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 3)
  ret <vscale x 2 x i64> %res
}

;
; SQDMLALB
;
define <vscale x 4 x i32> @sqdmlalb_i32(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlalb_i32
; CHECK: sqdmlalb z0.s, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlalb.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                         <vscale x 8 x i16> %b,
                                                                         <vscale x 8 x i16> %c,
                                                                         i32 1)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @sqdmlalb_i32_2(<vscale x 4 x i32> %a,
                                          <vscale x 8 x i16> %b,
                                          <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlalb_i32_2
; CHECK: sqdmlalb z0.s, z1.h, z2.h[7]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlalb.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                         <vscale x 8 x i16> %b,
                                                                         <vscale x 8 x i16> %c,
                                                                         i32 7)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqdmlalb_i64(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlalb_i64
; CHECK: sqdmlalb z0.d, z1.s, z2.s[0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlalb.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                         <vscale x 4 x i32> %b,
                                                                         <vscale x 4 x i32> %c,
                                                                         i32 0)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @sqdmlalb_i64_2(<vscale x 2 x i64> %a,
                                          <vscale x 4 x i32> %b,
                                          <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlalb_i64_2
; CHECK: sqdmlalb z0.d, z1.s, z2.s[3]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlalb.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                         <vscale x 4 x i32> %b,
                                                                         <vscale x 4 x i32> %c,
                                                                         i32 3)
  ret <vscale x 2 x i64> %res
}

;
; SQDMLALT
;
define <vscale x 4 x i32> @sqdmlalt_i32(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlalt_i32
; CHECK: sqdmlalt z0.s, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlalt.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                         <vscale x 8 x i16> %b,
                                                                         <vscale x 8 x i16> %c,
                                                                         i32 1)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @sqdmlalt_i32_2(<vscale x 4 x i32> %a,
                                          <vscale x 8 x i16> %b,
                                          <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlalt_i32_2
; CHECK: sqdmlalt z0.s, z1.h, z2.h[7]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlalt.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                         <vscale x 8 x i16> %b,
                                                                         <vscale x 8 x i16> %c,
                                                                         i32 7)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqdmlalt_i64(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlalt_i64
; CHECK: sqdmlalt z0.d, z1.s, z2.s[0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlalt.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                         <vscale x 4 x i32> %b,
                                                                         <vscale x 4 x i32> %c,
                                                                         i32 0)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @sqdmlalt_i64_2(<vscale x 2 x i64> %a,
                                          <vscale x 4 x i32> %b,
                                          <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlalt_i64_2
; CHECK: sqdmlalt z0.d, z1.s, z2.s[3]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlalt.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                         <vscale x 4 x i32> %b,
                                                                         <vscale x 4 x i32> %c,
                                                                         i32 3)
  ret <vscale x 2 x i64> %res
}

;
; SQDMLSLB
;
define <vscale x 4 x i32> @sqdmlslb_i32(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlslb_i32
; CHECK: sqdmlslb z0.s, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlslb.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                         <vscale x 8 x i16> %b,
                                                                         <vscale x 8 x i16> %c,
                                                                         i32 1)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @sqdmlslb_i32_2(<vscale x 4 x i32> %a,
                                          <vscale x 8 x i16> %b,
                                          <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlslb_i32_2
; CHECK: sqdmlslb z0.s, z1.h, z2.h[7]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlslb.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                         <vscale x 8 x i16> %b,
                                                                         <vscale x 8 x i16> %c,
                                                                         i32 7)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqdmlslb_i64(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlslb_i64
; CHECK: sqdmlslb z0.d, z1.s, z2.s[0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlslb.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                         <vscale x 4 x i32> %b,
                                                                         <vscale x 4 x i32> %c,
                                                                         i32 0)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @sqdmlslb_i64_2(<vscale x 2 x i64> %a,
                                          <vscale x 4 x i32> %b,
                                          <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlslb_i64_2
; CHECK: sqdmlslb z0.d, z1.s, z2.s[3]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlslb.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                         <vscale x 4 x i32> %b,
                                                                         <vscale x 4 x i32> %c,
                                                                         i32 3)
  ret <vscale x 2 x i64> %res
}

;
; SQDMLSLT
;
define <vscale x 4 x i32> @sqdmlslt_i32(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlslt_i32
; CHECK: sqdmlslt z0.s, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlslt.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                         <vscale x 8 x i16> %b,
                                                                         <vscale x 8 x i16> %c,
                                                                         i32 1)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @sqdmlslt_i32_2(<vscale x 4 x i32> %a,
                                          <vscale x 8 x i16> %b,
                                          <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlslt_i32_2
; CHECK: sqdmlslt z0.s, z1.h, z2.h[7]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlslt.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                         <vscale x 8 x i16> %b,
                                                                         <vscale x 8 x i16> %c,
                                                                         i32 7)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqdmlslt_i64(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlslt_i64
; CHECK: sqdmlslt z0.d, z1.s, z2.s[0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlslt.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                         <vscale x 4 x i32> %b,
                                                                         <vscale x 4 x i32> %c,
                                                                         i32 0)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @sqdmlslt_i64_2(<vscale x 2 x i64> %a,
                                          <vscale x 4 x i32> %b,
                                          <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlslt_i64_2
; CHECK: sqdmlslt z0.d, z1.s, z2.s[3]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlslt.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                         <vscale x 4 x i32> %b,
                                                                         <vscale x 4 x i32> %c,
                                                                         i32 3)
  ret <vscale x 2 x i64> %res
}

declare <vscale x 4 x i32> @llvm.aarch64.sve.smlalb.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smlalb.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smlalt.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smlalt.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umlalb.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umlalb.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umlalt.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umlalt.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smlslb.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smlslb.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smlslt.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smlslt.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umlslb.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umlslb.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umlslt.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umlslt.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlalb.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlalb.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlalt.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlalt.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlslb.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlslb.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlslt.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlslt.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
