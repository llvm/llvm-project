; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; PFIRST
;

define <vscale x 16 x i1> @pfirst_b8(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a) {
; CHECK-LABEL: pfirst_b8:
; CHECK: pfirst p1.b, p0, p1.b
; CHECK-NEXT: mov p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.pfirst.nxv16i1(<vscale x 16 x i1> %pg,
                                                                  <vscale x 16 x i1> %a)
  ret <vscale x 16 x i1> %out
}

;
; PNEXT
;

define <vscale x 16 x i1> @pnext_b8(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a) {
; CHECK-LABEL: pnext_b8:
; CHECK: pnext p1.b, p0, p1.b
; CHECK-NEXT: mov p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.pnext.nxv16i1(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i1> %a)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @pnext_b16(<vscale x 8 x i1> %pg, <vscale x 8 x i1> %a) {
; CHECK-LABEL: pnext_b16:
; CHECK: pnext p1.h, p0, p1.h
; CHECK-NEXT: mov p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.pnext.nxv8i1(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i1> %a)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @pnext_b32(<vscale x 4 x i1> %pg, <vscale x 4 x i1> %a) {
; CHECK-LABEL: pnext_b32:
; CHECK: pnext p1.s, p0, p1.s
; CHECK-NEXT: mov p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.pnext.nxv4i1(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i1> %a)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @pnext_b64(<vscale x 2 x i1> %pg, <vscale x 2 x i1> %a) {
; CHECK-LABEL: pnext_b64:
; CHECK: pnext p1.d, p0, p1.d
; CHECK-NEXT: mov p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.pnext.nxv2i1(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i1> %a)
  ret <vscale x 2 x i1> %out
}

;
; PUNPKHI
;

define <vscale x 8 x i1> @punpkhi_b16(<vscale x 16 x i1> %a) {
; CHECK-LABEL: punpkhi_b16
; CHECK: punpkhi p0.h, p0.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.punpkhi.nxv8i1(<vscale x 16 x i1> %a)
  ret <vscale x 8 x i1> %res
}

define <vscale x 4 x i1> @punpkhi_b8(<vscale x 8 x i1> %a) {
; CHECK-LABEL: punpkhi_b8
; CHECK: punpkhi p0.h, p0.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.punpkhi.nxv4i1(<vscale x 8 x i1> %a)
  ret <vscale x 4 x i1> %res
}

define <vscale x 2 x i1> @punpkhi_b4(<vscale x 4 x i1> %a) {
; CHECK-LABEL: punpkhi_b4
; CHECK: punpkhi p0.h, p0.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.punpkhi.nxv2i1(<vscale x 4 x i1> %a)
  ret <vscale x 2 x i1> %res
}

;
; PUNPKLO
;

define <vscale x 8 x i1> @punpklo_b16(<vscale x 16 x i1> %a) {
; CHECK-LABEL: punpklo_b16
; CHECK: punpklo p0.h, p0.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.punpklo.nxv8i1(<vscale x 16 x i1> %a)
  ret <vscale x 8 x i1> %res
}

define <vscale x 4 x i1> @punpklo_b8(<vscale x 8 x i1> %a) {
; CHECK-LABEL: punpklo_b8
; CHECK: punpklo p0.h, p0.b
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.punpklo.nxv4i1(<vscale x 8 x i1> %a)
  ret <vscale x 4 x i1> %res
}

define <vscale x 2 x i1> @punpklo_b4(<vscale x 4 x i1> %a) {
; CHECK-LABEL: punpklo_b4
; CHECK: punpklo p0.h, p0.b
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.punpklo.nxv2i1(<vscale x 4 x i1> %a)
  ret <vscale x 2 x i1> %res
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.pfirst.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.pnext.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.pnext.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.pnext.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.pnext.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>)

declare <vscale x 8 x i1> @llvm.aarch64.sve.punpkhi.nxv8i1(<vscale x 16 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.punpkhi.nxv4i1(<vscale x 8 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.punpkhi.nxv2i1(<vscale x 4 x i1>)

declare <vscale x 8 x i1> @llvm.aarch64.sve.punpklo.nxv8i1(<vscale x 16 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.punpklo.nxv4i1(<vscale x 8 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.punpklo.nxv2i1(<vscale x 4 x i1>)
