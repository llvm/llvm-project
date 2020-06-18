; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; Converting to svbool_t (<vscale x 16 x i1>)
;

define <vscale x 16 x i1> @reinterpret_bool_from_b(<vscale x 16 x i1> %pg) {
; CHECK-LABEL: reinterpret_bool_from_b:
; CHECK: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv16i1(<vscale x 16 x i1> %pg)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @reinterpret_bool_from_h(<vscale x 8 x i1> %pg) {
; CHECK-LABEL: reinterpret_bool_from_h:
; CHECK: ptrue p1.h
; CHECK-NEXT: ptrue p2.b
; CHECK-NEXT: and p0.b, p2/z, p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %pg)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @reinterpret_bool_from_s(<vscale x 4 x i1> %pg) {
; CHECK-LABEL: reinterpret_bool_from_s:
; CHECK: ptrue p1.s
; CHECK-NEXT: ptrue p2.b
; CHECK-NEXT: and p0.b, p2/z, p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %pg)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @reinterpret_bool_from_d(<vscale x 2 x i1> %pg) {
; CHECK-LABEL: reinterpret_bool_from_d:
; CHECK: ptrue p1.d
; CHECK-NEXT: ptrue p2.b
; CHECK-NEXT: and p0.b, p2/z, p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %pg)
  ret <vscale x 16 x i1> %out
}

;
; Converting from svbool_t
;

define <vscale x 16 x i1> @reinterpret_bool_to_b(<vscale x 16 x i1> %pg) {
; CHECK-LABEL: reinterpret_bool_to_b:
; CHECK: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv16i1(<vscale x 16 x i1> %pg)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @reinterpret_bool_to_h(<vscale x 16 x i1> %pg) {
; CHECK-LABEL: reinterpret_bool_to_h:
; CHECK: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @reinterpret_bool_to_s(<vscale x 16 x i1> %pg) {
; CHECK-LABEL: reinterpret_bool_to_s:
; CHECK: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @reinterpret_bool_to_d(<vscale x 16 x i1> %pg) {
; CHECK-LABEL: reinterpret_bool_to_d:
; CHECK: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  ret <vscale x 2 x i1> %out
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv16i1(<vscale x 16 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv16i1(<vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1>)
