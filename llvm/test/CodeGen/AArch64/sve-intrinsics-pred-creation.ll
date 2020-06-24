; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

;
; PTRUE
;

define <vscale x 16 x i1> @ptrue_b8() {
; CHECK-LABEL: ptrue_b8:
; CHECK: ptrue p0.b, pow2
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 0)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @ptrue_b16() {
; CHECK-LABEL: ptrue_b16:
; CHECK: ptrue p0.h, vl1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 1)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @ptrue_b32() {
; CHECK-LABEL: ptrue_b32:
; CHECK: ptrue p0.s, mul3
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 30)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @ptrue_b64() {
; CHECK-LABEL: ptrue_b64:
; CHECK: ptrue p0.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  ret <vscale x 2 x i1> %out
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 %pattern)
declare <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 %pattern)
declare <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 %pattern)
declare <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 %pattern)
