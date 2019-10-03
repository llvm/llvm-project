; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

define <vscale x 8 x half> @fadd_h(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fadd_h:
; CHECK: fadd z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %res = fadd <vscale x 8 x half> %a, %b
  ret <vscale x 8 x half> %res
}

define <vscale x 4 x float> @fadd_s(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fadd_s:
; CHECK: fadd z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %res = fadd <vscale x 4 x float> %a, %b
  ret <vscale x 4 x float> %res
}

define <vscale x 2 x double> @fadd_d(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fadd_d:
; CHECK: fadd z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = fadd <vscale x 2 x double> %a, %b
  ret <vscale x 2 x double> %res
}
