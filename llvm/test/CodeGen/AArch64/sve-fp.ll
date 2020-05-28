; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

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

define <vscale x 8 x half> @fsub_h(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fsub_h:
; CHECK: fsub z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %res = fsub <vscale x 8 x half> %a, %b
  ret <vscale x 8 x half> %res
}

define <vscale x 4 x float> @fsub_s(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fsub_s:
; CHECK: fsub z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %res = fsub <vscale x 4 x float> %a, %b
  ret <vscale x 4 x float> %res
}

define <vscale x 2 x double> @fsub_d(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fsub_d:
; CHECK: fsub z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = fsub <vscale x 2 x double> %a, %b
  ret <vscale x 2 x double> %res
}

define <vscale x 8 x half> @fmul_h(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: fmul_h:
; CHECK: fmul z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %res = fmul <vscale x 8 x half> %a, %b
  ret <vscale x 8 x half> %res
}

define <vscale x 4 x float> @fmul_s(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: fmul_s:
; CHECK: fmul z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %res = fmul <vscale x 4 x float> %a, %b
  ret <vscale x 4 x float> %res
}

define <vscale x 2 x double> @fmul_d(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: fmul_d:
; CHECK: fmul z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = fmul <vscale x 2 x double> %a, %b
  ret <vscale x 2 x double> %res
}

define <vscale x 8 x half> @frecps_h(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: frecps_h:
; CHECK: frecps z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %res = call <vscale x 8 x half> @llvm.aarch64.sve.frecps.x.nxv8f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %res
}

define <vscale x 4 x float> @frecps_s(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: frecps_s:
; CHECK: frecps z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %res = call <vscale x 4 x float> @llvm.aarch64.sve.frecps.x.nxv4f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %res
}

define <vscale x 2 x double> @frecps_d(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: frecps_d:
; CHECK: frecps z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x double> @llvm.aarch64.sve.frecps.x.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %res
}

define <vscale x 8 x half> @frsqrts_h(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: frsqrts_h:
; CHECK: frsqrts z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %res = call <vscale x 8 x half> @llvm.aarch64.sve.frsqrts.x.nxv8f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %res
}

define <vscale x 4 x float> @frsqrts_s(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: frsqrts_s:
; CHECK: frsqrts z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %res = call <vscale x 4 x float> @llvm.aarch64.sve.frsqrts.x.nxv4f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %res
}

define <vscale x 2 x double> @frsqrts_d(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: frsqrts_d:
; CHECK: frsqrts z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x double> @llvm.aarch64.sve.frsqrts.x.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %res
}

declare <vscale x 8 x half> @llvm.aarch64.sve.frecps.x.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float>  @llvm.aarch64.sve.frecps.x.nxv4f32(<vscale x 4 x float> , <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.frecps.x.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x half> @llvm.aarch64.sve.frsqrts.x.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.frsqrts.x.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.frsqrts.x.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)
