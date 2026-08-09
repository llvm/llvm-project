; RUN: llc -mtriple=aarch64 -mattr=+sve < %s | FileCheck %s

; --- Unsigned Division (UDIV) ---

define <vscale x 4 x i32> @fptoui_fdiv_uitofp_v4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: fptoui_fdiv_uitofp_v4i32:
; CHECK: udiv z{{[0-9]+}}.s, p{{[0-9]+}}/m, z{{[0-9]+}}.s, z{{[0-9]+}}.s
; CHECK: ret
  %fa = uitofp <vscale x 4 x i32> %a to <vscale x 4 x float>
  %fb = uitofp <vscale x 4 x i32> %b to <vscale x 4 x float>
  %fdiv = fdiv <vscale x 4 x float> %fa, %fb
  %res = fptoui <vscale x 4 x float> %fdiv to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @fptoui_fdiv_uitofp_v2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: fptoui_fdiv_uitofp_v2i64:
; CHECK: udiv z{{[0-9]+}}.d, p{{[0-9]+}}/m, z{{[0-9]+}}.d, z{{[0-9]+}}.d
; CHECK: ret
  %fa = uitofp <vscale x 2 x i64> %a to <vscale x 2 x double>
  %fb = uitofp <vscale x 2 x i64> %b to <vscale x 2 x double>
  %fdiv = fdiv <vscale x 2 x double> %fa, %fb
  %res = fptoui <vscale x 2 x double> %fdiv to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; --- Signed Division (SDIV) ---

define <vscale x 4 x i32> @fptosi_fdiv_sitofp_v4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: fptosi_fdiv_sitofp_v4i32:
; CHECK: sdiv z{{[0-9]+}}.s, p{{[0-9]+}}/m, z{{[0-9]+}}.s, z{{[0-9]+}}.s
; CHECK: ret
  %fa = sitofp <vscale x 4 x i32> %a to <vscale x 4 x float>
  %fb = sitofp <vscale x 4 x i32> %b to <vscale x 4 x float>
  %fdiv = fdiv <vscale x 4 x float> %fa, %fb
  %res = fptosi <vscale x 4 x float> %fdiv to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @fptosi_fdiv_sitofp_v2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: fptosi_fdiv_sitofp_v2i64:
; CHECK: sdiv z{{[0-9]+}}.d, p{{[0-9]+}}/m, z{{[0-9]+}}.d, z{{[0-9]+}}.d
; CHECK: ret
  %fa = sitofp <vscale x 2 x i64> %a to <vscale x 2 x double>
  %fb = sitofp <vscale x 2 x i64> %b to <vscale x 2 x double>
  %fdiv = fdiv <vscale x 2 x double> %fa, %fb
  %res = fptosi <vscale x 2 x double> %fdiv to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}