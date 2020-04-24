; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; SDIV
;

define <vscale x 4 x i32> @sdiv_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @sdiv_i32
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: sdiv z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %div = sdiv <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %div
}

define <vscale x 2 x i64> @sdiv_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: @sdiv_i64
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: sdiv z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %div = sdiv <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i64> %div
}

;
; UDIV
;

define <vscale x 4 x i32> @udiv_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @udiv_i32
; CHECK-DAG: ptrue p0.s
; CHECK-DAG: udiv z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %div = udiv <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %div
}

define <vscale x 2 x i64> @udiv_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: @udiv_i64
; CHECK-DAG: ptrue p0.d
; CHECK-DAG: udiv z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %div = udiv <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i64> %div
}
