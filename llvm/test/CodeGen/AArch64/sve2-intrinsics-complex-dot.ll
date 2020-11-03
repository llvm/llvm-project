; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning


;
; CDOT
;

define <vscale x 4 x i32> @cdot_s(<vscale x 4 x i32> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: cdot_s:
; CHECK: cdot z0.s, z1.b, z2.b, #0
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.cdot.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 16 x i8> %b,
                                                                <vscale x 16 x i8> %c,
                                                                i32 0)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @cdot_d(<vscale x 2 x i64> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: cdot_d:
; CHECK: cdot z0.d, z1.h, z2.h, #90
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.cdot.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 8 x i16> %b,
                                                                <vscale x 8 x i16> %c,
                                                                i32 90)
  ret <vscale x 2 x i64> %out
}

;
; CDOT(indexed)
;

define <vscale x 4 x i32> @cdot_s_idx(<vscale x 4 x i32> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: cdot_s_idx:
; CHECK: cdot z0.s, z1.b, z2.b[0], #180
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.cdot.lane.nxv4i32(<vscale x 4 x i32> %a,
                                                                     <vscale x 16 x i8> %b,
                                                                     <vscale x 16 x i8> %c,
                                                                     i32 0, i32 180)
  ret <vscale x 4 x i32> %out
}


define <vscale x 2 x i64> @cdot_d_idx(<vscale x 2 x i64> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: cdot_d_idx:
; CHECK: cdot z0.d, z1.h, z2.h[1], #270
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.cdot.lane.nxv2i64(<vscale x 2 x i64> %a,
                                                                     <vscale x 8 x i16> %b,
                                                                     <vscale x 8 x i16> %c,
                                                                     i32 1, i32 270)
  ret <vscale x 2 x i64> %out
}


declare <vscale x 4 x i32> @llvm.aarch64.sve.cdot.nxv4i32(<vscale x 4 x i32>, <vscale x 16 x i8>, <vscale x 16 x i8>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.cdot.nxv2i64(<vscale x 2 x i64>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.cdot.lane.nxv4i32(<vscale x 4 x i32>, <vscale x 16 x i8>, <vscale x 16 x i8>, i32, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.cdot.lane.nxv2i64(<vscale x 2 x i64>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32, i32)
