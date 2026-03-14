; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

define <vscale x 4 x i32> @sdiv_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sdiv_i32:
; CHECK: sdiv z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sdiv.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sdiv_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sdiv_i64:
; CHECK: sdiv z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sdiv.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 4 x i32> @udiv_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: udiv_i32:
; CHECK: udiv z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.udiv.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @udiv_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: udiv_i64:
; CHECK: udiv z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.udiv.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 4 x i32> @sdivr_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sdivr_i32:
; CHECK: sdivr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sdivr.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sdivr_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sdivr_i64:
; CHECK: sdivr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sdivr.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 4 x i32> @udivr_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: udivr_i32:
; CHECK: udivr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.udivr.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @udivr_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: udivr_i64:
; CHECK: udivr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.udivr.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

declare <vscale x  4 x i32> @llvm.aarch64.sve.sdiv.nxv4i32(<vscale x  4 x i1>, <vscale x  4 x i32>, <vscale x  4 x i32>)
declare <vscale x  2 x i64> @llvm.aarch64.sve.sdiv.nxv2i64(<vscale x  2 x i1>, <vscale x  2 x i64>, <vscale x  2 x i64>)
declare <vscale x  4 x i32> @llvm.aarch64.sve.udiv.nxv4i32(<vscale x  4 x i1>, <vscale x  4 x i32>, <vscale x  4 x i32>)
declare <vscale x  2 x i64> @llvm.aarch64.sve.udiv.nxv2i64(<vscale x  2 x i1>, <vscale x  2 x i64>, <vscale x  2 x i64>)
declare <vscale x  4 x i32> @llvm.aarch64.sve.sdivr.nxv4i32(<vscale x  4 x i1>, <vscale x  4 x i32>, <vscale x  4 x i32>)
declare <vscale x  2 x i64> @llvm.aarch64.sve.sdivr.nxv2i64(<vscale x  2 x i1>, <vscale x  2 x i64>, <vscale x  2 x i64>)
declare <vscale x  4 x i32> @llvm.aarch64.sve.udivr.nxv4i32(<vscale x  4 x i1>, <vscale x  4 x i32>, <vscale x  4 x i32>)
declare <vscale x  2 x i64> @llvm.aarch64.sve.udivr.nxv2i64(<vscale x  2 x i1>, <vscale x  2 x i64>, <vscale x  2 x i64>)

