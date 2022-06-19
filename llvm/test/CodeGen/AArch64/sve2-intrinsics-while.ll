; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme < %s | FileCheck %s

;
; WHILEGE
;

define <vscale x 16 x i1> @whilege_b_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilege_b_ww:
; CHECK: whilege p0.b, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilege.nxv16i1.i32(i32 %a, i32 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @whilege_b_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilege_b_xx:
; CHECK: whilege p0.b, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilege.nxv16i1.i64(i64 %a, i64 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @whilege_h_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilege_h_ww:
; CHECK: whilege p0.h, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilege.nxv8i1.i32(i32 %a, i32 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @whilege_h_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilege_h_xx:
; CHECK: whilege p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilege.nxv8i1.i64(i64 %a, i64 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @whilege_s_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilege_s_ww:
; CHECK: whilege p0.s, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilege.nxv4i1.i32(i32 %a, i32 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @whilege_s_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilege_s_xx:
; CHECK: whilege p0.s, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilege.nxv4i1.i64(i64 %a, i64 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @whilege_d_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilege_d_ww:
; CHECK: whilege p0.d, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilege.nxv2i1.i32(i32 %a, i32 %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @whilege_d_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilege_d_xx:
; CHECK: whilege p0.d, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilege.nxv2i1.i64(i64 %a, i64 %b)
  ret <vscale x 2 x i1> %out
}

;
; WHILEHS
;

define <vscale x 16 x i1> @whilehs_b_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilehs_b_ww:
; CHECK: whilehs p0.b, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilehs.nxv16i1.i32(i32 %a, i32 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @whilehs_b_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilehs_b_xx:
; CHECK: whilehs p0.b, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilehs.nxv16i1.i64(i64 %a, i64 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @whilehs_h_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilehs_h_ww:
; CHECK: whilehs p0.h, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilehs.nxv8i1.i32(i32 %a, i32 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @whilehs_h_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilehs_h_xx:
; CHECK: whilehs p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilehs.nxv8i1.i64(i64 %a, i64 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @whilehs_s_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilehs_s_ww:
; CHECK: whilehs p0.s, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilehs.nxv4i1.i32(i32 %a, i32 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @whilehs_s_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilehs_s_xx:
; CHECK: whilehs p0.s, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilehs.nxv4i1.i64(i64 %a, i64 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @whilehs_d_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilehs_d_ww:
; CHECK: whilehs p0.d, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilehs.nxv2i1.i32(i32 %a, i32 %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @whilehs_d_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilehs_d_xx:
; CHECK: whilehs p0.d, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilehs.nxv2i1.i64(i64 %a, i64 %b)
  ret <vscale x 2 x i1> %out
}

;
; WHILEGT
;

define <vscale x 16 x i1> @whilegt_b_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilegt_b_ww:
; CHECK: whilegt p0.b, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilegt.nxv16i1.i32(i32 %a, i32 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @whilegt_b_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilegt_b_xx:
; CHECK: whilegt p0.b, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilegt.nxv16i1.i64(i64 %a, i64 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @whilegt_h_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilegt_h_ww:
; CHECK: whilegt p0.h, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilegt.nxv8i1.i32(i32 %a, i32 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @whilegt_h_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilegt_h_xx:
; CHECK: whilegt p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilegt.nxv8i1.i64(i64 %a, i64 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @whilegt_s_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilegt_s_ww:
; CHECK: whilegt p0.s, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilegt.nxv4i1.i32(i32 %a, i32 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @whilegt_s_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilegt_s_xx:
; CHECK: whilegt p0.s, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilegt.nxv4i1.i64(i64 %a, i64 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @whilegt_d_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilegt_d_ww:
; CHECK: whilegt p0.d, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilegt.nxv2i1.i32(i32 %a, i32 %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @whilegt_d_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilegt_d_xx:
; CHECK: whilegt p0.d, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilegt.nxv2i1.i64(i64 %a, i64 %b)
  ret <vscale x 2 x i1> %out
}

;
; WHILEHI
;

define <vscale x 16 x i1> @whilehi_b_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilehi_b_ww:
; CHECK: whilehi p0.b, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilehi.nxv16i1.i32(i32 %a, i32 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @whilehi_b_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilehi_b_xx:
; CHECK: whilehi p0.b, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilehi.nxv16i1.i64(i64 %a, i64 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @whilehi_h_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilehi_h_ww:
; CHECK: whilehi p0.h, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilehi.nxv8i1.i32(i32 %a, i32 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @whilehi_h_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilehi_h_xx:
; CHECK: whilehi p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilehi.nxv8i1.i64(i64 %a, i64 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @whilehi_s_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilehi_s_ww:
; CHECK: whilehi p0.s, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilehi.nxv4i1.i32(i32 %a, i32 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @whilehi_s_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilehi_s_xx:
; CHECK: whilehi p0.s, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilehi.nxv4i1.i64(i64 %a, i64 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @whilehi_d_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilehi_d_ww:
; CHECK: whilehi p0.d, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilehi.nxv2i1.i32(i32 %a, i32 %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @whilehi_d_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilehi_d_xx:
; CHECK: whilehi p0.d, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilehi.nxv2i1.i64(i64 %a, i64 %b)
  ret <vscale x 2 x i1> %out
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.whilege.nxv16i1.i32(i32, i32)
declare <vscale x 16 x i1> @llvm.aarch64.sve.whilege.nxv16i1.i64(i64, i64)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilege.nxv8i1.i32(i32, i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilege.nxv8i1.i64(i64, i64)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilege.nxv4i1.i32(i32, i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilege.nxv4i1.i64(i64, i64)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilege.nxv2i1.i32(i32, i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilege.nxv2i1.i64(i64, i64)

declare <vscale x 16 x i1> @llvm.aarch64.sve.whilehs.nxv16i1.i32(i32, i32)
declare <vscale x 16 x i1> @llvm.aarch64.sve.whilehs.nxv16i1.i64(i64, i64)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilehs.nxv8i1.i32(i32, i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilehs.nxv8i1.i64(i64, i64)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilehs.nxv4i1.i32(i32, i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilehs.nxv4i1.i64(i64, i64)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilehs.nxv2i1.i32(i32, i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilehs.nxv2i1.i64(i64, i64)

declare <vscale x 16 x i1> @llvm.aarch64.sve.whilegt.nxv16i1.i32(i32, i32)
declare <vscale x 16 x i1> @llvm.aarch64.sve.whilegt.nxv16i1.i64(i64, i64)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilegt.nxv8i1.i32(i32, i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilegt.nxv8i1.i64(i64, i64)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilegt.nxv4i1.i32(i32, i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilegt.nxv4i1.i64(i64, i64)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilegt.nxv2i1.i32(i32, i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilegt.nxv2i1.i64(i64, i64)

declare <vscale x 16 x i1> @llvm.aarch64.sve.whilehi.nxv16i1.i32(i32, i32)
declare <vscale x 16 x i1> @llvm.aarch64.sve.whilehi.nxv16i1.i64(i64, i64)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilehi.nxv8i1.i32(i32, i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilehi.nxv8i1.i64(i64, i64)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilehi.nxv4i1.i32(i32, i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilehi.nxv4i1.i64(i64, i64)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilehi.nxv2i1.i32(i32, i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilehi.nxv2i1.i64(i64, i64)
