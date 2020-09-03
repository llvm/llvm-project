; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; WHILELE
;

define <vscale x 16 x i1> @whilele_b_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilele_b_ww:
; CHECK: whilele p0.b, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilele.nxv16i1.i32(i32 %a, i32 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @whilele_b_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilele_b_xx:
; CHECK: whilele p0.b, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilele.nxv16i1.i64(i64 %a, i64 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @whilele_h_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilele_h_ww:
; CHECK: whilele p0.h, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilele.nxv8i1.i32(i32 %a, i32 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @whilele_h_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilele_h_xx:
; CHECK: whilele p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilele.nxv8i1.i64(i64 %a, i64 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @whilele_s_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilele_s_ww:
; CHECK: whilele p0.s, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilele.nxv4i1.i32(i32 %a, i32 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @whilele_s_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilele_s_xx:
; CHECK: whilele p0.s, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilele.nxv4i1.i64(i64 %a, i64 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @whilele_d_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilele_d_ww:
; CHECK: whilele p0.d, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilele.nxv2i1.i32(i32 %a, i32 %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @whilele_d_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilele_d_xx:
; CHECK: whilele p0.d, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilele.nxv2i1.i64(i64 %a, i64 %b)
  ret <vscale x 2 x i1> %out
}

;
; WHILELO
;

define <vscale x 16 x i1> @whilelo_b_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilelo_b_ww:
; CHECK: whilelo p0.b, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilelo.nxv16i1.i32(i32 %a, i32 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @whilelo_b_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilelo_b_xx:
; CHECK: whilelo p0.b, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilelo.nxv16i1.i64(i64 %a, i64 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @whilelo_h_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilelo_h_ww:
; CHECK: whilelo p0.h, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilelo.nxv8i1.i32(i32 %a, i32 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @whilelo_h_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilelo_h_xx:
; CHECK: whilelo p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilelo.nxv8i1.i64(i64 %a, i64 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @whilelo_s_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilelo_s_ww:
; CHECK: whilelo p0.s, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilelo.nxv4i1.i32(i32 %a, i32 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @whilelo_s_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilelo_s_xx:
; CHECK: whilelo p0.s, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilelo.nxv4i1.i64(i64 %a, i64 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @whilelo_d_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilelo_d_ww:
; CHECK: whilelo p0.d, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilelo.nxv2i1.i32(i32 %a, i32 %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @whilelo_d_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilelo_d_xx:
; CHECK: whilelo p0.d, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilelo.nxv2i1.i64(i64 %a, i64 %b)
  ret <vscale x 2 x i1> %out
}

;
; WHILELS
;

define <vscale x 16 x i1> @whilels_b_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilels_b_ww:
; CHECK: whilels p0.b, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilels.nxv16i1.i32(i32 %a, i32 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @whilels_b_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilels_b_xx:
; CHECK: whilels p0.b, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilels.nxv16i1.i64(i64 %a, i64 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @whilels_h_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilels_h_ww:
; CHECK: whilels p0.h, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilels.nxv8i1.i32(i32 %a, i32 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @whilels_h_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilels_h_xx:
; CHECK: whilels p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilels.nxv8i1.i64(i64 %a, i64 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @whilels_s_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilels_s_ww:
; CHECK: whilels p0.s, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilels.nxv4i1.i32(i32 %a, i32 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @whilels_s_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilels_s_xx:
; CHECK: whilels p0.s, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilels.nxv4i1.i64(i64 %a, i64 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @whilels_d_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilels_d_ww:
; CHECK: whilels p0.d, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilels.nxv2i1.i32(i32 %a, i32 %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @whilels_d_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilels_d_xx:
; CHECK: whilels p0.d, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilels.nxv2i1.i64(i64 %a, i64 %b)
  ret <vscale x 2 x i1> %out
}

;
; WHILELT
;

define <vscale x 16 x i1> @whilelt_b_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilelt_b_ww:
; CHECK: whilelt p0.b, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilelt.nxv16i1.i32(i32 %a, i32 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @whilelt_b_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilelt_b_xx:
; CHECK: whilelt p0.b, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilelt.nxv16i1.i64(i64 %a, i64 %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @whilelt_h_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilelt_h_ww:
; CHECK: whilelt p0.h, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilelt.nxv8i1.i32(i32 %a, i32 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @whilelt_h_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilelt_h_xx:
; CHECK: whilelt p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilelt.nxv8i1.i64(i64 %a, i64 %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @whilelt_s_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilelt_s_ww:
; CHECK: whilelt p0.s, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilelt.nxv4i1.i32(i32 %a, i32 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 4 x i1> @whilelt_s_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilelt_s_xx:
; CHECK: whilelt p0.s, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilelt.nxv4i1.i64(i64 %a, i64 %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @whilelt_d_ww(i32 %a, i32 %b) {
; CHECK-LABEL: whilelt_d_ww:
; CHECK: whilelt p0.d, w0, w1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilelt.nxv2i1.i32(i32 %a, i32 %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 2 x i1> @whilelt_d_xx(i64 %a, i64 %b) {
; CHECK-LABEL: whilelt_d_xx:
; CHECK: whilelt p0.d, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilelt.nxv2i1.i64(i64 %a, i64 %b)
  ret <vscale x 2 x i1> %out
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.whilele.nxv16i1.i32(i32, i32)
declare <vscale x 16 x i1> @llvm.aarch64.sve.whilele.nxv16i1.i64(i64, i64)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilele.nxv8i1.i32(i32, i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilele.nxv8i1.i64(i64, i64)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilele.nxv4i1.i32(i32, i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilele.nxv4i1.i64(i64, i64)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilele.nxv2i1.i32(i32, i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilele.nxv2i1.i64(i64, i64)

declare <vscale x 16 x i1> @llvm.aarch64.sve.whilelo.nxv16i1.i32(i32, i32)
declare <vscale x 16 x i1> @llvm.aarch64.sve.whilelo.nxv16i1.i64(i64, i64)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilelo.nxv8i1.i32(i32, i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilelo.nxv8i1.i64(i64, i64)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilelo.nxv4i1.i32(i32, i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilelo.nxv4i1.i64(i64, i64)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilelo.nxv2i1.i32(i32, i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilelo.nxv2i1.i64(i64, i64)

declare <vscale x 16 x i1> @llvm.aarch64.sve.whilels.nxv16i1.i32(i32, i32)
declare <vscale x 16 x i1> @llvm.aarch64.sve.whilels.nxv16i1.i64(i64, i64)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilels.nxv8i1.i32(i32, i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilels.nxv8i1.i64(i64, i64)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilels.nxv4i1.i32(i32, i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilels.nxv4i1.i64(i64, i64)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilels.nxv2i1.i32(i32, i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilels.nxv2i1.i64(i64, i64)

declare <vscale x 16 x i1> @llvm.aarch64.sve.whilelt.nxv16i1.i32(i32, i32)
declare <vscale x 16 x i1> @llvm.aarch64.sve.whilelt.nxv16i1.i64(i64, i64)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilelt.nxv8i1.i32(i32, i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilelt.nxv8i1.i64(i64, i64)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilelt.nxv4i1.i32(i32, i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilelt.nxv4i1.i64(i64, i64)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilelt.nxv2i1.i32(i32, i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilelt.nxv2i1.i64(i64, i64)
