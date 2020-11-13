; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; ADCLB (vector, long, unpredicated)
;
define <vscale x 4 x i32> @adclb_i32(<vscale x 4 x i32> %a,
                                     <vscale x 4 x i32> %b,
                                     <vscale x 4 x i32> %c) {
; CHECK-LABEL: adclb_i32
; CHECK: adclb z0.s, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.adclb.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @adclb_i64(<vscale x 2 x i64> %a,
                                     <vscale x 2 x i64> %b,
                                     <vscale x 2 x i64> %c) {
; CHECK-LABEL: adclb_i64
; CHECK: adclb z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.adclb.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %res
}

;
; ADCLT (vector, long, unpredicated)
;
define <vscale x 4 x i32> @adclt_i32(<vscale x 4 x i32> %a,
                                     <vscale x 4 x i32> %b,
                                     <vscale x 4 x i32> %c) {
; CHECK-LABEL: adclt_i32
; CHECK: adclt z0.s, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.adclt.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @adclt_i64(<vscale x 2 x i64> %a,
                                     <vscale x 2 x i64> %b,
                                     <vscale x 2 x i64> %c) {
; CHECK-LABEL: adclt_i64
; CHECK: adclt z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.adclt.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %res
}

;
; SBCLB (vector, long, unpredicated)
;
define <vscale x 4 x i32> @sbclb_i32(<vscale x 4 x i32> %a,
                                     <vscale x 4 x i32> %b,
                                     <vscale x 4 x i32> %c) {
; CHECK-LABEL: sbclb_i32
; CHECK: sbclb z0.s, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sbclb.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sbclb_i64(<vscale x 2 x i64> %a,
                                     <vscale x 2 x i64> %b,
                                     <vscale x 2 x i64> %c) {
; CHECK-LABEL: sbclb_i64
; CHECK: sbclb z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sbclb.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %res
}

;
; SBCLT (vector, long, unpredicated)
;
define <vscale x 4 x i32> @sbclt_i32(<vscale x 4 x i32> %a,
                                     <vscale x 4 x i32> %b,
                                     <vscale x 4 x i32> %c) {
; CHECK-LABEL: sbclt_i32
; CHECK: sbclt z0.s, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sbclt.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sbclt_i64(<vscale x 2 x i64> %a,
                                     <vscale x 2 x i64> %b,
                                     <vscale x 2 x i64> %c) {
; CHECK-LABEL: sbclt_i64
; CHECK: sbclt z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sbclt.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %res
}

declare <vscale x 4 x i32> @llvm.aarch64.sve.adclb.nxv4i32(<vscale x 4 x i32>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.adclb.nxv2i64(<vscale x 2 x i64>,<vscale x 2 x i64>,<vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.adclt.nxv4i32(<vscale x 4 x i32>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.adclt.nxv2i64(<vscale x 2 x i64>,<vscale x 2 x i64>,<vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sbclb.nxv4i32(<vscale x 4 x i32>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sbclb.nxv2i64(<vscale x 2 x i64>,<vscale x 2 x i64>,<vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sbclt.nxv4i32(<vscale x 4 x i32>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sbclt.nxv2i64(<vscale x 2 x i64>,<vscale x 2 x i64>,<vscale x 2 x i64>)
