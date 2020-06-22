; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -verify-machineinstrs < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

;
; ADRB
;

define <vscale x 4 x i32> @adrb_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: adrb_i32:
; CHECK: adr z0.s, [z0.s, z1.s]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.adrb.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @adrb_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: adrb_i64:
; CHECK: adr z0.d, [z0.d, z1.d]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.adrb.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; ADRH
;

define <vscale x 4 x i32> @adrh_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: adrh_i32:
; CHECK: adr z0.s, [z0.s, z1.s, lsl #1]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.adrh.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @adrh_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: adrh_i64:
; CHECK: adr z0.d, [z0.d, z1.d, lsl #1]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.adrh.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; ADRW
;

define <vscale x 4 x i32> @adrw_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: adrw_i32:
; CHECK: adr z0.s, [z0.s, z1.s, lsl #2]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.adrw.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @adrw_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: adrw_i64:
; CHECK: adr z0.d, [z0.d, z1.d, lsl #2]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.adrw.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; ADRD
;

define <vscale x 4 x i32> @adrd_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: adrd_i32:
; CHECK: adr z0.s, [z0.s, z1.s, lsl #3]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.adrd.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @adrd_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: adrd_i64:
; CHECK: adr z0.d, [z0.d, z1.d, lsl #3]
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.adrd.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 4 x i32> @llvm.aarch64.sve.adrb.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.adrb.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.adrh.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.adrh.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.adrw.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.adrw.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.adrd.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.adrd.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
