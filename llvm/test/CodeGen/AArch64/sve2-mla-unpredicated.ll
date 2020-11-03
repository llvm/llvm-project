; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; SMLALB
;
define <vscale x 8 x i16> @smlalb_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: smlalb_i16
; CHECK: smlalb z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.smlalb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @smlalb_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: smlalb_i32
; CHECK: smlalb z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.smlalb.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @smlalb_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: smlalb_i64
; CHECK: smlalb z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.smlalb.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

;
; SMLALT
;
define <vscale x 8 x i16> @smlalt_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: smlalt_i16
; CHECK: smlalt z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.smlalt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @smlalt_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: smlalt_i32
; CHECK: smlalt z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.smlalt.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @smlalt_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: smlalt_i64
; CHECK: smlalt z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.smlalt.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

;
; UMLALB
;
define <vscale x 8 x i16> @umlalb_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: umlalb_i16
; CHECK: umlalb z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.umlalb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @umlalb_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: umlalb_i32
; CHECK: umlalb z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.umlalb.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @umlalb_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: umlalb_i64
; CHECK: umlalb z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.umlalb.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

;
; UMLALT
;
define <vscale x 8 x i16> @umlalt_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: umlalt_i16
; CHECK: umlalt z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.umlalt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @umlalt_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: umlalt_i32
; CHECK: umlalt z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.umlalt.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @umlalt_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: umlalt_i64
; CHECK: umlalt z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.umlalt.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

;
; SMLSLB
;
define <vscale x 8 x i16> @smlslb_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: smlslb_i16
; CHECK: smlslb z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.smlslb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @smlslb_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: smlslb_i32
; CHECK: smlslb z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.smlslb.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @smlslb_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: smlslb_i64
; CHECK: smlslb z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.smlslb.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

;
; SMLSLT
;
define <vscale x 8 x i16> @smlslt_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: smlslt_i16
; CHECK: smlslt z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.smlslt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @smlslt_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: smlslt_i32
; CHECK: smlslt z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.smlslt.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @smlslt_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: smlslt_i64
; CHECK: smlslt z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.smlslt.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

;
; UMLSLB
;
define <vscale x 8 x i16> @umlslb_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: umlslb_i16
; CHECK: umlslb z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.umlslb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @umlslb_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: umlslb_i32
; CHECK: umlslb z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.umlslb.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @umlslb_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: umlslb_i64
; CHECK: umlslb z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.umlslb.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

;
; UMLSLT
;
define <vscale x 8 x i16> @umlslt_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: umlslt_i16
; CHECK: umlslt z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.umlslt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @umlslt_i32(<vscale x 4 x i32> %a,
                                      <vscale x 8 x i16> %b,
                                      <vscale x 8 x i16> %c) {
; CHECK-LABEL: umlslt_i32
; CHECK: umlslt z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.umlslt.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @umlslt_i64(<vscale x 2 x i64> %a,
                                      <vscale x 4 x i32> %b,
                                      <vscale x 4 x i32> %c) {
; CHECK-LABEL: umlslt_i64
; CHECK: umlslt z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.umlslt.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

;
; SQDMLALB
;
define <vscale x 8 x i16> @sqdmlalb_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: sqdmlalb_i16
; CHECK: sqdmlalb z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdmlalb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sqdmlalb_i32(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlalb_i32
; CHECK: sqdmlalb z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlalb.nxv4i32(<vscale x 4 x i32> %a,
                                                                    <vscale x 8 x i16> %b,
                                                                    <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqdmlalb_i64(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlalb_i64
; CHECK: sqdmlalb z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlalb.nxv2i64(<vscale x 2 x i64> %a,
                                                                    <vscale x 4 x i32> %b,
                                                                    <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

;
; SQDMLALT
;
define <vscale x 8 x i16> @sqdmlalt_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: sqdmlalt_i16
; CHECK: sqdmlalt z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdmlalt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sqdmlalt_i32(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlalt_i32
; CHECK: sqdmlalt z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlalt.nxv4i32(<vscale x 4 x i32> %a,
                                                                    <vscale x 8 x i16> %b,
                                                                    <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqdmlalt_i64(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlalt_i64
; CHECK: sqdmlalt z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlalt.nxv2i64(<vscale x 2 x i64> %a,
                                                                    <vscale x 4 x i32> %b,
                                                                    <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

;
; SQDMLSLB
;
define <vscale x 8 x i16> @sqdmlslb_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: sqdmlslb_i16
; CHECK: sqdmlslb z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdmlslb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sqdmlslb_i32(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlslb_i32
; CHECK: sqdmlslb z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlslb.nxv4i32(<vscale x 4 x i32> %a,
                                                                    <vscale x 8 x i16> %b,
                                                                    <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqdmlslb_i64(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlslb_i64
; CHECK: sqdmlslb z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlslb.nxv2i64(<vscale x 2 x i64> %a,
                                                                    <vscale x 4 x i32> %b,
                                                                    <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

;
; SQDMLSLT
;
define <vscale x 8 x i16> @sqdmlslt_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: sqdmlslt_i16
; CHECK: sqdmlslt z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdmlslt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sqdmlslt_i32(<vscale x 4 x i32> %a,
                                        <vscale x 8 x i16> %b,
                                        <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlslt_i32
; CHECK: sqdmlslt z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlslt.nxv4i32(<vscale x 4 x i32> %a,
                                                                    <vscale x 8 x i16> %b,
                                                                    <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqdmlslt_i64(<vscale x 2 x i64> %a,
                                        <vscale x 4 x i32> %b,
                                        <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlslt_i64
; CHECK: sqdmlslt z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlslt.nxv2i64(<vscale x 2 x i64> %a,
                                                                    <vscale x 4 x i32> %b,
                                                                    <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

;
; SQDMLALBT
;
define <vscale x 8 x i16> @sqdmlalbt_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: sqdmlalbt_i16
; CHECK: sqdmlalbt z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdmlalbt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sqdmlalbt_i32(<vscale x 4 x i32> %a,
                                         <vscale x 8 x i16> %b,
                                         <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlalbt_i32
; CHECK: sqdmlalbt z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlalbt.nxv4i32(<vscale x 4 x i32> %a,
                                                                     <vscale x 8 x i16> %b,
                                                                     <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqdmlalbt_i64(<vscale x 2 x i64> %a,
                                         <vscale x 4 x i32> %b,
                                         <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlalbt_i64
; CHECK: sqdmlalbt z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlalbt.nxv2i64(<vscale x 2 x i64> %a,
                                                                     <vscale x 4 x i32> %b,
                                                                     <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

;
; SQDMLSLBT
;
define <vscale x 8 x i16> @sqdmlslbt_i16(<vscale x 8 x i16> %a,
                                      <vscale x 16 x i8> %b,
                                      <vscale x 16 x i8> %c) {
; CHECK-LABEL: sqdmlslbt_i16
; CHECK: sqdmlslbt z0.h, z1.b, z2.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdmlslbt.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sqdmlslbt_i32(<vscale x 4 x i32> %a,
                                         <vscale x 8 x i16> %b,
                                         <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqdmlslbt_i32
; CHECK: sqdmlslbt z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlslbt.nxv4i32(<vscale x 4 x i32> %a,
                                                                     <vscale x 8 x i16> %b,
                                                                     <vscale x 8 x i16> %c)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqdmlslbt_i64(<vscale x 2 x i64> %a,
                                         <vscale x 4 x i32> %b,
                                         <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqdmlslbt_i64
; CHECK: sqdmlslbt z0.d, z1.s, z2.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlslbt.nxv2i64(<vscale x 2 x i64> %a,
                                                                     <vscale x 4 x i32> %b,
                                                                     <vscale x 4 x i32> %c)
  ret <vscale x 2 x i64> %res
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.smlalb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smlalb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smlalb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.smlalt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smlalt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smlalt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.umlalb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umlalb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umlalb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.umlalt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umlalt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umlalt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.smlslb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smlslb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smlslb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.smlslt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.smlslt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.smlslt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.umlslb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umlslb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umlslb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.umlslt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.umlslt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.umlslt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdmlalb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlalb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlalb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdmlalt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlalt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlalt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdmlslb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlslb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlslb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdmlslt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlslt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlslt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdmlalbt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlalbt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlalbt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdmlslbt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmlslbt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmlslbt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
