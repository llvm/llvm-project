; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 -asm-verbose=0 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; HISTCNT
;

define <vscale x 4 x i32> @histcnt_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: histcnt_i32:
; CHECK: histcnt z0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.histcnt.nxv4i32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @histcnt_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: histcnt_i64:
; CHECK: histcnt z0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.histcnt.nxv2i64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x i64> %a,
                                                                   <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; HISTSEG
;

define <vscale x 16 x i8> @histseg(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: histseg:
; CHECK: histseg z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.histseg.nxv16i8(<vscale x 16 x i8> %a,
                                                                   <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

declare <vscale x 4 x i32> @llvm.aarch64.sve.histcnt.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.histcnt.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.histseg.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
