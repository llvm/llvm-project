; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme < %s | FileCheck %s

;
; PTEST
;

define i1 @ptest_any(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a) {
; CHECK-LABEL: ptest_any:
; CHECK: ptest p0, p1.b
; CHECK-NEXT: cset w0, ne
; CHECK-NEXT: ret
  %out = call i1 @llvm.aarch64.sve.ptest.any(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a)
  ret i1 %out
}

define i1 @ptest_first(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a) {
; CHECK-LABEL: ptest_first:
; CHECK: ptest p0, p1.b
; CHECK-NEXT: cset w0, mi
; CHECK-NEXT: ret
  %out = call i1 @llvm.aarch64.sve.ptest.first(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a)
  ret i1 %out
}

define i1 @ptest_last(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a) {
; CHECK-LABEL: ptest_last:
; CHECK: ptest p0, p1.b
; CHECK-NEXT: cset w0, lo
; CHECK-NEXT: ret
  %out = call i1 @llvm.aarch64.sve.ptest.last(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a)
  ret i1 %out
}

declare i1 @llvm.aarch64.sve.ptest.any(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a)
declare i1 @llvm.aarch64.sve.ptest.first(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a)
declare i1 @llvm.aarch64.sve.ptest.last(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a)
