; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

define float @fcvtxn(double %a) {
; CHECK-LABEL: fcvtxn:
; CHECK: fcvtxn s0, d0
; CHECK-NEXT: ret
  %vcvtxd.i = tail call float @llvm.aarch64.sisd.fcvtxn(double %a) nounwind
  ret float %vcvtxd.i
}

declare float @llvm.aarch64.sisd.fcvtxn(double) nounwind readnone
