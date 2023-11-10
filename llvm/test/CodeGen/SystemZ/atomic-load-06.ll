; Test float atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define float @f1(ptr %src) {
; CHECK-LABEL: f1:
; CHECK: lgf [[R:%r[0-9]+]], 0(%r2)
; CHECK: sllg [[R]], [[R]], 32
; CHECK: ldgr %f0, [[R]]
; CHECK: br %r14
  %val = load atomic float, ptr %src seq_cst, align 4
  ret float %val
}
