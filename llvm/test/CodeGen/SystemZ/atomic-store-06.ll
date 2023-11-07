; Test float atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(ptr %src, float %val) {
; CHECK-LABEL: f1:
; CHECK: lgdr [[R:%r[0-9]+]], %f0
; CHECK: srlg [[R]], [[R]], 32
; CHECK: st [[R]], 0(%r2)
; CHECK: br %r14
  store atomic float %val, ptr %src seq_cst, align 4
  ret void
}
