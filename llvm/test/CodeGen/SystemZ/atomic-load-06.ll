; Test float atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define float @f1(ptr %src) {
; CHECK-LABEL: f1:
; CHECK: le %f0, 0(%r2)
; CHECK: br %r14
  %val = load atomic float, ptr %src seq_cst, align 4
  ret float %val
}
