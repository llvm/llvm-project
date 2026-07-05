; Test double atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define double @f1(ptr %src) {
; CHECK-LABEL: f1:
; CHECK: ld %f0, 0(%r2)
; CHECK: br %r14
  %val = load atomic double, ptr %src seq_cst, align 8
  ret double %val
}
